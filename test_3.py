import mujoco.viewer
import time
from ikpy.chain import Chain
import transforms3d as tf
import numpy as np

URDF_PATH = "franka_emika_panda/panda.urdf"
_CHAIN_CACHE = None
IK_ACTIVE_MASK_PANDA_URDF = [False, True, True, True, True, True, True, True, False]


def get_chain_cached():
    """
    缓存 IK chain，避免每次 IK 都重新解析 URDF。
    """
    global _CHAIN_CACHE
    if _CHAIN_CACHE is None:
        chain = Chain.from_urdf_file(
            URDF_PATH,
            base_elements=["panda_link0"],
            active_links_mask=IK_ACTIVE_MASK_PANDA_URDF,
        )
        if len(chain.links) != len(IK_ACTIVE_MASK_PANDA_URDF):
            chain.active_links_mask = build_active_mask(chain)
        _CHAIN_CACHE = chain
    return _CHAIN_CACHE


def get_ee_contact_wrench_world(model, data, ee_body_ids, ee_pos_world):
    """
    计算末端接触外力/力矩（世界系）。
    返回 [Tx, Ty, Tz, Fx, Fy, Fz]，力矩参考点为 ee_pos_world。
    """
    net_force = np.zeros(3, dtype=np.float64)
    net_torque = np.zeros(3, dtype=np.float64)

    if data.ncon <= 0:
        return np.zeros(6, dtype=np.float64)

    for i in range(data.ncon):
        c = data.contact[i]
        b1 = int(model.geom_bodyid[int(c.geom1)])
        b2 = int(model.geom_bodyid[int(c.geom2)])

        # 过滤末端内部的自碰撞（如左手指碰右手指）
        if (b1 in ee_body_ids) and (b2 in ee_body_ids):
            continue

        # 如果两个物体都跟末端无关，跳过
        if (b1 not in ee_body_ids) and (b2 not in ee_body_ids):
            continue

        cf = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, i, cf)

        # 必须使用矩阵转置 .T，将接触坐标系正确转换到世界坐标系
        rot = c.frame.reshape(3, 3)
        f_world = rot.T @ cf[:3]   
        t_world = rot.T @ cf[3:6]  

        if b2 in ee_body_ids:
            sign = 1.0   # geom1 对 geom2 的力
        elif b1 in ee_body_ids:
            sign = -1.0  # 反作用力
        else:
            continue

        f_apply = sign * f_world
        t_apply = sign * t_world
        r = c.pos - ee_pos_world

        net_force += f_apply
        net_torque += t_apply + np.cross(r, f_apply)

    return np.concatenate([net_torque, net_force], axis=0)


def get_ee_cfrc_wrench_world(data, ee_body_ids):
    """
    读取 cfrc_ext 作为末端外力兜底。
    返回 [Tx, Ty, Tz, Fx, Fy, Fz]。
    """
    raw_sum = np.zeros(6, dtype=np.float64)
    for bid in ee_body_ids:
        raw_sum += data.cfrc_ext[bid]
    return np.concatenate([raw_sum[:3], raw_sum[3:]], axis=0)


def build_active_mask(chain):
    """
    自动生成 active_links_mask
    """
    mask = []
    for link in chain.links:
        jt = getattr(link, "joint_type", None)
        has_rot = bool(getattr(link, "has_rotation", False))
        has_trans = bool(getattr(link, "has_translation", False))
        inferred_fixed = (not has_rot) and (not has_trans)
        if jt == "fixed" or inferred_fixed:
            mask.append(False)
        else:
            mask.append(True)

    force_off_names = {"Base link", "tcp_fixed", "base", "robot_base", "panda_joint8"}
    for i, link in enumerate(chain.links):
        if link.name in force_off_names:
            mask[i] = False

    if not any(mask):
        raise ValueError("active_links_mask 全是 False，检查 URDF 关节类型解析是否正常。")
    return mask

def solve_ik_xyz_down(
    target_xyz,
    yaw_deg=-45.0,
    initial_active=None,
    orient_axis="Z",
    axis_down=np.array([0.0, 0.0, -1.0]),
    max_attempts=3,
    chain=None,
    target_rotmat=None,
    max_orientation_error_rad=0.08,
    lock_last_joint=False,
):
    if chain is None:
        chain = get_chain_cached()

    n_links = len(chain.links)
    mask_bool = np.array(chain.active_links_mask, dtype=bool)
    n_active = int(mask_bool.sum())

    q0_full = np.zeros(n_links, dtype=float)
    if initial_active is not None:
        initial_active = np.asarray(initial_active, dtype=float).reshape(-1)
        if initial_active.size != n_active:
            raise ValueError(f"initial_active 维度={initial_active.size}，但当前 active 关节数={n_active}。")
        q0_full[mask_bool] = initial_active

    best_q_full = None
    best_score = np.inf
    best_pos_err = np.inf
    best_rot_err = np.inf

    for attempt in range(max_attempts):
        if attempt == 0:
            q_init = q0_full.copy()
        else:
            q_init = q0_full.copy()
            noise = np.zeros_like(q_init)
            noise[mask_bool] = np.random.uniform(-0.15, 0.15, size=n_active)
            q_init = q_init + noise

        for i, link in enumerate(chain.links):
            lo, hi = getattr(link, "bounds", (-np.inf, np.inf))
            if np.isfinite(lo) or np.isfinite(hi):
                q_init[i] = np.clip(q_init[i], lo, hi)

        # 优先使用 6D 完整姿态 IK
        if target_rotmat is not None:
            frame_target = np.eye(4, dtype=float)
            frame_target[:3, :3] = np.asarray(target_rotmat, dtype=float)
            frame_target[:3, 3] = np.asarray(target_xyz, dtype=float)
            q_full = chain.inverse_kinematics_frame(
                target=frame_target,
                initial_position=q_init,
            )
        else:
            q_full = chain.inverse_kinematics(
                target_position=np.asarray(target_xyz, dtype=float),
                target_orientation=np.asarray(axis_down, dtype=float),
                orientation_mode=orient_axis,
                initial_position=q_init,
            )

        for i, link in enumerate(chain.links):
            lo, hi = getattr(link, "bounds", (-np.inf, np.inf))
            if np.isfinite(lo) or np.isfinite(hi):
                q_full[i] = np.clip(q_full[i], lo, hi)

        fk = chain.forward_kinematics(q_full)
        ee_xyz = fk[:3, 3]
        pos_err = float(np.linalg.norm(ee_xyz - np.asarray(target_xyz, dtype=float)))
        rot_err = 0.0
        if target_rotmat is not None:
            r_now = fk[:3, :3]
            dr = np.asarray(target_rotmat, dtype=float).T @ r_now
            cos_th = np.clip((np.trace(dr) - 1.0) * 0.5, -1.0, 1.0)
            rot_err = float(np.arccos(cos_th))

        score = pos_err + 0.35 * rot_err

        if score < best_score:
            best_score = score
            best_pos_err = pos_err
            best_rot_err = rot_err
            best_q_full = q_full

    q_full = best_q_full
    q_active = q_full[mask_bool].copy()

    if (target_rotmat is not None) and (initial_active is not None) and (best_rot_err > max_orientation_error_rad):
        q_active = np.asarray(initial_active, dtype=float).copy()
        q_full = q0_full.copy()

    if lock_last_joint and (initial_active is not None) and (q_active.size > 0):
        q_active[-1] = float(np.asarray(initial_active, dtype=float)[-1])
        q_full[mask_bool] = q_active

    if q_active.size > 0 and yaw_deg is not None:
        q_active[-1] += np.deg2rad(yaw_deg)

    return q_active, q_full, chain

def main():
    model = mujoco.MjModel.from_xml_path('franka_emika_panda/scene_wrench_test.xml')
    data = mujoco.MjData(model)
    
    if model.nkey > 0:
        data.qpos[:model.nq] = model.key_qpos[0, :model.nq]
        data.ctrl[:model.nu] = model.key_ctrl[0, :model.nu]
    mujoco.mj_forward(model, data)

    hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    left_finger_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_finger_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    ee_body_ids = {hand_body_id, left_finger_body_id, right_finger_body_id}

    peg_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "peg")
    initial_pose = data.ctrl[:7].copy()

    chain = get_chain_cached()
    n_links = len(chain.links)
    mask_bool = np.array(chain.active_links_mask, dtype=bool)
    q_full0 = np.zeros(n_links)
    q_full0[mask_bool] = initial_pose
    fk0 = chain.forward_kinematics(q_full0)
    
    start_pos = fk0[:3, 3]
    start_rot = fk0[:3, :3].copy() # 【关键提取】获取初始的完整 3x3 旋转矩阵

    # --- 轨迹参数调整 ---
    step_num = 900 
    distance_m = -0.45
    fixed_axis_down = np.array([0.0, 0.0, -1.0], dtype=float)
    fixed_axis_down = fixed_axis_down / (np.linalg.norm(fixed_axis_down) + 1e-9)
    current_active = initial_pose.copy()
    
    touch_down_fz_threshold = 4.0  
    substeps_per_waypoint = 2       

    print("Start online vertical trajectory...")
    with mujoco.viewer.launch_passive(model, data) as viewer:

        print(f"[init] peg world pos = {data.xpos[peg_body_id].copy()}")
        for i in range(1, step_num + 1):
            alpha = i / step_num
            target_pos = start_pos.copy()
            target_pos[2] += alpha * distance_m

            q, _, _ = solve_ik_xyz_down(
                target_pos,
                yaw_deg=None,
                initial_active=current_active,
                orient_axis="Z",
                axis_down=fixed_axis_down,
                max_attempts=3,                # 因为要约束完整的 6D，稍微增加 IK 尝试次数
                chain=chain,
                target_rotmat=start_rot,       # 【关键应用】锁定完整姿态！
                lock_last_joint=True,
            )
            current_active = q.copy()

            arm_dof = min(7, len(q), model.nu)
            data.ctrl[:arm_dof] = q[:arm_dof]
            
            # 【夹紧优化】使用一个更小的负数确保手爪死死咬住方块，克服触地或下落的惯性
            data.ctrl[7] = -0.05 

            touchdown_detected = False

            # --- 高频碰撞检测 (Guarded Move) ---
            for _ in range(substeps_per_waypoint):
                mujoco.mj_step(model, data)
                viewer.sync()
                # time.sleep(model.opt.timestep) 

                ee_pos = data.site(ee_site_id).xpos.copy()
                wrench_contact = get_ee_contact_wrench_world(model, data, ee_body_ids, ee_pos)
                
                # 一旦 Fz 大于阈值，立刻中断
                if wrench_contact[5] > touch_down_fz_threshold:
                    print(f"\n[TOUCHDOWN DETECTED] Immediate response! Step={i:03d}, Fz={wrench_contact[5]:+.4f}N")
                    touchdown_detected = True
                    break

            if touchdown_detected:
                # 冻结机械臂当前位姿
                data.ctrl[:arm_dof] = data.qpos[:arm_dof].copy()
                break 

            # 日志打印
            if (i % 50 == 0) or (i == 1):
                wrench_cfrc = get_ee_cfrc_wrench_world(data, ee_body_ids)
                if np.linalg.norm(wrench_contact[3:]) > 1e-6:
                    wrench = wrench_contact
                    src = "contact"
                else:
                    wrench = wrench_cfrc
                    src = "cfrc_ext"
                
                tx, ty, tz, fx, fy, fz = wrench
                print(f"[step {i:03d}] peg world pos = {data.xpos[peg_body_id].copy()}")
                print(
                    f"          ee wrench({src}) [Tx Ty Tz Fx Fy Fz] = "
                    f"[{tx:+.4f}, {ty:+.4f}, {tz:+.4f}, {fx:+.4f}, {fy:+.4f}, {fz:+.4f}]"
                )

        print("\nTrajectory finished or stopped. Holding position...")
        
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

if __name__ == "__main__":
    main()