import mujoco.viewer
import time
from ikpy.chain import Chain
import transforms3d as tf
import numpy as np

URDF_PATH = "franka_emika_panda/panda.urdf"
_CHAIN_CACHE = None


def get_chain_cached():
    """
    缓存 IK chain，避免每步 IK 都重复解析 URDF。
    """
    global _CHAIN_CACHE
    if _CHAIN_CACHE is None:
        chain = Chain.from_urdf_file(URDF_PATH, base_elements=["panda_link0"])
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

        # 【修正 2】如果两个物体都在末端集合内，说明是内部作用力（互相抵消），直接跳过
        if (b1 in ee_body_ids) and (b2 in ee_body_ids):
            continue

        # 如果两个物体都跟末端无关，跳过
        if (b1 not in ee_body_ids) and (b2 not in ee_body_ids):
            continue

        cf = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, i, cf)

        # 【修正 1】矩阵必须转置，才能正确地将接触系转换到世界系
        rot = c.frame.reshape(3, 3)
        f_world = rot.T @ cf[:3]   # 注意这里加了 .T
        t_world = rot.T @ cf[3:6]  # 注意这里加了 .T

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
    自动生成 active_links_mask：
    - fixed joint 一律 False
    - 其余默认 True（即 revolute/prismatic）
    """
    mask = []
    for link in chain.links:
        jt = getattr(link, "joint_type", None)
        has_rot = bool(getattr(link, "has_rotation", False))
        has_trans = bool(getattr(link, "has_translation", False))
        inferred_fixed = (not has_rot) and (not has_trans)
        # ikpy 的 fixed joint 就别让它 active
        if jt == "fixed" or inferred_fixed:
            mask.append(False)
        else:
            mask.append(True)

    # 额外保险：如果你想强制按名字关掉一些 link（比如 base/tcp_fixed）
    # （即使它们 joint_type 解析不准，也能兜底）
    force_off_names = {"Base link", "tcp_fixed", "base", "robot_base", "panda_joint8"}
    for i, link in enumerate(chain.links):
        if link.name in force_off_names:
            mask[i] = False

    # 兜底：至少要有 1 个 active
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
):
    """
    - target_xyz: 世界坐标系下的目标点
    - initial_active: 6 维（active 关节）初值
    - orient_axis: "X"/"Y"/"Z" 表示让末端哪个轴对齐 axis_down
    """
    if chain is None:
        chain = get_chain_cached()

    n_links = len(chain.links)
    mask_bool = np.array(chain.active_links_mask, dtype=bool)
    n_active = int(mask_bool.sum())

    # 构造 full 初值（长度 = n_links），再把 active 的部分填进去
    q0_full = np.zeros(n_links, dtype=float)
    if initial_active is not None:
        initial_active = np.asarray(initial_active, dtype=float).reshape(-1)
        if initial_active.size != n_active:
            raise ValueError(f"initial_active 维度={initial_active.size}，但当前 active 关节数={n_active}。")
        q0_full[mask_bool] = initial_active

    # 多次尝试：如果初值不好，可在 active 维度上做轻微扰动重试
    best_q_full = None
    best_err = np.inf

    for attempt in range(max_attempts):
        if attempt == 0:
            q_init = q0_full.copy()
        else:
            q_init = q0_full.copy()
            # 只扰动 active 关节，幅度小一点
            noise = np.zeros_like(q_init)
            noise[mask_bool] = np.random.uniform(-0.15, 0.15, size=n_active)
            q_init = q_init + noise

        # 先把初值夹到 bounds，避免 IK 报 Initial guess is outside of provided bounds
        for i, link in enumerate(chain.links):
            lo, hi = getattr(link, "bounds", (-np.inf, np.inf))
            if np.isfinite(lo) or np.isfinite(hi):
                q_init[i] = np.clip(q_init[i], lo, hi)

        q_full = chain.inverse_kinematics(
            target_position=np.asarray(target_xyz, dtype=float),
            target_orientation=np.asarray(axis_down, dtype=float),
            orientation_mode=orient_axis,
            initial_position=q_init,
        )

        # 夹紧到 bounds（未来换成有限关节范围时很关键）
        for i, link in enumerate(chain.links):
            lo, hi = getattr(link, "bounds", (-np.inf, np.inf))
            if np.isfinite(lo) or np.isfinite(hi):
                q_full[i] = np.clip(q_full[i], lo, hi)

        # 用正运动学算一下位置误差，挑最好的那次
        fk = chain.forward_kinematics(q_full)
        ee_xyz = fk[:3, 3]
        err = float(np.linalg.norm(ee_xyz - np.asarray(target_xyz, dtype=float)))

        if err < best_err:
            best_err = err
            best_q_full = q_full

    q_full = best_q_full
    q_active = q_full[mask_bool].copy()

    # 你原来用“最后一个 active 关节 + yaw”做末端偏航，这里保留
    if q_active.size > 0 and yaw_deg is not None:
        q_active[-1] += np.deg2rad(yaw_deg)

    # 调试信息（建议你先跑一次确认）
    # print("active mask:", list(zip(range(n_links), [l.name for l in chain.links], chain.active_links_mask)))
    # print("ik pos err:", best_err)

    return q_active, q_full, chain

def generate_vertical_trajectory(
    start_q_active,
    distance_m,
    step_num=50,
    orient_axis="Z",
    fixed_axis_down=np.array([0.0, 0.0, -1.0]),
):
    """
    生成一段竖直升降的关节轨迹。
    start_q_active: 起始姿态的 active 关节角度
    distance_m: 移动距离（米），正向上，负向下
    step_num: 插值点数量
    """
    # 1. 计算起始末端位置 (FK)
    chain = get_chain_cached()
    
    fixed_axis_down = np.asarray(fixed_axis_down, dtype=float)
    fixed_axis_down = fixed_axis_down / (np.linalg.norm(fixed_axis_down) + 1e-9)

    n_links = len(chain.links)
    mask_bool = np.array(chain.active_links_mask, dtype=bool)
    
    current_q_full = np.zeros(n_links)
    current_q_full[mask_bool] = start_q_active
    
    fk = chain.forward_kinematics(current_q_full)
    start_pos = fk[:3, 3] # XYZ
    
    path_q = []
    current_active = np.array(start_q_active, dtype=float)

    # 2. 循环生成路径
    for i in range(1, step_num + 1):
        # 线性插值目标 Z 高度
        delta = (i / step_num) * distance_m
        target_pos = start_pos.copy()
        target_pos[2] += delta # Z 轴加减
        
        # 调用 IK，并不再叠加额外 yaw，依赖上一帧初值保持连贯
        q_sol, _, _ = solve_ik_xyz_down(
            target_pos, 
            yaw_deg=None,  # 关键：不再叠加旋转
            initial_active=current_active, 
            orient_axis=orient_axis,
            axis_down=fixed_axis_down,
            max_attempts=2,
            chain=chain,
        )
        
        path_q.append(q_sol)
        current_active = q_sol # 更新 seeded guess
        
    return path_q

def generate_pose_trajectory(start_q_active, target_xyz, target_axis_down=None, orient_axis="Y", step_num=50, max_attempts=3):
    """
    从当前末端位姿平滑移动到给定目标位姿（位置 + 轴向），返回 active 关节角轨迹列表。

    参数:
    - start_q_active: 起始 active 关节向量
    - target_xyz: 目标末端位置 (3,)
    - target_axis_down: 目标末端轴方向向量 (3,)（若为 None 则保持起始朝向）
    - orient_axis: 在 IK 中表示末端哪个轴对齐 target_axis_down，取值 'X'/'Y'/'Z'
    - step_num: 插值步数
    - max_attempts: 每步 IK 的最大尝试次数

    说明:
    - 为了避免关节突变，每步 IK 使用上一帧求解结果作为初值。
    - 通过线性插值位置与轴向并对轴向归一化来实现平滑过渡。
    """
    chain = get_chain_cached()

    n_links = len(chain.links)
    mask_bool = np.array(chain.active_links_mask, dtype=bool)

    # 起始全关节向量
    current_q_full = np.zeros(n_links)
    current_q_full[mask_bool] = start_q_active

    fk = chain.forward_kinematics(current_q_full)
    start_pos = fk[:3, 3]
    rot = fk[:3, :3]

    if orient_axis == "X":
        start_axis = rot[:, 0]
    elif orient_axis == "Y":
        start_axis = rot[:, 1]
    else:
        start_axis = rot[:, 2]

    if target_axis_down is None:
        target_axis = start_axis.copy()
    else:
        target_axis = np.asarray(target_axis_down, dtype=float)
        norm = np.linalg.norm(target_axis)
        if norm < 1e-8:
            target_axis = start_axis.copy()
        else:
            target_axis = target_axis / norm

    path_q = []
    current_active = np.array(start_q_active, dtype=float)

    for i in range(1, step_num + 1):
        alpha = float(i) / step_num
        interp_pos = (1 - alpha) * start_pos + alpha * np.asarray(target_xyz, dtype=float)
        interp_axis = (1 - alpha) * start_axis + alpha * target_axis
        interp_axis = interp_axis / (np.linalg.norm(interp_axis) + 1e-9)

        q_sol, _, _ = solve_ik_xyz_down(
            interp_pos,
            yaw_deg=None,
            initial_active=current_active,
            orient_axis=orient_axis,
            axis_down=interp_axis,
            max_attempts=max_attempts,
            chain=chain,
        )

        path_q.append(q_sol)
        current_active = q_sol

    return path_q
 
def main():
    model = mujoco.MjModel.from_xml_path('franka_emika_panda/scene_wrench_test.xml')
    data = mujoco.MjData(model)
    # 加载关键帧（MuJoCo Python API 里使用 key_qpos/key_ctrl 数组）
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

    # 在线轨迹：避免先整段预规划等待过久。
    step_num = 200
    distance_m = -0.45
    orient_axis = "Z"
    fixed_axis_down = np.array([0.0, 0.0, -1.0], dtype=float)
    fixed_axis_down = fixed_axis_down / (np.linalg.norm(fixed_axis_down) + 1e-9)
    print_every_n = 5
    substeps_per_waypoint = 4
    realtime_scale = 1.0

    chain = get_chain_cached()
    mask_bool = np.array(chain.active_links_mask, dtype=bool)
    q0_full = np.zeros(len(chain.links), dtype=float)
    q0_full[mask_bool] = initial_pose
    start_pos = chain.forward_kinematics(q0_full)[:3, 3]
    current_active = initial_pose.copy()

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
                orient_axis=orient_axis,
                axis_down=fixed_axis_down,
                max_attempts=2,
                chain=chain,
            )
            current_active = q.copy()

            arm_dof = min(7, len(q), model.nu)
            data.ctrl[:arm_dof] = q[:arm_dof]
            data.ctrl[7] = 0.0

            for _ in range(substeps_per_waypoint):
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep * realtime_scale)

            ee_pos = data.site(ee_site_id).xpos.copy()
            wrench_contact = get_ee_contact_wrench_world(model, data, ee_body_ids, ee_pos)
            wrench_cfrc = get_ee_cfrc_wrench_world(data, ee_body_ids)
            if np.linalg.norm(wrench_contact[3:]) > 1e-6:
                wrench = wrench_contact
                src = "contact"
            else:
                wrench = wrench_cfrc
                src = "cfrc_ext"

            if (i % print_every_n == 0) or (i == 1) or (i == step_num):
                tx, ty, tz, fx, fy, fz = wrench
                print(f"[step {i:03d}] peg world pos = {data.xpos[peg_body_id].copy()}")
                print(
                    f"          ee wrench({src}) [Tx Ty Tz Fx Fy Fz] = "
                    f"[{tx:+.4f}, {ty:+.4f}, {tz:+.4f}, {fx:+.4f}, {fy:+.4f}, {fz:+.4f}]"
                )

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep * realtime_scale)

if __name__ == "__main__":
    main()


    
