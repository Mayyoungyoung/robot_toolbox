import mujoco
import mujoco.viewer
import numpy as np
from robot_solver import RobotSolver
import time



if __name__ == "__main__":
    solver = RobotSolver(
        urdf_path="model/UR5/ur5e_robotiq85/ur5e.urdf",
        base_elements=["base_link"],
        verbose=True
    )

    model = mujoco.MjModel.from_xml_path('model/UR5/ur5e_robotiq85/scene.xml')
    data = mujoco.MjData(model)
    data.qpos[:7] = [0.0, -1.57, 1.57, 0.0, 0.0, 0.0, 0.04] 
    mujoco.mj_forward(model, data)  # 更新数据结构以反映新的qpos
    num_active_links = np.sum(np.array(solver.active_links_mask, dtype=bool))
    print("num_active_links =", num_active_links)
    target_pos = [-0.3, -0.3, 0.7]          # x,y,z
    target_euler_deg = [0.0, 90.0, 0.0]            # RX,RY,RZ（度）
    q_initial = data.qpos[:num_active_links]
    reachable, q_sol = solver.is_pose_reachable(
        target_pos = target_pos,          # x,y,z
        target_euler_deg = target_euler_deg  ,
        return_solution=True
    )

    print("是否可达:", reachable)
    print("关节解:", q_sol)
    print("轨迹规划1中...")
    fk = solver.solve_fk(q_sol)
    print("fk:", fk[:3, 3])
    traj = solver.generate_cartesian_linear_trajectory(
        current_q=q_initial,
        target_pose=target_pos,
        target_euler_deg=target_euler_deg,
        num_steps=100
    )
    traj = solver.smooth_trajectory_quintic(traj)
    print("轨迹规划2中...")
    current_q = traj[-1]
    target_pos = [-0.3, -0.3, 0.2]          # x,y,z
    target_euler_deg = [0.0, 90.0, 0.0]           # RX,RY,RZ（度）
    reachable, q_sol = solver.is_pose_reachable(
        target_pos = target_pos,          # x,y,z
        target_euler_deg = target_euler_deg  ,
        return_solution=True
    )

    print("是否可达:", reachable)
    print("关节解:", q_sol)
    fk = solver.solve_fk(q_sol)
    print("fk:", fk[:3, 3])
    traj2 = solver.generate_cartesian_linear_trajectory(
            current_q=current_q,
            target_pose=target_pos,
            target_euler_deg=target_euler_deg,
            num_steps=200
        )
    traj2 = solver.smooth_trajectory_quintic(traj2)
    print("轨迹规划完成")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("开始仿真")
        time.sleep(2)
        solver.play_trajectory_until_close(model, data, viewer, traj, qpos_tol=0.02, max_inner_steps=100)
            
        hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")
        print("hand:", data.xpos[hand_id])
        
        solver.play_trajectory_until_close(model, data, viewer, traj2, qpos_tol=0.02, max_inner_steps=100,)
        hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")
        print("hand:", data.xpos[hand_id])
        
        while viewer.is_running():
            viewer.sync()

    
    # print("IK q:", q)
    # print("site pos:", data.site_xpos[0])

    # fk = solver.solve_fk(q)
    # link0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link0")
    # hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    # site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    # link7_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")

    # print("link0_id =", link0_id)
    # print("hand_id =", hand_id)
    # print("site_id =", site_id)

    # print("link0 world pos =", data.xpos[link0_id])
    # print("hand world pos =", data.xpos[hand_id])
    # print("attachment_site world pos =", data.site_xpos[site_id])
    # print("fk hand pos:", fk[:3, 3])
    # # 打印base_link和末端的朝向
    # print("link0 world ori =", data.xmat[link0_id])
    # print("link7 world ori =", data.xmat[link7_id])
    # print("hand world ori =", data.xmat[hand_id])
    

    