import mujoco
import mujoco.viewer
import numpy as np
from robot_solver import RobotSolver



if __name__ == "__main__":
    solver = RobotSolver(
        urdf_path="./franka_emika_panda/panda.urdf",
        base_elements=["panda_link0"],
        verbose=True
    )

    model = mujoco.MjModel.from_xml_path('./franka_emika_panda/scene_wrench_test.xml')
    data = mujoco.MjData(model)
    
    target_pos = [0.5, 0.0, 0.55]          # x,y,z
    target_euler_deg = [180.0, 0.0, -45.0]           # RX,RY,RZ（度）
    q_initial = data.qpos[:7]
   
    traj = solver.generate_cartesian_linear_trajectory( 
        current_q=q_initial,
        target_pose=target_pos,
        target_euler_deg=target_euler_deg,
        num_steps=1000
    )
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for q in traj:
            q_active = q[np.array(solver.active_links_mask, dtype=bool)]
            # data.qpos[:7] = q[1 : 8] 
            data.qpos[:7] = q_active[:7]
            mujoco.mj_step(model, data)
            viewer.sync()
        print("fk:", solver.solve_fk(q)[:3, 3])
        q_initial = data.qpos[:7]
        target_pos = [0.5, 0.0, 0.25]          # x,y,z
        target_euler_deg = [180.0, 0.0, -45.0]           # RX,RY,RZ（度）
        traj2 = solver.generate_cartesian_linear_trajectory( 
        current_q=q_initial,
        target_pose=target_pos,
        target_euler_deg=target_euler_deg,
        num_steps=1000
    )
        for q in traj2:
            q_active = q[np.array(solver.active_links_mask, dtype=bool)]
            # data.qpos[:7] = q[1 : 8] 
            data.qpos[:7] = q_active[:7]
            mujoco.mj_step(model, data)
            viewer.sync()
        print("fk:", solver.solve_fk(q)[:3, 3])
        
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
    

    