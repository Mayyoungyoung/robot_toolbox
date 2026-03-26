**RobotSolver 使用说明**

本文档说明如何在项目中使用 `RobotSolver`（见 [robot_solver.py](robot_solver.py)）。包含类的构造、常用方法以及每个函数的示例调用。

**快速开始**
- **环境**: Python, ikpy, mujoco, numpy, scipy（与仓库 demo 相同环境）。
- **示例导入**:

```python
from robot_solver import RobotSolver
import numpy as np
```

**创建实例**
- **构造**: `RobotSolver(urdf_path, base_elements=None, verbose=True)`

示例:

```python
solver = RobotSolver(
    urdf_path="./model/franka_emika_panda/panda.urdf",
    base_elements=["panda_link0"],
    verbose=True,
)
```

**方法与示例**

- **print_chain_info()**: 打印链信息（索引、名称、关节类型、是否活动与限位）。

```python
solver.print_chain_info()
```

- **build_initial_position(initial_q=None, strategy="zero") -> np.ndarray**
  - 生成 full-length 初始关节向量（长度等于 `len(self.chain.links)`）。
  - `initial_q` 可传入 active-length 或 full-length 向量。
  - `strategy`: "zero"（默认）或 "mid"（使用关节限位中点）。

示例:

```python
# 使用 strategy="mid" 生成初始姿态
q0 = solver.build_initial_position(strategy="mid")

# 从 active-length 向量扩展到 full-length
active_q = np.zeros(int(np.sum(solver.active_links_mask)))
q_full = solver.build_initial_position(initial_q=active_q)
```

- **euler_to_matrix(euler_xyz, degrees=False) -> np.ndarray(3x3)**
  - 将 XYZ 顺序的欧拉角转换为 3x3 旋转矩阵。

示例:

```python
R = solver.euler_to_matrix([180.0, 0.0, -45.0], degrees=True)
```

- **solve_ik(target_position, target_orientation=None, orientation_mode="all", initial_position=None, degrees=False) -> np.ndarray**
  - 求解 IK，返回 full-length 关节向量（与 ikpy 一致）。
  - `target_orientation` 可传 3 元欧拉角或 3x3 矩阵。

示例:

```python
target_pos = [0.5, 0.3, 0.65]
target_euler_deg = [180.0, 0.0, -45.0]

q_sol = solver.solve_ik(
    target_position=target_pos,
    target_orientation=target_euler_deg,
    orientation_mode="all",
    degrees=True,
)
print("IK 解（full q）：", q_sol)
```

- **solve_fk(q) -> np.ndarray(4x4)**
  - 计算正运动学，返回同 ikpy 约定的 4x4 变换矩阵。

示例:

```python
T = solver.solve_fk(q_sol)
pos = T[:3, 3]
print("FK 末端位置：", pos)
```

- **generate_trajectory(start_q, end_q, num_steps=100) -> np.ndarray**
  - joint-space 线性插值，输出 shape=(num_steps, D)。

示例:

```python
traj = solver.generate_trajectory(q0, q_sol, num_steps=50)
```

- **generate_pose_trajectory(current_q, target_pose, target_euler_deg, num_steps=100)**
  - 从 `current_q` 到目标位姿做 IK 并返回 joint-space 轨迹。

示例:

```python
traj = solver.generate_pose_trajectory(
    current_q=q0,
    target_pose=[0.5,0.3,0.65],
    target_euler_deg=[180.0,0.0,-45.0],
    num_steps=100,
)
```

- **generate_cartesian_linear_trajectory(current_q, target_pose, target_euler_deg, num_steps=100, num_seed_trials=4, noise_scale=0.03)**
  - 笛卡尔直线（位置线性、姿态 Slerp）逐步做 IK；每步尝试多种 seed，选择最接近参考的解。适用于需要末端 Cartesian 直线轨迹的场景。

示例（与 mujoco 配合播放）：

```python
traj = solver.generate_cartesian_linear_trajectory(
    current_q=q0,
    target_pose=[0.5,0.3,0.65],
    target_euler_deg=[180.0,0.0,-45.0],
    num_steps=200,
)

# 将 active joints 发送到 mujoco 控制
import mujoco
model = mujoco.MjModel.from_xml_path('model/franka_emika_panda/scene_wrench_test.xml')
data = mujoco.MjData(model)
for q in traj:
    q_active = q[np.array(solver.active_links_mask, dtype=bool)]
    data.ctrl[:len(q_active)] = q_active
    mujoco.mj_step(model, data)
```

- **generate_min_joint_change_trajectory(current_q, target_pose, target_euler_deg, num_steps=30, num_seed_trials=6, noise_scale=0.03, retry_noise_scale=0.08, pos_weight=10.0, ori_weight=2.0)**
  - 优先最小化关节变化的轨迹生成（不强制笛卡尔直线）。适合优先平滑关节运动的场景。

示例:

```python
traj = solver.generate_min_joint_change_trajectory(
    current_q=q0,
    target_pose=[0.4,0.2,0.5],
    target_euler_deg=[180.0,0.0,-45.0],
    num_steps=40,
)
```

- **smooth_trajectory_quintic(trajectory, output_steps=None, keep_endpoints=True, return_derivatives=False)**
  - 对已有轨迹做 quintic 时间缩放平滑（首末速度与加速度为 0）。

示例:

```python
smooth = solver.smooth_trajectory_quintic(traj, output_steps=200)
```

- **reachable_space(num_samples=20000, end_effector_only=True, return_stats=True, seed=42, active_joint_only=True)**
  - 采样法估算可达位置点云，返回 `(points, stats)`。

示例:

```python
points, stats = solver.reachable_space(num_samples=5000, seed=0)
print("采样点数：", stats['num_points'])
```

- **is_pose_reachable(target_pos, target_euler_deg, pos_tolerance=1e-3, ori_tolerance_deg=2.0, num_seed_trials=20, seed=42, initial_position=None, return_solution=False)**
  - 多 seed IK 检验位姿可达性，返回布尔（与可选解）。

示例:

```python
ok, q_sol = solver.is_pose_reachable(
    target_pos=[0.5,0.3,0.65],
    target_euler_deg=[180.0,0.0,-45.0],
    num_seed_trials=10,
    seed=1,
    return_solution=True,
)
print("是否可达：", ok)
```

- **play_trajectory(model, data, viewer, traj, num_active_links=None, steps_per_waypoint=20)**
  - 在 mujoco 环境中播放轨迹（逐 waypoint 多步 mj_step）。

示例: 参考 demo_1.py 中的播放方式（仓库内示例）。

- **play_trajectory_until_close(model, data, viewer, traj, num_active_links=None, qpos_tol=0.02, max_inner_steps=100)**
  - 播放轨迹并在每个 waypoint 等待机器人跟踪到位（按阈值）。

示例: 同上，结合 `data.qpos` 或 `data.ctrl` 使用。

**调试建议**
- 如果 IK 多次失败：
  - 尝试不同的 `initial_position` 或使用 `strategy=\"mid\"` 作为 seed；
  - 增大 `num_seed_trials` 或 `noise_scale`；
  - 确认 URDF 中关节限位是否正确导入。

**参考**
- 源码位置: [robot_solver.py](robot_solver.py)
- 示例运行脚本: demo_1.py、demo_2.py、demo_3.py

如需我把这个文档提交为 README.md（替换仓库原有 README），或再把每个示例写成可执行的小脚本，请告诉我。
