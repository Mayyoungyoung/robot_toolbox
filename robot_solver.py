import numpy as np
from ikpy.chain import Chain
import transforms3d as tf
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

class RobotSolver:
    def __init__(self, urdf_path, base_elements=None, verbose=True):
        """
        初始化 RobotSolver 并构建 IK 链信息。

        Args:
        - urdf_path (str): URDF 文件路径或能被 `ikpy.Chain.from_urdf_file` 识别的路径。
        - base_elements (list|None): 可选，传递给 `ikpy` 的 base_elements 列表以指定链的基座元素。
        - verbose (bool): 是否打印链信息。

        Returns:
        - None: 构造函数不返回值，但会设置属性 `chain`, `active_links_mask`, `joint_limits` 等。

        作用:
        - 使用 ikpy 从 URDF 构建链，并提取活动关节掩码与关节限位用于后续 IK 调用。
        """
        self.urdf_path = urdf_path
        self.base_elements = base_elements
        self.verbose = verbose

        if base_elements is not None:
            self.chain = Chain.from_urdf_file(
                urdf_path,
                base_elements=base_elements
            )
        else:
            self.chain = Chain.from_urdf_file(urdf_path)

        self.active_links_mask = self._build_active_links_mask()
        self.chain.active_links_mask = self.active_links_mask

        self.joint_limits = self._extract_joint_limits()

        if self.verbose:
            self.print_chain_info()

    def _build_active_links_mask(self):
        """
        构建活动关节掩码。

        Args:
        - None

        Returns:
        - mask (list of bool): 与 `self.chain.links` 对应的布尔列表，True 表示该链接对应的关节可驱动（revolute/prismatic/continuous）。

        作用:
        - 扫描链中每个 link 的 `joint_type` 字段来判断关节是否为可动关节。
        """

        mask = []
        for link in self.chain.links:
            jt = str(getattr(link, "joint_type", "fixed")).lower()
            movable = jt in ("revolute", "prismatic", "continuous")
            mask.append(movable)
        return mask

    def _extract_joint_limits(self):
        """
        提取关节限位信息。

        Args:
        - None

        Returns:
        - limits (list): 与 `self.chain.links` 对应的列表，每个元素为 `(lower, upper)` 或 `None`（如果无限位信息）。

        作用:
        - 从每个 link 的 `bounds` 属性读取上下限并转换为浮点数，供初始姿态生成或约束使用。
        """

        limits = []
        for link in self.chain.links:
            bound = getattr(link, "bounds", None)
            jt = str(getattr(link, "joint_type", "fixed")).lower()

            if jt in ("revolute", "prismatic", "continuous") and bound is not None:
                try:
                    limits.append((float(bound[0]), float(bound[1])))
                except Exception:
                    limits.append(None)
            else:
                limits.append(None)
        return limits

    def print_chain_info(self):
        """
        打印链的基本信息到标准输出。

        Args:
        - None

        Returns:
        - None

        作用:
        - 将每个 link 的索引、名称、关节类型、是否为活动关节以及限位信息打印出来，便于调试。
        """

        print("-" * 70)
        print("IKPy 链结构 (Index | Name | Joint Type | Active | Limits)")
        for i, link in enumerate(self.chain.links):
            print(
                f"{i:>2} | "
                f"{getattr(link, 'name', f'link_{i}'):<25} | "
                f"{str(getattr(link, 'joint_type', 'unknown')):<10} | "
                f"{str(self.active_links_mask[i]):<5} | "
                f"{self.joint_limits[i]}"
            )
        print("-" * 70)
        print("active_links_mask =", self.active_links_mask)
        print("-" * 70)

    def build_initial_position(self, initial_q=None, strategy="zero"):
        """
        构建初始关节位置向量。

        Args:
        - initial_q (np.ndarray, optional): 预设的初始关节位置（可以是 7 维活动关节向量或全长向量）。
        - strategy (str): 初始策略，支持 `"zero"`（接近 0）和 `"mid"`（限位中点）。

        Returns:
        - q0 (np.ndarray): 长度等于 `len(self.chain.links)` 的浮点数组。
        """
        num_links = len(self.chain.links)
        q0 = np.zeros(num_links, dtype=float)
        
        # 1. 如果提供了 initial_q，处理长度匹配并填充
        if initial_q is not None:
            initial_q = np.asarray(initial_q)
            if len(initial_q) == num_links:
                q0 = initial_q.copy()
            else:
                mask_bool = np.array(self.active_links_mask, dtype=bool)
                if len(initial_q) == np.sum(mask_bool):
                    q0[mask_bool] = initial_q
                else:
                    raise ValueError(f"initial_q 长度不匹配：期望 {num_links} 或 {np.sum(mask_bool)}，实际得到 {len(initial_q)}")
        
        # 2. 如果没有提供 initial_q，则走策略生成
        else:
            for i, lim in enumerate(self.joint_limits):
                if not self.active_links_mask[i]:
                    continue
                
                if lim is None:
                    q0[i] = 0.0
                else:
                    low, high = lim
                    if strategy == "mid":
                        q0[i] = 0.5 * (low + high)
                    else:
                        # strategy == 'zero'：将 0 截断到 [low, high]
                        q0[i] = np.clip(0.0, low, high)
        
        # 3. 最终安全性检查：确保所有活动关节都在限位内（防止传入的 initial_q 越界导致 IK 报错）
        for i, lim in enumerate(self.joint_limits):
            if self.active_links_mask[i] and lim is not None:
                q0[i] = np.clip(q0[i], lim[0], lim[1])

        return q0

    @staticmethod
    def euler_to_matrix(euler_xyz, degrees=False):
        """
        将欧拉角 (XYZ 顺序) 转换为 3x3 旋转矩阵。

        Args:
        - euler_xyz (array-like of 3): 欧拉角 [rx, ry, rz]，单位为弧度或度（取决于 `degrees`）。
        - degrees (bool): 如果为 True，则输入以度为单位，会先转换为弧度。

        Returns:
        - R (np.ndarray): 3x3 旋转矩阵。

        作用:
        - 为需要矩阵形式的目标姿态提供转换工具（用于 IK 的目标 orientation）。
        """

        euler_xyz = np.asarray(euler_xyz, dtype=float)
        if degrees:
            euler_xyz = np.deg2rad(euler_xyz)
        return tf.euler.euler2mat(*euler_xyz)

    def solve_ik(
        self,
        target_position,
        target_orientation=None,
        orientation_mode="all",
        initial_position=None,
        degrees=False
    ):
        """
        求解逆运动学得到关节变量。

        Args:
        - target_position (array-like of 3): 目标末端位置 [x, y, z]（米或与模型一致的单位）。
        - target_orientation (array-like or matrix, optional): 如果提供，支持 3 元欧拉角 (XYZ) 或 3x3 旋转矩阵，用于指定目标姿态。
        - orientation_mode (str): 传给 `ikpy` 的 orientation_mode，默认 `"all"`。
        - initial_position (np.ndarray, optional): 初始位置，用于 IK 求解。
        - degrees (bool): 如果 `target_orientation` 以欧拉角给出且为度，则设为 True。

        Returns:
        - q (np.ndarray): IK 求解得到的关节向量（长度等于 `len(self.chain.links)`），包含 base 到末端的所有 link 对应的值（与 ikpy 的约定一致）。

        作用:
        - 使用 `self.chain.inverse_kinematics` 执行 IK 求解，支持可选的姿态约束和初始值。
        """

        target_position = np.asarray(target_position, dtype=float)

        if initial_position is None:
            initial_position = self.build_initial_position(strategy="mid")

        kwargs = {
            "target_position": target_position,
            "initial_position": initial_position
        }

        if target_orientation is not None:
            target_orientation = np.asarray(target_orientation, dtype=float)
            if target_orientation.shape == (3,):
                target_orientation = self.euler_to_matrix(
                    target_orientation,
                    degrees=degrees
                )
            kwargs["target_orientation"] = target_orientation
            kwargs["orientation_mode"] = orientation_mode

        return self.chain.inverse_kinematics(**kwargs)

    def solve_fk(self, q):
        return self.chain.forward_kinematics(q)


    def generate_trajectory(self, start_q, end_q, num_steps=100):
        """
        生成从 start_q 到 end_q 的线性插值轨迹。

        Args:
        - start_q (array-like): 起始关节向量。
        - end_q (array-like): 目标关节向量。
        - num_steps (int): 轨迹中的步骤数。

        Returns:
        - trajectory (np.ndarray): 形状为 (num_steps, len(self.chain.links)) 的数组，包含每个步骤的关节向量。

        作用:
        - 为给定的起始和目标关节配置生成平滑的线性插值轨迹，便于执行或可视化。
        """

        start_q = np.asarray(start_q, dtype=float)
        end_q = np.asarray(end_q, dtype=float)

        trajectory = np.linspace(start_q, end_q, num=num_steps)
        return trajectory

    def generate_pose_trajectory(self, current_q, target_pose, target_euler_deg, num_steps=100):
        """
        生成从当前姿态到目标姿态的轨迹。

        Args:
        - current_q (array-like): 当前关节向量。
        - target_pose (array-like of 3): 目标末端位置 [x, y, z]。
        - target_euler_deg (array-like of 3): 目标末端欧拉角 [rx, ry, rz]，单位为度。
        - num_steps (int): 轨迹中的步骤数。
        - degrees (bool): 如果 True，则 `target_euler_deg` 已经是度数，无需转换。

        Returns:
        - trajectory (np.ndarray): 形状为 (num_steps, len(self.chain.links)) 的数组，包含每个步骤的关节向量。

        作用:
        - 从当前关节配置开始，生成一个平滑的轨迹，使末端逐渐移动到指定的位置和姿态。
        """
        current_q = self.build_initial_position(initial_q=current_q)
        target_q = self.solve_ik(
            target_position=target_pose,
            target_orientation=target_euler_deg,
            orientation_mode="all",
            initial_position=current_q,
            degrees =  True
        )
        
        traj = self.generate_trajectory(current_q, target_q, num_steps=num_steps)
        return traj


    def generate_cartesian_linear_trajectory(
        self,
        current_q,
        target_pose,
        target_euler_deg,
        num_steps=100,
        num_seed_trials=8,
        noise_scale=0.03,
    ):
        """
        笛卡尔直线轨迹规划：
        - 位置 Lerp
        - 姿态 Slerp
        - 每一步多次 IK 尝试，选取与当前姿态最接近的解

        修改点：
        1. 第一次 IK 求解时，优先用 mid 作为参考解，而不是 current_q
        2. 每一步尝试多个 seed（last_q / mid / 扰动版本），
        从成功结果中选取与“当前参考姿态”差值最小的一组解

        Args:
            current_q (array-like): 当前关节向量，可为活动关节向量或 full 向量
            target_pose (array-like): 目标位置 [x, y, z]
            target_euler_deg (array-like): 目标欧拉角 [rx, ry, rz]，单位度
            num_steps (int): 插值步数
            num_seed_trials (int): 每一步总共尝试多少个 seed
            noise_scale (float): 扰动尺度（弧度），建议 0.01 ~ 0.05

        Returns:
            np.ndarray: shape = (num_steps, len(self.chain.links))
        """

        def _clip_to_joint_limits(q_full):
            """将 full 向量裁剪到关节范围内。"""
            q_full = np.array(q_full, dtype=float).copy()
            for j, link in enumerate(self.chain.links):
                if hasattr(link, "bounds") and link.bounds is not None:
                    low, high = link.bounds
                    if low is not None and high is not None:
                        q_full[j] = np.clip(q_full[j], low, high)
            return q_full

        def _make_seed_candidates(step_idx, last_q, curr_q_full, mid_q_full):
            """
            生成本步候选 seed。
            step_idx == 1 时，优先用 mid，而不是 current_q / last_q。
            后续步优先用 last_q。
            """
            seeds = []

            # 第一步：强制优先 mid
            if step_idx == 1:
                seeds.append(mid_q_full.copy())
                seeds.append(curr_q_full.copy())
                seeds.append(last_q.copy())
            else:
                seeds.append(last_q.copy())
                seeds.append(mid_q_full.copy())
                seeds.append(curr_q_full.copy())

            # 在主要 seed 基础上加扰动
            base_for_noise = [seeds[0], seeds[1]]
            per_base = max(1, (num_seed_trials - len(seeds)) // max(1, len(base_for_noise)))

            for base_q in base_for_noise:
                for _ in range(per_base):
                    q_try = np.array(base_q, dtype=float).copy()

                    # 只对 active joints 加扰动
                    if hasattr(self, "active_links_mask"):
                        active_idx = np.where(np.array(self.active_links_mask, dtype=bool))[0]
                    else:
                        active_idx = np.arange(len(q_try))

                    noise = np.random.normal(loc=0.0, scale=noise_scale, size=len(active_idx))
                    q_try[active_idx] += noise
                    q_try = _clip_to_joint_limits(q_try)
                    seeds.append(q_try)

            # 去重，避免完全重复
            uniq = []
            for s in seeds:
                duplicated = False
                for u in uniq:
                    if np.allclose(s, u, atol=1e-8):
                        duplicated = True
                        break
                if not duplicated:
                    uniq.append(s)

            return uniq[:num_seed_trials]

        def _joint_distance(q1, q2):
            """
            关节空间距离。这里只用 L2。
            你如果后面想更稳一点，也可以给某些关节加权。
            """
            q1 = np.asarray(q1, dtype=float)
            q2 = np.asarray(q2, dtype=float)
            return np.linalg.norm(q1 - q2)

        def _solve_ik_multi_seed(
            interp_pos,
            interp_rot_mat,
            step_idx,
            last_q,
            curr_q_full,
            mid_q_full,
        ):
            """
            多 seed 求解 IK，从所有成功解中选一个与参考姿态最接近的。
            第一步参考姿态设为 mid，后续参考姿态设为 last_q。
            """
            seed_candidates = _make_seed_candidates(
                step_idx=step_idx,
                last_q=last_q,
                curr_q_full=curr_q_full,
                mid_q_full=mid_q_full,
            )

            # 第一步优先贴近 mid，后续优先贴近上一帧
            if step_idx == 1:
                q_ref_for_selection = mid_q_full
            else:
                q_ref_for_selection = last_q

            best_q = None
            best_cost = np.inf
            fail_msgs = []

            for seed in seed_candidates:
                try:
                    # 注意：这里假设你的 solve_ik 已经支持 initial_position
                    q_sol = self.solve_ik(
                        target_position=interp_pos,
                        target_orientation=interp_rot_mat,
                        orientation_mode="all",
                        initial_position=seed,
                        degrees=False
                    )

                    q_sol = np.asarray(q_sol, dtype=float)

                    # 代价 1：尽量接近当前参考姿态
                    joint_cost = _joint_distance(q_sol, q_ref_for_selection)

                    # 代价 2：位置误差也要看一下，防止求出“看起来近但末端不准”的解
                    fk = self.solve_fk(q_sol)
                    pos_err = np.linalg.norm(fk[:3, 3] - interp_pos)

                    # 组合代价：位置误差优先，再考虑关节变化平滑性
                    total_cost = 10.0 * pos_err + joint_cost

                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_q = q_sol

                except Exception as e:
                    fail_msgs.append(str(e))

            if best_q is None:
                raise RuntimeError(
                    "多 seed IK 全部失败。最后几个错误："
                    + " | ".join(fail_msgs[-3:])
                )

            return best_q

        # 1. 统一 current_q 长度
        curr_q_full = self.build_initial_position(initial_q=current_q)

        # 2. 准备一个稳定的 mid 作为第一步参考解
        mid_q_full = self.build_initial_position(strategy="mid")

        # 3. 用 current_q 的 FK 作为轨迹起点
        start_matrix = self.solve_fk(curr_q_full)
        start_pos = start_matrix[:3, 3]
        start_rot_mat = start_matrix[:3, :3]

        # 4. 目标位姿
        target_pos = np.asarray(target_pose, dtype=float)
        target_rot_mat = self.euler_to_matrix(target_euler_deg, degrees=True)

        # 5. 姿态插值器
        key_rots = R.from_matrix([start_rot_mat, target_rot_mat])
        key_times = [0.0, 1.0]
        slerp_interpolator = Slerp(key_times, key_rots)

        # 6. 初始化轨迹
        trajectory = [curr_q_full.copy()]
        last_q = curr_q_full.copy()

        # 7. 增量式循环求解
        for i in range(1, num_steps):
            alpha = i / (num_steps - 1)

            # 位置插值
            interp_pos = (1.0 - alpha) * start_pos + alpha * target_pos

            # 姿态插值
            interp_rot = slerp_interpolator([alpha])[0]
            interp_rot_mat = interp_rot.as_matrix()

            try:
                step_q = _solve_ik_multi_seed(
                    interp_pos=interp_pos,
                    interp_rot_mat=interp_rot_mat,
                    step_idx=i,
                    last_q=last_q,
                    curr_q_full=curr_q_full,
                    mid_q_full=mid_q_full,
                )

                trajectory.append(step_q)
                last_q = step_q.copy()

            except Exception as e:
                print(f"警告: 第 {i} 步多 seed IK 求解失败，保持上一帧状态。错误信息: {e}")
                trajectory.append(last_q.copy())

        return np.array(trajectory)

