import numpy as np
from ikpy.chain import Chain
import transforms3d as tf
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import mujoco

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
        num_seed_trials=4,
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
            np.ndarray: shape = (num_steps, len(self.chain.links)) 全关节向量包含False部分
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

    def generate_min_joint_change_trajectory(
        self,
        current_q,
        target_pose,
        target_euler_deg,
        num_steps=30,
        num_seed_trials=6,
        noise_scale=0.03,
        retry_noise_scale=0.08,
        pos_weight=10.0,
        ori_weight=2.0,
    ):
        """
        最小关节变化优先的轨迹规划：
        - 不要求笛卡尔直线
        - 通过多 seed IK 搜索多个可行解
        - 从成功结果中选取与“当前参考姿态”差值最小的一组
        - 第一次优先用 mid 作为参考，而不是 current_q

        Args:
            current_q (array-like):
                当前关节向量，可为活动关节长度或 full 长度
            target_pose (array-like):
                目标末端位置 [x, y, z]
            target_euler_deg (array-like):
                目标末端欧拉角 [rx, ry, rz]，单位：度
            num_steps (int):
                轨迹步数。建议 20~50
            num_seed_trials (int):
                每一步最多尝试多少个 seed
            noise_scale (float):
                正常扰动尺度
            retry_noise_scale (float):
                本轮都失败时，用更大扰动重试
            pos_weight (float):
                末端位置误差权重
            ori_weight (float):
                姿态误差权重

        Returns:
            np.ndarray:
                shape = (num_steps, len(self.chain.links))
        """

        def _clip_to_joint_limits(q_full):
            """裁剪到关节范围内。"""
            q_full = np.asarray(q_full, dtype=float).copy()
            for j, lim in enumerate(self.joint_limits):
                if lim is not None:
                    low, high = lim
                    q_full[j] = np.clip(q_full[j], low, high)
            return q_full

        def _get_active_indices():
            if hasattr(self, "active_links_mask"):
                return np.where(np.array(self.active_links_mask, dtype=bool))[0]
            return np.arange(len(self.chain.links))

        def _angle_wrap(a):
            return (a + np.pi) % (2 * np.pi) - np.pi

        def _joint_distance(q1, q2):
            """
            关节空间距离。
            对 revolute / continuous 来说，用 wrap 后再比较更合理。
            prismatic 关节直接线性差值。
            """
            q1 = np.asarray(q1, dtype=float)
            q2 = np.asarray(q2, dtype=float)
            dq = q1 - q2

            for i, link in enumerate(self.chain.links):
                jt = str(getattr(link, "joint_type", "fixed")).lower()
                if jt in ("revolute", "continuous"):
                    dq[i] = _angle_wrap(dq[i])

            return np.linalg.norm(dq)

        def _rotation_distance(R1, R2):
            """
            姿态误差：旋转角误差（弧度）
            """
            R_rel = R1.T @ R2
            trace_val = np.trace(R_rel)
            cos_theta = (trace_val - 1.0) / 2.0
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            return np.arccos(cos_theta)

        def _make_seed_candidates(step_idx, last_q, curr_q_full, mid_q_full, local_noise_scale):
            """
            生成多 seed：
            - 第一步：mid 优先
            - 后续：last_q 优先
            - 再加若干扰动版本
            """
            seeds = []

            if step_idx == 1:
                primary = [mid_q_full.copy(), curr_q_full.copy(), last_q.copy()]
            else:
                primary = [last_q.copy(), mid_q_full.copy(), curr_q_full.copy()]

            seeds.extend(primary)

            active_idx = _get_active_indices()

            # 对主要 seed 加扰动
            base_for_noise = primary[:2]
            per_base = max(2, (num_seed_trials - len(primary)) // max(1, len(base_for_noise)))

            for base_q in base_for_noise:
                for _ in range(per_base):
                    q_try = np.asarray(base_q, dtype=float).copy()
                    noise = np.random.normal(0.0, local_noise_scale, size=len(active_idx))
                    q_try[active_idx] += noise
                    q_try = _clip_to_joint_limits(q_try)
                    seeds.append(q_try)

            # 再补几个纯随机偏向 mid / last 的点
            while len(seeds) < num_seed_trials:
                if step_idx == 1:
                    base_q = mid_q_full.copy()
                else:
                    base_q = last_q.copy()

                q_try = np.asarray(base_q, dtype=float).copy()
                noise = np.random.normal(0.0, local_noise_scale, size=len(active_idx))
                q_try[active_idx] += noise
                q_try = _clip_to_joint_limits(q_try)
                seeds.append(q_try)

            # 去重
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

        def _solve_ik_multi_seed(target_pos, target_rot_mat, step_idx, last_q, curr_q_full, mid_q_full):
            """
            多 seed 求解 IK，并从成功结果中选一个最优解：
            - 第一步：更偏向 mid
            - 后续：更偏向 last_q
            """
            fail_msgs = []

            for local_noise_scale in [noise_scale, retry_noise_scale]:
                seed_candidates = _make_seed_candidates(
                    step_idx=step_idx,
                    last_q=last_q,
                    curr_q_full=curr_q_full,
                    mid_q_full=mid_q_full,
                    local_noise_scale=local_noise_scale,
                )

                if step_idx == 1:
                    q_ref = mid_q_full
                else:
                    q_ref = last_q

                best_q = None
                best_cost = np.inf

                for seed in seed_candidates:
                    try:
                        q_sol = self.solve_ik(
                            target_position=target_pos,
                            target_orientation=target_rot_mat,
                            orientation_mode="all",
                            initial_position=seed,
                            degrees=False,
                        )

                        q_sol = np.asarray(q_sol, dtype=float)
                        q_sol = _clip_to_joint_limits(q_sol)

                        fk = self.solve_fk(q_sol)
                        fk_pos = fk[:3, 3]
                        fk_rot = fk[:3, :3]

                        pos_err = np.linalg.norm(fk_pos - target_pos)
                        ori_err = _rotation_distance(fk_rot, target_rot_mat)
                        joint_cost = _joint_distance(q_sol, q_ref)

                        # 总代价：末端误差 + 平滑性
                        total_cost = pos_weight * pos_err + ori_weight * ori_err + joint_cost

                        if total_cost < best_cost:
                            best_cost = total_cost
                            best_q = q_sol

                    except Exception as e:
                        fail_msgs.append(str(e))

                if best_q is not None:
                    return best_q

            raise RuntimeError(
                "多 seed IK 全部失败。最后几个错误：" + " | ".join(fail_msgs[-3:])
            )

        # =========================================================
        # 主流程
        # =========================================================
        curr_q_full = self.build_initial_position(initial_q=current_q)
        mid_q_full = self.build_initial_position(strategy="mid")

        target_pos = np.asarray(target_pose, dtype=float)
        target_rot_mat = self.euler_to_matrix(target_euler_deg, degrees=True)

        trajectory = [curr_q_full.copy()]
        last_q = curr_q_full.copy()

        for i in range(1, num_steps):
            try:
                step_q = _solve_ik_multi_seed(
                    target_pos=target_pos,
                    target_rot_mat=target_rot_mat,
                    step_idx=i,
                    last_q=last_q,
                    curr_q_full=curr_q_full,
                    mid_q_full=mid_q_full,
                )
                trajectory.append(step_q.copy())
                last_q = step_q.copy()

            except Exception as e:
                print(f"警告: 第 {i} 步 IK 失败，保持上一帧。错误信息: {e}")
                trajectory.append(last_q.copy())
        return np.array(trajectory)
    
    def smooth_trajectory_quintic(
        self,
        trajectory,
        output_steps=None,
        keep_endpoints=True,
        return_derivatives=False,
    ):
        """
        对已有离散关节轨迹做 5 次时间缩放平滑，使得：
        - 起点速度 = 0
        - 终点速度 = 0
        - 起点加速度 = 0
        - 终点加速度 = 0

        注意：
        1. 该函数不改变轨迹的几何路径，只改变沿路径的时间参数化
        2. 输入轨迹应为 full q，shape = (N, len(self.chain.links))
        3. 输出轨迹仍为 full q

        Args:
            trajectory (np.ndarray):
                shape = (N, len(self.chain.links))
                每一行是一个关节向量（full q）
            output_steps (int | None):
                输出轨迹点数。
                - None: 默认与输入轨迹点数相同
                - int: 按指定点数重采样
            keep_endpoints (bool):
                是否强制让输出首尾点精确等于输入首尾点
            return_derivatives (bool):
                是否同时返回速度和加速度（数值差分近似）

        Returns:
            np.ndarray 或 tuple:
                若 return_derivatives=False:
                    smooth_traj, shape = (M, D)
                若 return_derivatives=True:
                    smooth_traj, vel_traj, acc_traj
        """
        import numpy as np

        traj = np.asarray(trajectory, dtype=float)
        if traj.ndim != 2:
            raise ValueError(f"trajectory 必须是二维数组，当前 shape={traj.shape}")

        n_in, dim = traj.shape
        expected_dim = len(self.chain.links)

        if dim != expected_dim:
            raise ValueError(
                f"trajectory 每步维度应为 len(self.chain.links)={expected_dim}，"
                f"当前得到 dim={dim}"
            )

        if n_in < 2:
            raise ValueError("trajectory 至少需要 2 个点")

        if output_steps is None:
            output_steps = n_in
        if output_steps < 2:
            raise ValueError("output_steps 至少为 2")

        # --------------------------------------------------
        # 1) 用累计弦长参数化原始路径，避免原轨迹局部点距不均导致采样畸变
        # --------------------------------------------------
        seg_len = np.linalg.norm(np.diff(traj, axis=0), axis=1)   # shape=(n_in-1,)
        total_len = np.sum(seg_len)

        if total_len < 1e-12:
            # 整条轨迹几乎不动，直接复制
            smooth_traj = np.repeat(traj[:1], output_steps, axis=0)
            if keep_endpoints:
                smooth_traj[0] = traj[0]
                smooth_traj[-1] = traj[-1]

            if return_derivatives:
                vel = np.zeros_like(smooth_traj)
                acc = np.zeros_like(smooth_traj)
                return smooth_traj, vel, acc
            return smooth_traj

        s_in = np.zeros(n_in)
        s_in[1:] = np.cumsum(seg_len)
        s_in = s_in / s_in[-1]   # 归一化到 [0,1]

        # --------------------------------------------------
        # 2) 生成输出时间 τ，并通过 quintic time scaling 得到 s_out
        #    s(τ) = 10τ^3 - 15τ^4 + 6τ^5
        # --------------------------------------------------
        tau = np.linspace(0.0, 1.0, output_steps)
        s_out = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5

        # --------------------------------------------------
        # 3) 对每个关节沿路径参数 s 做插值
        #    这里用 np.interp，稳定、简单、依赖少
        # --------------------------------------------------
        smooth_traj = np.zeros((output_steps, dim), dtype=float)
        for j in range(dim):
            smooth_traj[:, j] = np.interp(s_out, s_in, traj[:, j])

        # --------------------------------------------------
        # 4) 强制首尾点完全一致
        # --------------------------------------------------
        if keep_endpoints:
            smooth_traj[0] = traj[0]
            smooth_traj[-1] = traj[-1]

        # --------------------------------------------------
        # 5) 可选：返回速度、加速度（数值差分）
        #    这里默认输出是“单位总时长 T=1”下的导数
        #    如果你以后有真实总时长 T，可再做缩放：
        #      qdot_real  = qdot / T
        #      qddot_real = qddot / T^2
        # --------------------------------------------------
        if return_derivatives:
            dt = 1.0 / (output_steps - 1)
            vel = np.gradient(smooth_traj, dt, axis=0, edge_order=2)
            acc = np.gradient(vel, dt, axis=0, edge_order=2)

            if keep_endpoints:
                vel[0] = 0.0
                vel[-1] = 0.0
                acc[0] = 0.0
                acc[-1] = 0.0

            return smooth_traj, vel, acc

        return smooth_traj
    
    def reachable_space(self):
        """
        计算机器人可达空间
        """

    def reachable_space(
    self,
    num_samples=20000,
    end_effector_only=True,
    return_stats=True,
    seed=42,
    active_joint_only=True,
):
        """
        计算机器人可达空间（采样法）

        方法：
        - 在关节范围内随机采样
        - 对每组关节做 FK
        - 收集末端位置点云，近似表示机器人可达空间

        Args:
            num_samples (int):
                随机采样数量，越大越密。常用 5000~50000
            end_effector_only (bool):
                True: 只返回末端位置点云
                False: 预留参数，目前仍返回末端位置点云
            return_stats (bool):
                是否额外返回统计信息（xyz范围、半径范围等）
            seed (int | None):
                随机种子
            active_joint_only (bool):
                True: 只对 active joints 随机采样，其余固定为 0
                False: 对所有非 fixed 且有范围的关节采样

        Returns:
            points (np.ndarray):
                shape = (N, 3)，每一行是一个末端可达点 [x, y, z]

            stats (dict, optional):
                若 return_stats=True，则额外返回：
                {
                    "num_points": int,
                    "x_range": [xmin, xmax],
                    "y_range": [ymin, ymax],
                    "z_range": [zmin, zmax],
                    "radius_range": [rmin, rmax],
                    "center": [cx, cy, cz]
                }
        """
        import numpy as np

        rng = np.random.default_rng(seed)
        n_links = len(self.chain.links)

        # full q 初始化
        q_template = np.zeros(n_links, dtype=float)

        # active mask
        if hasattr(self, "active_links_mask"):
            active_mask = np.array(self.active_links_mask, dtype=bool)
        else:
            active_mask = np.ones(n_links, dtype=bool)

        points = []

        for _ in range(num_samples):
            q = q_template.copy()

            for i, link in enumerate(self.chain.links):
                joint_type = str(getattr(link, "joint_type", "fixed")).lower()
                bounds = getattr(link, "bounds", None)

                # fixed joint 直接跳过
                if joint_type == "fixed":
                    continue

                # 如果只采 active joints，则 inactive 不采样
                if active_joint_only and (not active_mask[i]):
                    continue

                # 优先用 link.bounds
                if bounds is not None and len(bounds) == 2:
                    low, high = bounds
                    if low is not None and high is not None:
                        # 处理无限范围或异常范围
                        if np.isfinite(low) and np.isfinite(high) and high > low:
                            q[i] = rng.uniform(low, high)
                            continue

                # 如果没有有效 bounds，则给一个保守默认范围
                if joint_type in ("revolute", "continuous"):
                    q[i] = rng.uniform(-np.pi, np.pi)
                elif joint_type == "prismatic":
                    q[i] = 0.0
                else:
                    q[i] = 0.0

            try:
                fk = self.solve_fk(q)
                pos = fk[:3, 3]
                if np.all(np.isfinite(pos)):
                    points.append(pos.copy())
            except Exception:
                continue

        if len(points) == 0:
            raise RuntimeError("未采样到有效可达点，请检查 FK 或关节范围设置。")

        points = np.asarray(points, dtype=float)

        if not return_stats:
            return points

        xmin, ymin, zmin = np.min(points, axis=0)
        xmax, ymax, zmax = np.max(points, axis=0)

        radii = np.linalg.norm(points, axis=1)

        stats = {
            "num_points": int(len(points)),
            "x_range": [float(xmin), float(xmax)],
            "y_range": [float(ymin), float(ymax)],
            "z_range": [float(zmin), float(zmax)],
            "radius_range": [float(np.min(radii)), float(np.max(radii))],
            "center": np.mean(points, axis=0).tolist(),
        }

        return points, stats


    def is_pose_reachable(
        self,
        target_pos,
        target_euler_deg,
        pos_tolerance=1e-3,
        ori_tolerance_deg=2.0,
        num_seed_trials=20,
        seed=42,
        initial_position=None,
        return_solution=False,
    ):
        """
        检测给定目标位姿（位置+姿态）是否可达

        方法：
        - 对目标位姿进行多 seed IK 尝试
        - 若某次 IK 成功，且 FK 验证的位置误差和姿态误差都在阈值内，则认为可达

        Args:
            target_pos (array-like):
                目标位置 [x, y, z]
            target_euler_deg (array-like):
                目标欧拉角 [rx, ry, rz]，单位：度
            pos_tolerance (float):
                位置误差阈值（米）
            ori_tolerance_deg (float):
                姿态误差阈值（度）
            num_seed_trials (int):
                随机 seed 尝试次数
            seed (int | None):
                随机种子
            initial_position (array-like | None):
                可选初始关节。若提供，会优先尝试
            return_solution (bool):
                是否返回可达时对应的关节解

        Returns:
            reachable (bool)
            q_sol (np.ndarray | None, optional)
        """
        import numpy as np

        rng = np.random.default_rng(seed)
        target_pos = np.asarray(target_pos, dtype=float)
        target_rot = self.euler_to_matrix(target_euler_deg, degrees=True)
        ori_tolerance_rad = np.deg2rad(ori_tolerance_deg)

        n_links = len(self.chain.links)
        q_zero = np.zeros(n_links, dtype=float)

        if hasattr(self, "active_links_mask"):
            active_mask = np.array(self.active_links_mask, dtype=bool)
        else:
            active_mask = np.ones(n_links, dtype=bool)

        def _rotation_distance(R1, R2):
            """
            计算两个旋转矩阵之间的最小旋转角误差（弧度）
            """
            R_rel = R1.T @ R2
            trace_val = np.trace(R_rel)
            cos_theta = (trace_val - 1.0) / 2.0
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            return np.arccos(cos_theta)

        def _sample_random_q():
            q = q_zero.copy()
            for i, link in enumerate(self.chain.links):
                joint_type = str(getattr(link, "joint_type", "fixed")).lower()
                bounds = getattr(link, "bounds", None)

                if joint_type == "fixed":
                    continue
                if not active_mask[i]:
                    continue

                if bounds is not None and len(bounds) == 2:
                    low, high = bounds
                    if (
                        low is not None and high is not None
                        and np.isfinite(low) and np.isfinite(high)
                        and high > low
                    ):
                        q[i] = rng.uniform(low, high)
                        continue

                if joint_type in ("revolute", "continuous"):
                    q[i] = rng.uniform(-np.pi, np.pi)
                elif joint_type == "prismatic":
                    q[i] = 0.0
                else:
                    q[i] = 0.0
            return q

        seed_list = []

        # 1. 优先尝试用户给的初始关节
        if initial_position is not None:
            try:
                q_init = self.build_initial_position(initial_q=initial_position)
            except Exception:
                q_init = np.asarray(initial_position, dtype=float)
            seed_list.append(q_init)

        # 2. 尝试 mid
        try:
            q_mid = self.build_initial_position(strategy="mid")
            seed_list.append(q_mid)
        except Exception:
            pass

        # 3. 补充随机 seed
        while len(seed_list) < num_seed_trials:
            seed_list.append(_sample_random_q())

        best_q = None
        best_cost = np.inf

        for q_seed in seed_list:
            try:
                q_sol = self.solve_ik(
                    target_position=target_pos,
                    target_orientation=target_rot,
                    orientation_mode="all",
                    initial_position=q_seed,
                    degrees=False,
                )

                q_sol = np.asarray(q_sol, dtype=float)

                fk = self.solve_fk(q_sol)
                fk_pos = fk[:3, 3]
                fk_rot = fk[:3, :3]

                pos_err = np.linalg.norm(fk_pos - target_pos)
                ori_err = _rotation_distance(fk_rot, target_rot)

                if pos_err < pos_tolerance and ori_err < ori_tolerance_rad:
                    if return_solution:
                        return True, q_sol
                    return True

                # 即使没达阈值，也保存一个“最接近”的解
                total_cost = pos_err + ori_err
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_q = q_sol

            except Exception:
                continue

        if return_solution:
            return False, best_q
        return False


    def play_trajectory(
        self,
        model,
        data,
        viewer,
        traj,
        num_active_links=None,
        steps_per_waypoint=20
    ):
        active_mask = np.array(self.active_links_mask, dtype=bool)

        if num_active_links is None:
            num_active_links = int(np.sum(active_mask))

        for q in traj:
            q = np.asarray(q, dtype=float)
            q_active = q[active_mask]

            data.ctrl[:num_active_links] = q_active[:num_active_links]

            for _ in range(steps_per_waypoint):
                mujoco.mj_step(model, data)
                viewer.sync()


    def play_trajectory_until_close(
        self,
        model,
        data,
        viewer,
        traj,
        num_active_links=None,
        qpos_tol=0.02,
        max_inner_steps=100,
    ):
        """
        播放轨迹，并在每个 waypoint 处等待机器人尽量跟踪到位。

        Args:
            model: mujoco.MjModel
            data: mujoco.MjData
            viewer: mujoco viewer
            traj (np.ndarray or list):
                轨迹，可为：
                - shape = (N, len(self.chain.links)) 的 full q
                - shape = (N, num_active_links) 的 active joint q
            num_active_links (int | None):
                控制的活动关节数量。None 时自动取 active_links_mask 中 True 的个数
            qpos_tol (float):
                当前 waypoint 的关节误差阈值
            max_inner_steps (int):
                每个 waypoint 最多执行多少个 mj_step
        """
        active_mask = np.array(self.active_links_mask, dtype=bool)

        if num_active_links is None:
            num_active_links = int(np.sum(active_mask))

        for q in traj:
            q = np.asarray(q, dtype=float)

            # 兼容 full q 和 active q 两种输入
            if len(q) == len(active_mask):
                q_active = q[active_mask]
            elif len(q) == num_active_links:
                q_active = q
            else:
                raise ValueError(
                    f"轨迹单步维度不匹配: len(q)={len(q)}, "
                    f"len(active_mask)={len(active_mask)}, "
                    f"num_active_links={num_active_links}"
                )

            q_cmd = q_active[:num_active_links]
            data.ctrl[:num_active_links] = q_cmd

            for _ in range(max_inner_steps):
                mujoco.mj_step(model, data)
                viewer.sync()

                err = np.linalg.norm(data.qpos[:num_active_links] - q_cmd)
                if err < qpos_tol:
                    break