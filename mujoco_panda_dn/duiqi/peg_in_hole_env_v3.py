import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from typing import Tuple

class PegInHoleEnv(gym.Env):
    """
    三阶段强化学习环境：机械臂夹取peg并插入hole
    
    策略：
    1. 到达阶段：降低z，使peg到hole上方
    2. 对准阶段：调整xy位置，使peg对准孔
    3. 插入阶段：垂直向下插入peg进入孔
    """
    
    def __init__(self, xml_path='./franka_emika_panda/scene_hole.xml', 
                 render=False, max_steps=500):
        super(PegInHoleEnv, self).__init__()
        
        # 加载MuJoCo模型
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.xml_path = xml_path
        self.render_mode = render
        self.max_steps = max_steps
        self.step_count = 0
        
        # 初始状态
        self.home_key_id = self.model.key("grasp_home").id
        
        # 身体ID
        self.peg_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "peg")
        self.hole_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hole")
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        
        # ========== 动作空间 ==========
        # 8维：7个关节速度指令 + 1个夹爪控制
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(8,),
            dtype=np.float32
        )
        
        # ========== 观察空间 ==========
        # 关节角(7) + 关节速度(7) + peg位置(3) + hole位置(3) + ee位置(3) = 23维
        obs_dim = 7 + 7 + 3 + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # 速度缩放
        self.joint_velocity_scale = 0.5  # rad/s
        self.gripper_velocity_scale = 0.05  # m/s
        
        # 阶段参数
        self.reach_threshold = 0.15      # 第1阶段：z_diff > 0.15m时为到达阶段
        self.alignment_threshold = 0.05  # 第2阶段：xy_dist < 0.05m时进入对准阶段
        self.insertion_threshold = 0.01  # 第3阶段：z_diff < 0.01m时进入插入阶段
        
        # 成功判定
        self.success_xy_thresh = 0.02
        self.success_z_thresh = 0.035
        
        self.viewer = None
        
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 从keyframe重置
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        mujoco.mj_forward(self.model, self.data)
        
        # 固定hole位置
        self.data.body(self.hole_body_id).xpos[:] = [0.7, 0.0, 0.03]
        self.data.body(self.hole_body_id).cvel[:] = 0
        
        self.step_count = 0
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """获取观察状态"""
        # 关节角（7个）
        q = self.data.qpos[:7].copy()
        
        # 关节速度（7个）
        qd = self.data.qvel[:7].copy()
        
        # peg位置
        peg_pos = self.data.body(self.peg_body_id).xpos.copy()
        
        # hole位置
        hole_pos = self.data.body(self.hole_body_id).xpos.copy()
        
        # 末端执行器位置
        ee_pos = self.data.site(self.ee_site_id).xpos.copy()
        
        # 拼接观察
        obs = np.concatenate([q, qd, peg_pos, hole_pos, ee_pos]).astype(np.float32)
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """执行一步动作"""
        self.step_count += 1
        
        # 解析动作
        joint_vel_cmd = action[:7] * self.joint_velocity_scale  # rad/s
        gripper_cmd = action[7] * self.gripper_velocity_scale
        
        # 速度控制
        dt = self.model.opt.timestep
        target_q = self.data.qpos[:7].copy() + joint_vel_cmd * dt
        
        # 限制关节范围
        for i in range(7):
            target_q[i] = np.clip(target_q[i],
                                  self.model.jnt_range[i, 0],
                                  self.model.jnt_range[i, 1])
        
        # 设置控制
        self.data.ctrl[:7] = target_q
        self.data.ctrl[7] = gripper_cmd
        
        # 执行10步仿真
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        # 获取观察和奖励
        obs = self._get_observation()
        reward, info = self._compute_reward()
        
        terminated = self._check_success()
        truncated = self.step_count >= self.max_steps
        done = terminated or truncated
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self) -> Tuple[float, dict]:
        """三阶段奖励函数"""
        info = {}
        reward = 0.0
        
        peg_pos = self.data.body(self.peg_body_id).xpos
        hole_pos = self.data.body(self.hole_body_id).xpos
        
        # 距离指标
        xy_dist = np.linalg.norm(peg_pos[:2] - hole_pos[:2])
        z_diff = hole_pos[2] - peg_pos[2]  # 负值表示peg在hole下方
        
        info['xy_distance'] = xy_dist
        info['z_diff'] = z_diff
        
        # ===== 判断当前阶段 =====
        if z_diff > self.reach_threshold:
            phase = 'reaching'
        elif z_diff > self.insertion_threshold and xy_dist < self.alignment_threshold:
            phase = 'aligning'
        else:
            phase = 'inserting'
        
        info['phase'] = phase
        
        # ===== 阶段1：到达（z还很高）=====
        if phase == 'reaching':
            # 主要目标：下降，使z_diff减小
            # 次要目标：不要偏离太远
            
            # 下降奖励
            reach_reward = -0.2 * z_diff  # z_diff越大越好（负值越大）
            reward += reach_reward
            
            # xy偏离惩罚（防止乱跑）
            if xy_dist > 0.3:
                xy_penalty = -0.1 * (xy_dist - 0.3)
                reward += xy_penalty
        
        # ===== 阶段2：对准（z接近，xy还需要调整）=====
        elif phase == 'aligning':
            # 主要目标：对准xy位置
            # 次要目标：保持z高度
            
            # xy对准奖励
            alignment_reward = -1.0 * xy_dist  # -1.0 ~ 0
            reward += alignment_reward
            
            # 保持z高度（不要插入得太快）
            if z_diff > self.insertion_threshold:
                reward += 0.1  # 保持当前高度的小额奖励
            else:
                # 已经进入插入阶段
                reward -= 0.1
        
        # ===== 阶段3：插入（xy已对准，z开始下沉）=====
        else:  # phase == 'inserting'
            # 主要目标：垂直下沉，保持xy对准
            
            # 插入深度奖励
            if z_diff > 0:  # 还在hole上方
                insert_reward = -0.1 * z_diff  # 鼓励下沉
            else:  # 已插入
                insert_reward = 0.3 * (-z_diff)  # 插入越深奖励越大
            reward += insert_reward
            
            # xy对准奖励（防止插入时偏移）
            xy_align_reward = -0.5 * xy_dist
            reward += xy_align_reward
        
        # ===== 通用奖励 =====
        
        # 完成奖励
        if self._check_success():
            reward += 10.0
            info['success'] = True
        else:
            info['success'] = False
        
        # 时间惩罚
        reward -= 0.01
        
        info['total_reward'] = reward
        return reward, info
    
    def _check_success(self) -> bool:
        """判断是否成功"""
        peg_pos = self.data.body(self.peg_body_id).xpos
        hole_pos = self.data.body(self.hole_body_id).xpos
        
        xy_dist = np.linalg.norm(peg_pos[:2] - hole_pos[:2])
        z_insertion = hole_pos[2] - peg_pos[2]
        
        if xy_dist < self.success_xy_thresh and z_insertion > self.success_z_thresh:
            return True
        return False
    
    def render(self, mode='human'):
        """渲染环境"""
        if self.viewer is None:
            from mujoco import viewer
            self.viewer = viewer.launch_passive(self.model, self.data)
        
        self.viewer.sync()
        return None
    
    def close(self):
        """关闭环境"""
        if self.viewer is not None:
            self.viewer.close()
    
    def seed(self, seed=None):
        """设置随机种子"""
        np.random.seed(seed)
        return [seed]
