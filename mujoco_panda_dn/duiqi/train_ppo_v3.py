"""
v3版本训练脚本：三阶段策略训练
"""
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import json
from peg_in_hole_env_v3 import PegInHoleEnv


class TrainingCallback(BaseCallback):
    """自定义回调"""
    
    def __init__(self, save_freq=5000, log_dir='logs/peg_in_hole_v3'):
        super().__init__()
        self.save_freq = save_freq
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.stats_log = []
    
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(f'{self.log_dir}/model_step_{self.n_calls}')
            print(f"[Callback] Saved model at step {self.n_calls}")
        
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            mean_length = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
            
            self.stats_log.append({
                'step': self.n_calls,
                'mean_ep_reward': float(mean_reward),
                'mean_ep_length': float(mean_length)
            })
            
            if self.n_calls % self.save_freq == 0:
                print(f"Step {self.n_calls}: Mean Reward = {mean_reward:.2f}, Mean Length = {mean_length:.0f}")
        
        return True
    
    def _on_training_end(self):
        with open(f'{self.log_dir}/training_stats.json', 'w') as f:
            json.dump(self.stats_log, f, indent=2)
        print(f"Training stats saved to {self.log_dir}/training_stats.json")


def train(total_timesteps=300000, learning_rate=3e-4):
    """训练PPO模型"""
    
    print("=" * 60)
    print("v3版本：三阶段策略训练")
    print("=" * 60)
    
    env = PegInHoleEnv(
        xml_path='./franka_emika_panda/scene_hole.xml',
        render=False,
        max_steps=500
    )
    
    print(f"观察空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    print(f"\n策略：")
    print(f"  1. 到达阶段：降低z到hole上方")
    print(f"  2. 对准阶段：调整xy位置对准孔")
    print(f"  3. 插入阶段：垂直下沉插入孔")
    print()
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1
    )
    
    callback = TrainingCallback(save_freq=10000, log_dir='./logs/peg_in_hole_v3')
    
    eval_env = PegInHoleEnv(
        xml_path='./franka_emika_panda/scene_hole.xml',
        render=False,
        max_steps=500
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./logs/peg_in_hole_v3/best_model',
        log_path='./logs/peg_in_hole_v3',
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=False,
        render=False
    )
    
    print("=" * 60)
    print(f"开始训练 (总步数: {total_timesteps})...")
    print("=" * 60)
    print()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[callback, eval_callback],
        progress_bar=True
    )
    
    model.save('./logs/peg_in_hole_v3/final_model')
    print("\n训练完成！最终模型已保存到 ./logs/peg_in_hole_v3/final_model")
    
    env.close()
    eval_env.close()


if __name__ == "__main__":
    train(total_timesteps=300000, learning_rate=3e-4)
