"""
v3版本评估脚本
"""
import numpy as np
from stable_baselines3 import PPO
from peg_in_hole_env_v3 import PegInHoleEnv
import mujoco.viewer


def demo_trained_policy(model_path, num_episodes=3):
    """
    演示三阶段策略
    """
    
    print("=" * 60)
    print("v3版本：三阶段演示")
    print("=" * 60 + "\n")
    
    env = PegInHoleEnv(
        xml_path='./franka_emika_panda/scene_hole.xml',
        render=False,
        max_steps=2000
    )
    
    model = PPO.load(model_path)
    
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for ep in range(num_episodes):
            if not viewer.is_running():
                print("\n窗口已关闭")
                break
                
            print(f"\n[Episode {ep+1}/{num_episodes}]")
            
            obs, info = env.reset()
            done = False
            step_count = 0
            
            while not done and viewer.is_running():
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                step_count += 1
                
                viewer.sync()
                
                if step_count % 100 == 0:
                    print(f"  Step {step_count}: "
                          f"Phase={info.get('phase', '?')}, "
                          f"XY={info.get('xy_distance', 0):.3f}m, "
                          f"Z={info.get('z_diff', 0):.3f}m, "
                          f"Reward={reward:.3f}")
                
                if terminated:
                    print(f"  ✓ 成功 (步数: {step_count})")
                    for _ in range(200):
                        if not viewer.is_running():
                            break
                        mujoco.mj_step(env.model, env.data)
                        viewer.sync()
                    break
            
            if not terminated:
                print(f"  ✗ 未成功 (步数: {step_count})")
    
    env.close()
    print("\n演示结束！")


def evaluate_model(model_path, num_episodes=5):
    """评估模型"""
    
    print("=" * 60)
    print(f"加载模型: {model_path}")
    print("=" * 60 + "\n")
    
    env = PegInHoleEnv(
        xml_path='./franka_emika_panda/scene_hole.xml',
        render=False,
        max_steps=2000
    )
    
    model = PPO.load(model_path)
    
    success_count = 0
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        step_count = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
        
        if terminated:
            success_count += 1
            print(f"Episode {ep+1}: ✓ 成功 (步数: {step_count})")
        else:
            print(f"Episode {ep+1}: ✗ 失败 (步数: {step_count})")
    
    print()
    print("=" * 60)
    print(f"成功率: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./logs/peg_in_hole_v3/best_model/best_model')
    parser.add_argument('--demo', action='store_true', default=True)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--episodes', type=int, default=3)
    
    args = parser.parse_args()
    
    if args.eval:
        evaluate_model(args.model, num_episodes=args.episodes)
    else:
        demo_trained_policy(args.model, num_episodes=args.episodes)
