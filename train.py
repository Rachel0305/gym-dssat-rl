import logging
import sys
import os

# -----------------------------
# 1. 强制使用 /tmp/logs 和 /tmp/output
log_dir = "/tmp/logs"
output_dir = "/tmp/output"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

log_file_path = os.path.join(log_dir, "train_fer_log0116.txt")

# 2. 初始化 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.info(f"Logging to: {log_file_path}")
logging.info(f"Output directory: {output_dir}")

# -----------------------------
# 3. 导入其他模块（在 logging 初始化后）
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from sb3_wrapper import GymDssatWrapper
from gym_dssat_pdi.envs.utils import utils as dssat_utils
import gym

# -----------------------------
if __name__ == "__main__":
    try:
        # 创建环境
        env_args = {
            'log_saving_path': os.path.join(log_dir, 'dssat_fer_pdi.log'),
            'mode': 'fertilization',
            'seed': 123,
            'random_weather': True,
        }

        logging.info(f'###########################\n## MODE: {env_args["mode"]} ##\n###########################')

        env = GymDssatWrapper(
            gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
        )

        # PPO agent
        ppo_args = {
            'seed': 123,
            'gamma': 1,
        }
        ppo_agent = PPO('MlpPolicy', env, **ppo_args)

        # 保存模型路径
        path = os.path.join(output_dir, env_args["mode"])
        os.makedirs(path, exist_ok=True)

        # eval callback
        eval_env_args = {**env_args, 'seed': 345}
        eval_env = GymDssatWrapper(
            gym.make('GymDssatPdi-v0', **eval_env_args)
        )

        eval_callback = EvalCallback(
            eval_env,
            eval_freq=1000,
            best_model_save_path=path,
            deterministic=True,
            n_eval_episodes=10
        )

        # Train
        total_timesteps = 5_000
        logging.info("Training PPO agent...")
        ppo_agent.learn(total_timesteps=total_timesteps, callback=eval_callback)

        # 保存最终模型
        ppo_agent.save(os.path.join(path, "final_model"))
        logging.info("Training done")

    finally:
        env.close()
