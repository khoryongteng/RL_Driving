from Environments.car_racing_modification import carRacingGreen  
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os

log_path = os.path.join("Logs", "PPO_Green")

env = carRacingGreen()
env = Monitor(env)

for i in range(3):

    ppo_path = os.path.join("Models", "PPO_Green", "PPO_"+str(i+1))

    checkpoint_callback = CheckpointCallback(save_freq=20000, save_path=ppo_path, name_prefix="PPO_Green")

    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=log_path, learning_rate=0.00003)
    model.learn(total_timesteps = 1000000, callback=checkpoint_callback)