from Environments.car_racing_modification import carRacingResized
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import pandas as pd
import os

env = carRacingResized(72)
env = Monitor(env)
for i in range(1, 4):
    results = []
    for step in range(20000, 1000001, 20000):
        ppo_path = os.path.join('Models', 'PPO_Resized_72', "PPO_"+str(i), "PPO_Resized_72_"+str(step)+"_steps")
        model = PPO.load(ppo_path, env=env)
        mean_reward, std = evaluate_policy(model, env, n_eval_episodes=10, render=False)
        results.append((step, mean_reward, std))
    outputdf = pd.DataFrame(results)
    outputdf.rename(columns={0: "Step", 1: "Value", 2:"Std"}, inplace=True)
    os.makedirs("Evaluation/PPO_Resized_72/PPO_"+str(i), exist_ok=True)  
    outputdf.to_csv("Evaluation/PPO_Resized_72/PPO_"+str(i)+"/eval_rew_mean.csv")
    