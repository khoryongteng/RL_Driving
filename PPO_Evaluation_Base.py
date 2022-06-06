from Environments.car_racing import CarRacing    
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import pandas as pd
import os

env = CarRacing()
env = Monitor(env)
for i in range(1, 4):
    results = []
    for step in range(20000, 1000001, 20000):
        ppo_path = os.path.join('Models', 'PPO_Base', "PPO_"+str(i), "PPO_Base_"+str(step)+"_steps")
        model = PPO.load(ppo_path, env=env)
        mean_reward, std = evaluate_policy(model, env, n_eval_episodes=10, render=False)
        results.append((step, mean_reward, std))
    outputdf = pd.DataFrame(results)
    outputdf.rename(columns={0: "Step", 1: "Value", 2:"Std"}, inplace=True)
    os.makedirs("Evaluation/PPO_Base/PPO_"+str(i), exist_ok=True)  
    outputdf.to_csv("Evaluation/PPO_Base/PPO_"+str(i)+"/eval_rew_mean.csv")
    