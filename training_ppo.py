import gym

from stable_baselines3 import PPO

env = gym.make("MountainCar-v0")


model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500000)
model.save("ppo_mountaincar")
print("Done")