%matplotlib
import gym
import math
from TurtleBotGym import TurtleBot
from stable_baselines3 import DQN
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

# env = TurtleBot(1)
# env = gym.make("CartPole-v1")
env = gym.make("MountainCar-v0")
observation = env.reset()

# model = DQN.load("dqn_cartpole", env=env)
model = PPO.load("ppo_mountaincar", env=env)
# print(observation)
for i in range(0,500):
    # action = env.action_space.sample()
    action, _state = model.predict(observation, deterministic=True)
    observation, reward, terminated, _ = env.step(action)
    env.render()
    if terminated:
        print('done')
        observation = env.reset()
env.close()