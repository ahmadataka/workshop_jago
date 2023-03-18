import gym

from stable_baselines3 import DQN

env = gym.make("CartPole-v1")


model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=750000)
model.save("dqn_cartpole")
print("Done")