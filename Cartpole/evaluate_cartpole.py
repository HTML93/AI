# evaluate_cartpole.py

import gym
from stable_baselines3 import PPO

# Create environment
env = gym.make('CartPole-v1')

# Load the trained model
model = PPO.load("ppo_cartpole")

# Evaluate the trained agent
episodes = 5
for episode in range(episodes):
    obs = env.reset()
    done = False
    total_rewards = 0
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        total_rewards += rewards
        env.render()
    print(f"Episode: {episode + 1}, Total Reward: {total_rewards}")
env.close()
