# render_cartpole.py

import gym
import pygame
from stable_baselines3 import PPO

# Initialize pygame
pygame.init()

# Set up display
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption('CartPole')

# Load the trained model
model = PPO.load("ppo_cartpole")

# Create environment
env = gym.make('CartPole-v1')

# Run the trained agent
obs = env.reset()
done = False

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    screen.fill((0, 0, 0))

    # Render environment using pygame
    env.render(mode='rgb_array')
    frame = env.render(mode='rgb_array')
    frame = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    screen.blit(frame, (0, 0))

    pygame.display.update()

pygame.quit()
env.close()
