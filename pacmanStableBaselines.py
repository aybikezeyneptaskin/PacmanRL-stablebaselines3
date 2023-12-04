import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from gymnasium.wrappers import HumanRendering
from gymnasium.wrappers import FrameStack
from gymnasium.wrappers import GrayScaleObservation
from gymnasium.wrappers import ResizeObservation
import os

#vec_env = make_vec_env("ALE/Pacman-v5", n_envs=4)
"""
#random actions: 
episodes = 5
for episode in range(1, episodes+1):
    state = env.reset() 
    done = False
    score = 0 
    
    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, _, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()
"""

def show_img(img, hide_colorbar=False):
    if len(img.shape) < 3 or img.shape[2] == 1:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)

def plot_observation(observation):
    frame_count = observation.shape[0]
    _, axes = plt.subplots(1, frame_count, figsize=(frame_count*4, 5))
    for i in range(frame_count):
        axes[i].imshow(observation[i], cmap='gray')
        axes[i].axis('off')
    plt.show()

#env = make_atari_env('ALE/Pacman-v5', n_envs=1, seed=0)
#env = VecFrameStack(env, n_stack=4)

env = gym.make("ALE/Pacman-v5")
env = GrayScaleObservation(env)
env = ResizeObservation(env, 84)
env = FrameStack(env, 4)

#env = gym.make("ALE/Pacman-v5")
#env = FrameStack(env, 4)

log_path = os.path.join('Training', 'Logs')

#model = DQN("CnnPolicy", env, verbose=1, tensorboard_log=log_path, buffer_size=10000)

model = DQN.load("pacmanlearn1M", env=env, tensorboard_log=log_path, verbose=1)


#model.learn(total_timesteps=1000000)
#model.save("pacmanlearn")
env = gym.make("ALE/Pacman-v5", render_mode='human')
env = GrayScaleObservation(env)
env = ResizeObservation(env, 84)
env = FrameStack(env, 4)

episodes = 5
for episode in range(1, episodes+1):
    state = env.reset() 
    done = False
    score = 0 
    
    obs, _ = env.reset()
    obs = np.asarray(obs)
    #plot_observation(obs)

    while not done:
        obs = np.asarray(obs)
        action, _states = model.predict(obs)
        obs, rewards, done, _, info = env.step(action)
        env.render()
        score+=rewards
    print('Episode:{} Score:{}'.format(episode, score))
env.close()

