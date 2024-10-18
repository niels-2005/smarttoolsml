import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv


def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value

    return func


def example_usage():
    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])

    # with learning rate scheduler
    model = PPO("MlpPolicy", env, learning_rate=linear_schedule(0.001), verbose=1)

    # Training des Modells
    model.learn(total_timesteps=10000)
