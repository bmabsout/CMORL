from typing import Callable
import numpy as np
import gymnasium as gym

RewardFnType = Callable[[np.ndarray, np.ndarray, gym.Env], float]

class CMORL():
    def __init__(self, dim: int, reward_fn: RewardFnType, q_composer):
        self.dim = dim
        self.reward_fn = reward_fn
        self.q_composer = q_composer

    def __call__(self, state, action, env: gym.Env):
        return self.reward_fn(state, action, env)
    
    


