from typing import Callable
import numpy as np
import gymnasium as gym

from cmorl.utils.loss_composition import p_mean

RewardFnType = Callable[[np.ndarray, np.ndarray, gym.Env], float]

def default_q_composer(q_values, p_batch=0, p_objectives=-4.0):
    qs_c = p_mean(q_values, p=p_batch, axis=0)
    q_c = p_mean(qs_c, p=p_objectives)
    return qs_c, q_c


class CMORL():
    def __init__(self, dim: int, reward_fn: RewardFnType, q_composer=default_q_composer):
        self.dim = dim
        self.reward_fn = reward_fn
        self.q_composer = q_composer

    def __call__(self, state, action, env: gym.Env):
        return self.reward_fn(state, action, env)
    
    


