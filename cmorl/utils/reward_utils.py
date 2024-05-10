from dataclasses import dataclass
from typing import Callable
import numpy as np
import gymnasium as gym

from cmorl.utils.loss_composition import p_mean

@dataclass
class Transition():
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    done: bool
    info: dict

def random_transition(env: gym.Env):
    return Transition(
        state=env.observation_space.sample(),
        action=env.action_space.sample(),
        next_state=env.observation_space.sample(),
        done=False,
        info={}
    )
    

RewardFnType = Callable[[Transition], float|np.ndarray]

def default_q_composer(q_values, p_batch=0, p_objectives=-4.0):
    qs_c = p_mean(q_values, p=p_batch, axis=0)
    q_c = p_mean(qs_c, p=p_objectives)
    return qs_c, q_c

class CMORL():
    def __init__(self, reward_fn: RewardFnType, q_composer=default_q_composer, shape: int=None):
        self.reward_fn = reward_fn
        self.q_composer = q_composer
        self._shape = shape

    def calculate_space(self, env):
        if self._shape is not None:
            return gym.spaces.Box(low=0.0, high=1.0, shape=[self._shape])
        example_rw = self.reward_fn(random_transition(env), env)
        return gym.spaces.Box(low = 0.0, high=1.0, shape=example_rw.shape, dtype=example_rw.dtype) 

    def __call__(self, transition: Transition, env: gym.Env):
        return self.reward_fn(transition, env)

