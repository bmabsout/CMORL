from dataclasses import dataclass
from typing import Callable
import numpy as np
import gymnasium as gym
from typing import TypeAlias

import tensorflow as tf
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
    

RewardFnType: TypeAlias = Callable[[Transition], float|np.ndarray]

@tf.function
def default_q_composer(q_values, p_batch=0, p_objectives=-4.0, scalarize_batch_first=True):
    qs_c = p_mean(q_values,
                  p=p_batch if scalarize_batch_first else p_objectives,
                  axis=0 if scalarize_batch_first else 1)
    q_c = p_mean(qs_c, p=p_objectives if scalarize_batch_first else p_batch)
    return qs_c, q_c

@dataclass
class CMORL:
    reward_fn: RewardFnType
    q_composer: Callable = default_q_composer
    shape: int = None

    def calculate_space(self, env):
        if self.shape is not None:
            return gym.spaces.Box(low=0.0, high=1.0, shape=[self.shape])
        example_rw = self.reward_fn(random_transition(env), env)
        return gym.spaces.Box(low = 0.0, high=1.0, shape=example_rw.shape, dtype=example_rw.dtype) 

    def __call__(self, transition: Transition, env: gym.Env):
        return self.reward_fn(transition, env)

