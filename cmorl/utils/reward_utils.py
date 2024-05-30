from dataclasses import dataclass
from typing import Callable, NamedTuple, TypeVar
import numpy as np
import gymnasium as gym
from typing import TypeAlias

import tensorflow as tf # type: ignore
from cmorl.utils.loss_composition import p_mean


class Transition(NamedTuple):
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

EnvType = TypeVar('EnvType', bound=gym.Env)
RewardFnType: TypeAlias = Callable[[Transition, EnvType], float|np.ndarray]

def discounted_sum(gamma, minimum=0, maximum=np.inf):
    return (gamma**minimum)*(1-gamma**(maximum-minimum+1))/(1-gamma)

def discounted_window(rewards, gamma, axis=0, normalize=True, done=True, window_size=1) -> np.ndarray:
    """
        discounts the rewards in the axis dimension of the rewards array by a window of size window_size
        example: rewards = [r1, r2, r3], window_size=2 -> [r1 + gamma*r2, r2 + gamma*r3, r3]
    """
    axis_front = np.swapaxes(np.array(rewards), 0, axis)
    axis_size = axis_front.shape[0]
    broadcast_shape = [axis_size] + [1]*(axis_front.ndim-1)
    indices = np.arange(axis_size).reshape(*broadcast_shape)
    normalization_factor = (1.0 - gamma) if normalize else 1.0
    with_discounts = axis_front*discounted_sum(gamma, np.maximum(indices-(window_size-1),0), indices)
    if not done:
        with_discounts[-1] *= discounted_sum(gamma)
    unswapped = np.swapaxes(with_discounts, 0, axis)
    return  unswapped*normalization_factor

def values(rewards, gamma, normalize=True, done=True):
    rev_rewards = rewards[::-1]
    rev_values = np.zeros_like(rewards)
    rev_values[0] = rev_rewards[0] * (1 if done else discounted_sum(gamma))
    for i in range(1, rev_rewards.shape[0]):
        rev_values[i] = rev_rewards[i] + gamma*rev_values[i-1]
    normalization_factor = ((1.0 - gamma) if normalize else 1.0)
    return rev_values[::-1]*normalization_factor

def estimated_value_fn(rewards, gamma, done=True, normalize=True, axis=0):
    return np.mean(discounted_window(rewards, gamma, done=done, normalize=normalize, axis=axis, window_size=np.inf), axis=axis)

@tf.function
def default_q_composer(q_values, p_batch=0, p_objectives=-4.0, scalarize_batch_first=True):
    qs_c = p_mean(q_values,
                  p=p_batch if scalarize_batch_first else p_objectives,
                  axis=0 if scalarize_batch_first else 1)
    q_c = p_mean(qs_c, p=p_objectives if scalarize_batch_first else p_batch)
    return qs_c, q_c

class CMORL:
    def __init__(self, reward_fn: RewardFnType, q_composer: Callable = default_q_composer, shape: int | None = None):
        self.reward_fn = reward_fn
        self.q_composer = q_composer
        self.shape = shape

    def calculate_space(self, env):
        if self.shape is not None:
            return gym.spaces.Box(low=0.0, high=1.0, shape=[self.shape])
        example_rw = self.reward_fn(random_transition(env), env)
        return gym.spaces.Box(low = 0.0, high=1.0, shape=example_rw.shape, dtype=example_rw.dtype) 

    def __call__(self, transition: Transition, env: gym.Env):
        return self.reward_fn(transition, env)

