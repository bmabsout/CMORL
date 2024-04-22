from os import path
from typing import Optional

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

import tensorflow as tf
from cmorl.utils.loss_composition import p_mean
from cmorl.utils.reward_utils import CMORL, RewardFnType

def multi_dim_reward(state, action, env: "BoidsEnv"):

    return rw_vec


def composed_reward_fn(state, action, env: "BoidsEnv"):
    
    return reward


@tf.function
def q_composer(q_values):
    # q1_c = q_values[0]
    # q2_c = q_values[1]
    # q_values = tf.stack([q1_c, q2_c], axis=0)
    qs_c = tf.reduce_mean(q_values, axis=0)
    q_c = p_mean(qs_c, p=-4.0)
    return qs_c, q_c



class BoidsEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        screen=None,
        numBoids: int = 10,
        # reward_fn: RewardFnType = composed_reward_fn,
        reward_fn: RewardFnType = multi_dim_reward,
    ):
        max_speed = 8
        max_acc = 8
        max_angular_speed = 30.0*np.pi/180.0 # rads/s
        dt = 1.0/20.0

        self.render_mode = render_mode

        self.screen_dim = 500
        self.screen = screen
        self.clock = None
        self.isopen = True
        
        max_vel = np.array([max_speed, max_speed])
        max_pos = np.array([1.0, 1.0])
        min_pos = np.array([0.0, 0.0])
        obs_low = np.tile(np.concatenate([min_pos, -max_vel]), self.numBoids)
        obs_high = np.tile(np.concatenate([max_pos, max_vel]), self.numBoids)
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.tile([-max_acc, -max_angular_speed], numBoids),
            high=np.tile([max_acc, max_angular_speed], numBoids),
            dtype=np.float32
        )

        reward_dim = reward_fn(
            self.observation_space.sample(), self.action_space.sample(), self
        ).shape[0]
        self.cmorl = CMORL(reward_dim, reward_fn, q_composer)

        # @tf.function
        def difference_eq(flat_state: tf.Tensor, flat_u: tf.Tensor):
            # state is a tensor of shape (4*numBoids)
            # u is a tensor of shape (2*numBoids)
            state = tf.transpose(tf.reshape(flat_state, (numBoids, 4)))
            # state has shape (4, numBoids)
            pos = state[:2]
            vel = state[2:]
            u = tf.transpose(tf.reshape(flat_u, (numBoids, 2)))
            angle_change = u[1]
            acc = u[0]
            angle_changexy = tf.stack([tf.cos(angle_change), tf.sin(angle_change)], axis=0)
            # using leapfrog integration
            new_vel = vel + (acc * angle_changexy)*dt
            # clamp velocity
            new_vel = tf.clip_by_norm(new_vel, max_speed)
            new_pos = pos + new_vel*dt
            # let the position wrap ala torus
            new_pos = tf.math.floormod(new_pos, 1.0)
            return tf.reshape(tf.transpose(tf.concat([new_pos, new_vel], axis=0)), -1)


        self.difference_eq = difference_eq


    def step(self, u):

      
        reward = self.cmorl(self.state, u, self)
        return self._get_obs(), reward, False, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.state = self.observation_space.sample()
        if self.render_mode == "human":
            self.render()
        return self.state, {}

    def render(self):
        exit(1)

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def normed_angular_distance(a, b):
    diff = (b - a + np.pi) % (2 * np.pi) - np.pi
    return np.abs(diff + 2 * np.pi if diff < -np.pi else diff) / np.pi
