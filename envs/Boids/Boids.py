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

    return np.array([0.0])


def composed_reward_fn(state, action, env: "BoidsEnv"):
    
    return 0.0


@tf.function
def q_composer(q_values):
    # q1_c = q_values[0]
    # q2_c = q_values[1]
    # q_values = tf.stack([q1_c, q2_c], axis=0)
    qs_c = tf.reduce_mean(q_values, axis=0)
    q_c = p_mean(qs_c, p=-4.0)
    return qs_c, q_c



class BoidsEnv(gym.Env):

    """
    Description:
        A flock of boids is simulated in a toroidal environment. Each boid has a position and velocity. The boids
        are controlled by the user. The user can control the acceleration and angular velocity of each boid.

    Source:
        This environment corresponds to the boids simulation in the classic computer graphics paper by Craig Reynolds.
    
    Observation:
        Type: Box(4*numBoids)
        Num     Observation               Min                     Max
        0       x position                0                       1
        1       y position                0                       1
        2       x velocity                -max_speed              max_speed
        3       y velocity                -max_speed              max_speed
    
    Actions:
        Type: Box(2*numBoids)
        Num     Action                      Min                     Max
        0       acceleration                -max_acc                max_acc
        1       angular velocity            -max_angular_speed      max_angular_speed
        
    """


    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 20,
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
        obs_low = np.tile(np.concatenate([min_pos, -max_vel]), numBoids)
        obs_high = np.tile(np.concatenate([max_pos, max_vel]), numBoids)
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

        self.state = self.difference_eq(self.state, u)
        reward = self.cmorl(self.state, u, self)
        return self.state, reward, False, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.state = self.observation_space.sample()
        if self.render_mode == "human":
            self.render()
        return self.state, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        for i in range(0, self.state.shape[0], 4):
            x = int(self.state[i]*self.screen_dim)
            y = int(self.state[i+1]*self.screen_dim)
            x2 = int(x + self.state[i+2]*self.screen_dim)
            y2 = int(y + self.state[i+3]*self.screen_dim)
            pygame.draw.line(self.surf, (0, 0, 0), (x, y), (x2, y2), 1)
            pygame.gfxdraw.filled_circle(self.surf, x, y, 5, (0, 0, 0))
        
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def normed_angular_distance(a, b):
    diff = (b - a + np.pi) % (2 * np.pi) - np.pi
    return np.abs(diff + 2 * np.pi if diff < -np.pi else diff) / np.pi


if __name__ == "__main__":
    env = BoidsEnv(numBoids=1, render_mode="human")
    env.reset()
    env.state = tf.constant([0.5, 0.5, 0.0, 0.0], dtype=tf.float32)
    for _ in range(1000):
        env.step(env.action_space.sample())
        env.render()
    env.close()