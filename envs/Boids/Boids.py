from typing import Optional

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled

import tensorflow as tf
from cmorl.utils.loss_composition import p_mean
from cmorl.utils.reward_utils import CMORL, RewardFnType
from .toroid_utils import toroidal_distance, toroidal_pairwise_dist

@tf.function
def flatten_upper_triangle(matrix):
    """
        Flattens the upper triangle of a matrix.
        Args:
            matrix,    [m,n] matrix
        Returns:
            flattened,    [(m*n-1)/2] vector
    """
    upper_triangle = tf.where(tf.linalg.band_part(tf.ones_like(matrix), -1, 0) == 0.0, True, False)
    return tf.boolean_mask(matrix, upper_triangle)

def composed_reward_fn(flat_state, flat_u, env: "BoidsEnv"):
    return 0.0

@tf.function
def convert_state_to_dict(flat_state: tf.Tensor, numBoids: int):
    state = tf.transpose(tf.reshape(flat_state, (numBoids, 4)))
    return {"pos": state[:2], "vel": state[2:]}

@tf.function
def convert_action_to_dict(flat_u: tf.Tensor, numBoids: int):
    u = tf.transpose(tf.reshape(flat_u, (numBoids, 2)))
    return {"setpoint_vel_mag": u[0], "angle_change": u[1]}

def multi_dim_reward(flat_state, flat_u, env: "BoidsEnv"):
    state = convert_state_to_dict(flat_state, env.numBoids)
    action = convert_action_to_dict(flat_u, env.numBoids)
    go_fast = tf.norm(state["vel"], axis=0)/env.max_speed
    max_toroidal_distance = (0.5**2.0 + 0.5**2.0)**0.5 # toroidal distance means at worst we are (0.5, 0.5) away
    dists = flatten_upper_triangle(toroidal_pairwise_dist(state["pos"], state["pos"]))/ max_toroidal_distance
    minimize_distance = 1.0 - dists # tf.where(dists > 0.7, 0.0, (0.7-dists)/0.7)
    avoid_collisions = tf.where(dists < 0.025, dists/0.025, 1.0)
    small_actions = 1.0 - tf.abs(action["angle_change"])/convert_action_to_dict(env.action_space.high, env.numBoids)["angle_change"]
    return np.concatenate([go_fast, minimize_distance**2.0, avoid_collisions, small_actions])

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
        0       setpoint velocity           0.0                     max_speed
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
        numBoids: int = 5,
        max_speed = 1.0,
        max_angular_speed = 360.0*np.pi/180.0, # rads/s
        reward_fn: RewardFnType = multi_dim_reward,
    ):
        self.max_speed = max_speed
        self.numBoids = numBoids

        self.dt = 1.0/(self.metadata["render_fps"] if render_mode == "human" else 20.0)

        self.render_mode = render_mode

        self.screen_dim = 500
        self.screen = screen
        self.clock = None
        self.isopen = True
        
        max_vel = np.array([max_speed, max_speed], dtype=np.float32)
        max_pos = np.array([1.0, 1.0], dtype=np.float32)
        min_pos = np.array([0.0, 0.0], dtype=np.float32)
        obs_low = np.tile(np.concatenate([min_pos, -max_vel]), numBoids)
        obs_high = np.tile(np.concatenate([max_pos, max_vel]), numBoids)
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.tile([0.0, -max_angular_speed], numBoids),
            high=np.tile([max_speed, max_angular_speed], numBoids),
            dtype=np.float32
        )

        reward_dim = reward_fn(
            self.observation_space.sample(), self.action_space.sample(), self
        ).shape[0]
        self.cmorl = CMORL(reward_dim, reward_fn, q_composer)

    def get_obs(self):
        return self.state


    def step(self, u):
        self.state = difference_eq(self.state, u, self.numBoids, self.dt, self.max_speed)
        reward = self.cmorl(self.state, u, self)
        return self.get_obs(), reward, False, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.state = options["state"] if options else self.observation_space.sample()
        if self.render_mode == "human":
            self.render()
        return self.get_obs(), {}

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
        
        

        self.drawBoids()

        if self.render_mode == "human":
            # use mouse position to set the first boid's pos
            # mouse_pos = pygame.mouse.get_pos()
            # self.state = np.array(self.state, dtype=np.float32)
            # self.state[0] = 1.0 + mouse_pos[0]/self.screen_dim
            # self.state[1] = 1.0 - mouse_pos[1]/self.screen_dim
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        
    def drawBoids(self):
        import pygame
        import pygame.gfxdraw
        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))
        screen_state = convert_state_to_dict(tf.cast(self.state*self.screen_dim, tf.int32), self.numBoids)
        for i in range(self.numBoids):
            screen_pos = screen_state["pos"][:, i]
            screen_vel = screen_state["vel"][:, i]
            pygame.gfxdraw.filled_circle( self.surf, *screen_pos, 5, (0, 0, 0))
            pygame.draw.line(self.surf, (0, 0, 0), screen_pos,screen_pos+screen_vel, 1)
        
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False




@tf.function
def q_composer(q_values):
    qs_c = p_mean(q_values, p=0.0, axis=0)
    q_c = p_mean(qs_c, p=-4.0)
    return qs_c, q_c


@tf.function
def difference_eq(flat_state: tf.Tensor, flat_u: tf.Tensor, numBoids: int, dt: float, max_speed: float):
    # state is a tensor of shape (4*numBoids)
    # u is a tensor of shape (2*numBoids)
    state = convert_state_to_dict(flat_state, numBoids)
    action = convert_action_to_dict(flat_u, numBoids)
    # get angle of vel
    curr_angle = tf.atan2(state["vel"][1], state["vel"][0])
    new_angle = curr_angle + action["angle_change"]*dt
    new_anglexy = tf.stack([tf.cos(new_angle), tf.sin(new_angle)], axis=0)
    # using leapfrog integration
    vel_mag = tf.norm(state["vel"], axis=0)
    new_vel_mag = vel_mag*(1.0 - dt) + action["setpoint_vel_mag"]*dt
    new_vel = new_anglexy*new_vel_mag
    # new_vel = vel*(1.0 - dt) + (setpoint_vel - vel)*dt
    # clamp velocity
    new_vel = tf.clip_by_norm(new_vel, max_speed, axes=[0])
    new_pos = state["pos"] + new_vel*dt
    # let the position wrap ala torus
    new_pos = tf.math.floormod(new_pos, 1.0)
    return tf.reshape(tf.transpose(tf.concat([new_pos, new_vel], axis=0)), [-1])

if __name__ == "__main__":
    env = BoidsEnv(numBoids=2, render_mode="human")
    state = np.tile([0.5, 0.5, 0.0, 0.0], env.numBoids)
    state[0] = 0.0
    state[1] = 0.0
    state[4] = 0.05
    state[5] = 0.05
    env.reset(options={"state": tf.constant(state, dtype=tf.float32)})
    # env.state = tf.constant([0.5, 0.5, 0.0, 0.0], dtype=tf.float32)
    for _ in range(1000):
        # action = np.tile(np.array([0.0, 0.0], dtype=np.float32), env.numBoids)
        action = env.action_space.sample()
        # action[0] = 1.0
        # action[1] = env.action_space.high[1]
        env.step(action)
        env.render()
    env.close()

gym.register("Boids-v0", BoidsEnv) # type:ignore