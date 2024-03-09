import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}

import tensorflow as tf
from cmorl.utils.loss_composition import p_mean
from cmorl.utils.reward_utils import CMORL, RewardFnType


def multi_dim_reward(state, action, env: "ReacherEnv"):
    vec = env.get_body_com("fingertip") - env.get_body_com("target")
    reward_performance = np.clip(1.0 - np.linalg.norm(vec) / 0.4, 0.0, 1.0) ** 2.0
    reward_actuation = 1 - np.square(action).sum() / 2
    rw_vec = np.array([reward_performance**2, reward_actuation**0.5], dtype=np.float32)
    return rw_vec


def composed_reward_fn(state, action, env):
    rew_vec = multi_dim_reward(state, action, env)
    reward = p_mean(rew_vec, p=-4.0)
    return reward


@tf.function
def q_composer(q_values):
    # q1_c = q_values[0] ** 2
    # q2_c = q_values[1]
    # q_values = tf.stack([q1_c, q2_c], axis=0)
    qs_c = tf.reduce_mean(q_values, axis=0)
    q_c = p_mean(qs_c, p=-4.0)
    return qs_c, q_c


def sample_point_in_circle(np_random, radius, bias=0.0):
    angle = np_random.uniform() * np.pi * 2
    r = radius * np_random.uniform() ** (2.0**bias / 2.0)
    return r * np.sin(angle), r * np.cos(angle)


class ReacherEnv(MujocoEnv, utils.EzPickle):
    """
    ## Description
    "Reacher" is a two-jointed robot arm. The goal is to move the robot's end effector (called *fingertip*) close to a
    target that is spawned at a random position.
    ## Action Space
    The action space is a `Box(-1, 1, (2,), float32)`. An action `(a, b)` represents the torques applied at the hinge joints.
    | Num | Action                                                                          | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit |
    |-----|---------------------------------------------------------------------------------|-------------|-------------|--------------------------|-------|------|
    | 0   | Torque applied at the first hinge (connecting the link to the point of fixture) | -1 | 1 | joint0  | hinge | torque (N m) |
    | 1   |  Torque applied at the second hinge (connecting the two links)                  | -1 | 1 | joint1  | hinge | torque (N m) |
    ## Observation Space
    Observations consist of
    - The cosine of the angles of the two arms
    - The sine of the angles of the two arms
    - The coordinates of the target
    - The angular velocities of the arms
    - The vector between the target and the reacher's fingertip (3 dimensional with the last element being 0)
    The observation is a `ndarray` with shape `(11,)` where the elements correspond to the following:
    | Num | Observation                                                                                    | Min  | Max | Name (in corresponding XML file) | Joint | Unit                     |
    | --- | ---------------------------------------------------------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
    | 0   | cosine of the angle of the first arm                                                           | -Inf | Inf | cos(joint0)                      | hinge | unitless                 |
    | 1   | cosine of the angle of the second arm                                                          | -Inf | Inf | cos(joint1)                      | hinge | unitless                 |
    | 2   | sine of the angle of the first arm                                                             | -Inf | Inf | sin(joint0)                      | hinge | unitless                 |
    | 3   | sine of the angle of the second arm                                                            | -Inf | Inf | sin(joint1)                      | hinge | unitless                 |
    | 4   | x-coordinate of the target                                                                    | -Inf | Inf | target_x                         | slide | position (m)             |
    | 5   | y-coordinate of the target                                                                    | -Inf | Inf | target_y                         | slide | position (m)             |
    | 6   | angular velocity of the first arm                                                              | -Inf | Inf | joint0                           | hinge | angular velocity (rad/s) |
    | 7   | angular velocity of the second arm                                                             | -Inf | Inf | joint1                           | hinge | angular velocity (rad/s) |
    | 8   | x-value of position_fingertip - position_target                                                | -Inf | Inf | NA                               | slide | position (m)             |
    | 9   | y-value of position_fingertip - position_target                                                | -Inf | Inf | NA                               | slide | position (m)             |
    | 10  | z-value of position_fingertip - position_target (constantly 0 since reacher is 2d and z is same for both) | -Inf | Inf | NA                               | slide | position (m)             |
    Most Gym environments just return the positions and velocity of the
    joints in the `.xml` file as the state of the environment. However, in
    reacher the state is created by combining only certain elements of the
    position and velocity, and performing some function transformations on them.
    If one is to read the `.xml` for reacher then they will find 4 joints:
    | Num | Observation                 | Min      | Max      | Name (in corresponding XML file) | Joint | Unit               |
    |-----|-----------------------------|----------|----------|----------------------------------|-------|--------------------|
    | 0   | angle of the first arm      | -Inf     | Inf      | joint0                           | hinge | angle (rad)        |
    | 1   | angle of the second arm     | -Inf     | Inf      | joint1                           | hinge | angle (rad)        |
    | 2   | x-coordinate of the target  | -Inf     | Inf      | target_x                         | slide | position (m)       |
    | 3   | y-coordinate of the target  | -Inf     | Inf      | target_y                         | slide | position (m)       |
    ## Rewards
    The reward consists of two parts:
    - *reward_distance*: This reward is a measure of how far the *fingertip*
    of the reacher (the unattached end) is from the target, with a more negative
    value assigned for when the reacher's *fingertip* is further away from the
    target. It is calculated as the negative vector norm of (position of
    the fingertip - position of target), or *-norm("fingertip" - "target")*.
    - *reward_control*: A negative reward for penalising the walker if
    it takes actions that are too large. It is measured as the negative squared
    Euclidean norm of the action, i.e. as *- sum(action<sup>2</sup>)*.
    The total reward returned is ***reward*** *=* *reward_distance + reward_control*
    Unlike other environments, Reacher does not allow you to specify weights for the individual reward terms.
    However, `info` does contain the keys *reward_dist* and *reward_ctrl*. Thus, if you'd like to weight the terms,
    you should create a wrapper that computes the weighted reward from `info`.
    ## Starting State
    All observations start in state
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    with a noise added for stochasticity. A uniform noise in the range
    [-0.1, 0.1] is added to the positional attributes, while the target position
    is selected uniformly at random in a disk of radius 0.2 around the origin.
    Independent, uniform noise in the
    range of [-0.005, 0.005] is added to the velocities, and the last
    element ("fingertip" - "target") is calculated at the end once everything
    is set. The default setting has a framerate of 2 and a *dt = 2 * 0.01 = 0.02*
    ## Episode End
    The episode ends when any of the following happens:
    1. Truncation: The episode duration reaches a 50 timesteps (with a new random target popping up if the reacher's fingertip reaches it before 50 timesteps)
    2. Termination: Any of the state space values is no longer finite.
    ## Arguments
    No additional arguments are currently supported (in v2 and lower),
    but modifications can be made to the XML file in the assets folder
    (or by changing the path to a modified XML file in another folder)..
    ```python
    import gymnasium as gym
    env = gym.make('Reacher-v4')
    ```
    There is no v3 for Reacher, unlike the robot environments where a v3 and
    beyond take `gymnasium.make` kwargs such as `xml_file`, `ctrl_cost_weight`, `reset_noise_scale`, etc.
    ## Version History
    * v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1.3
    * v2: All continuous control environments now use mujoco-py >= 1.50
    * v1: max_time_steps raised to 1000 for robot based tasks (not including reacher, which has a max_time_steps of 50). Added reward_threshold to environments.
    * v0: Initial versions release (1.0.0)
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(
        self,
        goal_distance=0.2,
        bias=0.0,
        reward_fn: RewardFnType = multi_dim_reward,  # type: ignore
        **kwargs
    ):
        self.goal_distance = goal_distance
        self.bias = bias
        utils.EzPickle.__init__(self, goal_distance=goal_distance, bias=bias, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            "reacher.xml",
            2,
            observation_space=observation_space,
            # default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs
        )
        reward_dim = reward_fn(
            self.observation_space.sample(), self.action_space.sample(), self  # type: ignore
        ).shape[  # type: ignore
            0
        ]  # type: ignore
        self.cmorl = CMORL(reward_dim, reward_fn, q_composer)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()
        reward = self.cmorl(obs, action, self)
        return (
            obs,
            reward,
            False,
            False,
            # dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl),
            {},
        )

    def reset_model(self):
        qpos = (
            self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            + self.init_qpos
        )
        self.goal = sample_point_in_circle(
            self.np_random, self.goal_distance, self.bias
        )
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.data.qpos.flat[:2]
        return np.concatenate(
            [
                np.cos(theta),
                np.sin(theta),
                self.data.qpos.flat[2:],
                self.data.qvel.flat[:2],
                self.get_body_com("fingertip") - self.get_body_com("target"),
            ]
        )
