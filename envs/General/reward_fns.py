import numpy as np

from cmorl.utils.loss_composition import p_mean
from cmorl.utils.reward_utils import  Transition
from gymnasium.envs.box2d.lunar_lander import LunarLander
from gymnasium.envs.mujoco.reacher import ReacherEnv

def mujoco_multi_dim_reward_joints_x_velocity(transition: Transition, env, speed_multiplier=1.0):
    action_rw = (1.0 - np.abs(transition.action))
    if not hasattr(env, "prev_xpos"):
        env.prev_xpos = np.copy(env.data.xpos)
    x_velocities = (env.data.xpos - env.prev_xpos) / env.dt
    env.prev_xpos = np.copy(env.data.xpos)
    forward_rw = np.clip(x_velocities[1:, 0]*speed_multiplier, 0.0, 1.0)
    return np.hstack([forward_rw, action_rw**0.5])

def composed_reward_fn(transition, env):
    rew_vec = mujoco_multi_dim_reward_joints_x_velocity(transition, env)
    reward = p_mean(rew_vec, p=-4.0)
    return reward

def multi_dim_reacher(transition: Transition, env: ReacherEnv):
    reward_performance = 1.0 - np.clip(np.abs(transition.next_state[-3:-1])/0.2, 0.0, 1.0)
    reward_actuation = 1 - np.abs(transition.action)
    # print(transition.next_state[-3:-1])
    # print("rw:", reward_performance)
    rw_vec = np.concatenate([reward_performance**2.0, reward_actuation], dtype=np.float32)
    return rw_vec

def normed_angular_distance(a, b):
    diff = (b - a + np.pi) % (2 * np.pi) - np.pi
    return np.abs(diff + 2 * np.pi if diff < -np.pi else diff) / np.pi

def multi_dim_pendulum(transition: Transition, env, setpoint):
    # check if action is an array or a scalar
    u = np.squeeze(transition.action)
    th, thdot = env.state  # th := theta
    angle_rw = 1.0 - normed_angular_distance(th, setpoint)

    # Normalizing the torque to be in the range [0, 1]
    normalized_u = u / env.max_torque
    normalized_u = abs(normalized_u)
    actuation_rw = 1.0 - normalized_u
    
    # Merge the angle reward and the normalized torque into a single reward vector
    thdot_rw = 1.0 - np.abs(thdot) / env.max_speed
    rw_vec = np.array([angle_rw, actuation_rw], dtype=np.float32)
    return rw_vec

def lunar_lander_rw(transition: Transition, env: LunarLander):
    dist_x = np.clip(
        (1.0 - 3 * np.abs(transition.next_state[0]) / env.observation_space.high[0]), 0.0, 1.0
    )
    dist_y = np.clip((1.0 - np.abs(transition.next_state[1]) / env.observation_space.high[1]), 0.0, 1.0)
    first_leg = 0.1 + 0.9 * transition.next_state[6]
    second_leg = 0.1 + 0.9 * transition.next_state[7]

    return np.array([dist_x**2, dist_y**2, first_leg**0.5, second_leg**0.5])
