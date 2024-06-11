import numpy as np
import tensorflow as tf

from cmorl.utils.loss_composition import p_mean
from cmorl.utils.reward_utils import  Transition
from gymnasium.envs.box2d.lunar_lander import LunarLander
from gymnasium.envs.mujoco.reacher import ReacherEnv
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

def mujoco_multi_dim_reward_joints_x_velocity(transition: Transition, env: MujocoEnv, speed_multiplier=1.0):
    action_rw = (1.0 - np.abs(transition.action))
    if not hasattr(env, "prev_xpos"):
        env.prev_xpos = np.copy(env.data.xpos) # type: ignore
    x_velocities = (env.data.xpos - env.prev_xpos) / env.dt # type: ignore
    env.prev_xpos = np.copy(env.data.xpos) # type: ignore
    forward_rw = np.clip(x_velocities[1:, 0]*speed_multiplier, 0.0, 1.0)
    return np.hstack([forward_rw, action_rw**0.5])

def composed_reward_fn(transition, env):
    rew_vec = mujoco_multi_dim_reward_joints_x_velocity(transition, env)
    reward = p_mean(rew_vec, p=-4.0)
    return reward

def multi_dim_reacher(transition: Transition, env: ReacherEnv) -> np.ndarray:
    reward_performance = 1.0 - np.clip(np.abs(transition.next_state[-3:-1])/0.2, 0.0, 1.0)
    reward_actuation = 1 - np.abs(transition.action)
    # print(transition.next_state[-3:-1])
    # print("rw:", reward_performance)
    rw_vec = np.concatenate([reward_performance**2.0, reward_actuation], dtype=np.float32)
    return rw_vec

def normed_angular_distance(a, b):
    diff = (b - a + np.pi) % (2 * np.pi) - np.pi
    return np.abs(diff + 2 * np.pi if diff < -np.pi else diff) / np.pi

def multi_dim_pendulum(transition: Transition, env, setpoint) -> np.ndarray:
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

def lunar_lander_rw(transition: Transition, env: LunarLander)  -> np.ndarray:
    dist_x = np.clip(
        (1.0 - np.abs(transition.next_state[0]) / env.observation_space.high[0]), 0.0, 1.0 # type: ignore
    )
    very_near_dist_x = np.clip(
        (1.0 - 10*np.abs(transition.next_state[0]) / env.observation_space.high[0]), 0.0, 1.0 # type: ignore
    )
    dist_y = np.clip((1.0 - np.abs(transition.next_state[1]) / env.observation_space.high[1]), 0.0, 1.0) # type: ignore
    very_near_dist_y = np.clip((1.0 - 10*np.abs(transition.next_state[1]) / env.observation_space.high[1]), 0.0, 1.0) # type: ignore
    vx = transition.next_state[3]
    vy = transition.next_state[4]
    minize_speed_near_ground = 1.0 - np.clip(np.linalg.norm([vx, vy])*10.0, 0.0, 1.0)
    first_leg = transition.next_state[6]*minize_speed_near_ground
    second_leg = transition.next_state[7]*minize_speed_near_ground
    fuel_costs = 1.0 - np.abs(transition.action)

    return np.concatenate([[very_near_dist_x, very_near_dist_y], fuel_costs, [first_leg, second_leg]])

@tf.function
def lander_composer(q_values, p_batch=0, p_objectives=-4.0):
    qs_c = p_mean(q_values, p=p_batch, axis=0)
    q_c = p_mean([qs_c[0], qs_c[1], 0.5+0.5*qs_c[2], 0.1+0.9*qs_c[3]**0.5, 0.1+0.9*qs_c[4]**0.5], p=p_objectives)
    # q_c = p_mean(qs_c, p=p_objectives)
    return qs_c, q_c