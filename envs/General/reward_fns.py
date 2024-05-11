from functools import partial
import numpy as np

from cmorl.utils.loss_composition import p_mean
from cmorl.utils.reward_utils import  Transition
from envs.Reacher.reacher import ReacherEnv

def mujoco_multi_dim_reward_joints_x_velocity(transition: Transition, env):
    action_rw = (1.0 - np.abs(transition.action)) # *0.5+0.5
    tanhed_speed = 0.0
    if ("x_velocity" in transition.info):
        tanhed_speed = np.tanh(transition.info["x_velocity"])
    else:
        print("Warning: x_velocity not found in transition.info")

    # if the forward_reward is negative, it should be scaled between 0 and 0.1
    forward_rw = (tanhed_speed+1.0)*0.1 if tanhed_speed < 0.0 else 0.1+tanhed_speed*0.9
    return np.hstack([[forward_rw], action_rw])

def two_dim_reward(transition, env, p=0.0):
    all_rw = mujoco_multi_dim_reward_joints_x_velocity(transition, env)
    forward_rw = all_rw[0]
    action_rw = all_rw[1:]
    ctrl_reward = p_mean(action_rw, p=p)

    rw_vec = np.array([forward_rw, ctrl_reward], dtype=np.float32)
    return rw_vec

def composed_reward_fn(transition, env):
    rew_vec = mujoco_multi_dim_reward_joints_x_velocity(transition, env)
    reward = p_mean(rew_vec, p=-4.0)
    return reward

def multi_dim_reacher(transition: Transition, env: ReacherEnv):
    fingertip_target_error = transition.next_state[-3:-1]
    reward_performance = np.clip(1.0 - np.linalg.norm(fingertip_target_error) / 0.4, 0.0, 1.0)
    reward_actuation = 1 - np.abs(transition.action)
    rw_vec = np.concatenate([[reward_performance], reward_actuation], dtype=np.float32)
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


reward_fns = {
    "Ant-v4": mujoco_multi_dim_reward_joints_x_velocity,
    "HalfCheetah-v4": mujoco_multi_dim_reward_joints_x_velocity,
    "Hopper-v4": mujoco_multi_dim_reward_joints_x_velocity,
    "Reacher-v4": multi_dim_reacher,
    "Pendulum-v1": partial(multi_dim_pendulum, setpoint=0.0),
}