from functools import partial
import numpy as np
import tensorflow as tf

from cmorl.utils.loss_composition import p_mean, then
from cmorl.utils.reward_utils import  CMORL, Transition
from gymnasium.envs.box2d.lunar_lander import LunarLander
from gymnasium.envs.mujoco.reacher import ReacherEnv
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

from envs.Bittle.opencat_gym_env import OpenCatGymEnv

def mujoco_multi_dim_reward_joints_x_velocity(transition: Transition, env: MujocoEnv, speed_multiplier=1.0):
    action_rw = (1.0 - np.abs(transition.action))
    if not hasattr(env, "prev_xpos"):
        env.prev_xpos = np.copy(env.data.xpos) # type: ignore
    x_velocities = (env.data.xpos - env.prev_xpos) / env.dt # type: ignore
    env.prev_xpos = np.copy(env.data.xpos) # type: ignore
    forward_rw = np.clip(x_velocities[1:, 0]*speed_multiplier, 0.0, 1.0)
    return np.hstack([forward_rw, action_rw])


def bittle_rw(transition: Transition, env: OpenCatGymEnv):
    action_rw = (1.0 - np.abs(transition.action))
    forward = transition.info.get("forward", 0.0)
    change_direction = transition.info.get("change_direction", env.action_space.low*0.0)
    body_stability = transition.info.get("body_stability", np.zeros(3))
    # return np.hstack([[forward], change_direction, action_rw])
    return np.hstack([[forward], change_direction])
    # return np.array([forward])

def mujoco_CMORL(speed_multiplier=1.0, num_actions=3):
    @tf.function
    def mujoco_composer(q_values, p_batch=0, p_objectives=-4.0):
        qs_c = p_mean(q_values, p=p_batch, axis=0)
        forward = p_mean(qs_c[0:-num_actions], p=0.0)
        action = 1-(1-p_mean(qs_c[-num_actions:], p=0.0))**5.0
        q_c = p_mean((forward, action), p=p_objectives)
        return qs_c, (1.0 - (1-q_c)**2.0)
    return CMORL(partial(mujoco_multi_dim_reward_joints_x_velocity, speed_multiplier=speed_multiplier), mujoco_composer)

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
    nearness = 1.0 - np.clip(
        np.linalg.norm(transition.next_state[0:2]), 0.0, 1.0 # type: ignore
    )
    very_nearness = 1.0 - np.clip(
        10*np.linalg.norm(transition.next_state[0:2]), 0.0, 1.0 # type: ignore
    )
    speed = transition.next_state[2:4] / env.observation_space.high[2:4] # type: ignore
    minize_speed_near_ground = 1.0 - np.clip(np.linalg.norm(speed)*10.0, 0.0, 1.0)
    legs = transition.next_state[6:8]*minize_speed_near_ground
    fuel_costs = 1.0 - np.abs(transition.action/env.action_space.high) # type: ignore
    # return np.concatenate([[nearness**4.0, very_nearness**2.0], fuel_costs, legs])
    return np.concatenate([[nearness**4.0], fuel_costs, legs])
    # return np.concatenate([[nearness**4.0]])
    # return legs

@tf.function
def lander_composer(q_values, p_batch=0, p_objectives=-4.0):
    qs_c = p_mean(q_values, p=p_batch, axis=0)
    # nearness = p_mean(qs_c[0:2]**0.5, p=0.0)
    nearness=(1-(1-qs_c[0])**4.0)
    legs_touch = p_mean(qs_c[3:5], p=0.0)**0.25
    fuel_cost = 1.0 - (1-p_mean(qs_c[1:3], p=0.0))**10
    # schedule = then(nearness, legs_touch, slack=1e-3)
    q_c = p_mean([nearness, legs_touch, fuel_cost], p=p_objectives)
    # q_c = legs_touch
    # q_c = 1-(1-p_mean([qs_c[0], qs_c[1], qs_c[2]**0.2, 0.01+0.99*qs_c[3], 0.01+0.99*qs_c[4]], p=p_objectives))**2.0
    # # q_c = p_mean(qs_c, p=p_objectives)
    return tf.concat([qs_c, [nearness, legs_touch]],axis=0), q_c #(1.0 - (1.0 - q_c)**2.0)