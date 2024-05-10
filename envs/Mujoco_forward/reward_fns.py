import numpy as np

from cmorl.utils.loss_composition import p_mean
from cmorl.utils.reward_utils import  Transition

def multi_dim_reward_joints(transition: Transition, env):
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
    all_rw = multi_dim_reward_joints(transition, env)
    forward_rw = all_rw[0]
    action_rw = all_rw[1:]
    ctrl_reward = p_mean(action_rw, p=p)

    rw_vec = np.array([forward_rw, ctrl_reward], dtype=np.float32)
    return rw_vec

def composed_reward_fn(transition, env):
    rew_vec = multi_dim_reward_joints(transition, env)
    reward = p_mean(rew_vec, p=-4.0)
    return reward