from functools import partial
from typing import Callable

import gymnasium
from gymnasium.wrappers.time_limit import TimeLimit
import numpy as np

from cmorl.rl_algs.ddpg.hyperparams import HyperParams, combine, default_hypers
from cmorl.utils.reward_utils import CMORL, perf_schedule
from cmorl import reward_fns
from envs import Boids

class Config:
    def __init__(self, cmorl: CMORL | None = None, hypers: HyperParams = HyperParams(), wrapper = gymnasium.Wrapper):
        self.cmorl = cmorl
        self.hypers = combine(default_hypers(), hypers)
        self.wrapper = wrapper

class FixSleepingLander(gymnasium.Wrapper):
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        if not self.env.lander.awake:
            truncated = True
            done = False
        return obs, reward, done, truncated, info
    
class ForcedTimeLimit(TimeLimit):
    def step(self, action):
        obs, reward, done, _, info = super().step(action)
        truncated = self._elapsed_steps >= self._max_episode_steps
        return obs, reward, done, truncated, info

env_configs: dict[str, Config] = {
    "Reacher-v4": Config(
        CMORL(reward_fns.multi_dim_reacher),
        HyperParams(
            ac_kwargs={
                "obs_normalizer": [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 2.0, 2.0, 2.0],
            },
        ),
        wrapper=partial(ForcedTimeLimit, max_episode_steps=200),
    ),
    "Ant-v4": Config(
        reward_fns.mujoco_CMORL(num_actions=8, speed_multiplier=0.5),
        HyperParams(env_args={"use_contact_forces": True}, epochs=100, act_noise=0.05),
    ),
    "Hopper-v4": Config(
        reward_fns.mujoco_CMORL(num_actions=3, speed_multiplier=0.7),
        HyperParams(
            ac_kwargs = {
                "critic_hidden_sizes": [512, 512],
                "actor_hidden_sizes": [32, 32],
            },
            epochs=20,
            act_noise=0.1,
            p_objectives=-1.0,
            p_batch=1.0,
            q_batch=1.0,
            q_objectives=1.0),
    ),
    "HalfCheetah-v4": Config(
        reward_fns.mujoco_CMORL(num_actions=6, speed_multiplier=0.1),
        HyperParams(epochs=200, act_noise=0.0, p_objectives=0.0),
    ),
    "Pendulum-v1": Config(
        CMORL(partial(reward_fns.multi_dim_pendulum, setpoint=0.0)),
        HyperParams(
            ac_kwargs = {
                "critic_hidden_sizes": [128, 128],
                "actor_hidden_sizes": [32, 32],
            },
            epochs=10,
            pi_lr=3e-3,
            q_lr=3e-3,
            act_noise=0.0,
            p_objectives=-4.0,
        )
    ),
    "Pendulum-custom": Config(
        CMORL(partial(reward_fns.multi_dim_pendulum, setpoint=0.0))
    ),
    "LunarLanderContinuous-v2": Config(
        CMORL(reward_fns.lunar_lander_rw, reward_fns.lander_composer),
        HyperParams(
            ac_kwargs={
                "obs_normalizer": gymnasium.make("LunarLanderContinuous-v2").observation_space.high,
            },
            epochs=30,
            p_objectives=-1.0,
        ),
        wrapper=lambda x: TimeLimit(FixSleepingLander(x), max_episode_steps=400),
    ),
    "Bittle-custom": Config(
        CMORL(reward_fns.bittle_rw),
        HyperParams(
            max_ep_len=400,
            env_args={"observe_joints": True},
            # qd_power=0.5
        ),
    ),
    "Boids-v0": Config(
        CMORL(Boids.multi_dim_reward, randomization_schedule=perf_schedule),
        HyperParams(
            max_ep_len=400,
        )
    ),
}

def get_env_and_config(env_name: str) -> tuple[Callable[..., gymnasium.Env], Config]:
    config = env_configs.get(env_name, Config())
    make_env = lambda **kwargs: config.wrapper(gymnasium.make(env_name, **{**kwargs, **config.hypers.env_args}))
    return make_env, config 
