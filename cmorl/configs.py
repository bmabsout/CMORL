from functools import partial
from typing import Callable

import gymnasium
from gymnasium.wrappers import TimeLimit

from cmorl.rl_algs.ddpg.hyperparams import HyperParams, combine, default_hypers
from cmorl.utils.reward_utils import CMORL, perf_schedule
from cmorl import reward_fns

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
        reward_fns.mujoco_CMORL(num_actions=8),
        HyperParams(env_args={"use_contact_forces": True}, epochs=100),
    ),
    "Hopper-v4": Config(
        reward_fns.mujoco_CMORL(speed_multiplier=0.5, num_actions=3),
        HyperParams(epochs=60, act_noise=0.1),
    ),
    "HalfCheetah-v4": Config(
        reward_fns.mujoco_CMORL(speed_multiplier=0.15, num_actions=6),
        HyperParams(epochs=200, act_noise=0.05),
    ),
    "Pendulum-v1": Config(
        CMORL(partial(reward_fns.multi_dim_pendulum, setpoint=0.0))
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
            epochs=50,
            q_d=1.0,
            # p_objectives=0.0,
            # p_batch=1.0,
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
}

def get_env_and_config(env_name: str) -> tuple[Callable[..., gymnasium.Env], Config]:
    config = env_configs.get(env_name, Config())
    make_env = lambda **kwargs: config.wrapper(gymnasium.make(env_name, **{**kwargs, **config.hypers.env_args}))
    return make_env, config 
