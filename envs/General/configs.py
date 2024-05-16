from functools import partial

from cmorl.rl_algs.ddpg.hyperparams import HyperParams
from cmorl.utils.reward_utils import CMORL
from envs.LunarLander.lunar_lander import LunarLander
import reward_fns

class Config:
    def __init__(self, cmorl: CMORL | None = None, hypers: HyperParams = HyperParams()):
        self.cmorl = cmorl
        self.hypers = hypers

env_configs: dict[str, Config] = {
    "Reacher-v4": Config(
        CMORL(reward_fns.multi_dim_reacher),
        HyperParams(
            ac_kwargs={
                "obs_normalizer": [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 2.0, 2.0, 2.0],
            },
        ),
    ),
    "Ant-v4": Config(
        CMORL(partial(reward_fns.mujoco_multi_dim_reward_joints_x_velocity)),
        HyperParams(pi_lr=1e-3, q_lr=1e-3, env_args={"use_contact_forces": True}, epochs=100, ac_kwargs={"critic_hidden_sizes": (512, 512), "actor_hidden_sizes": (64, 64)}),
    ),
    "Hopper-v4": Config(
        CMORL(partial(reward_fns.mujoco_multi_dim_reward_joints_x_velocity, speed_multiplier=2.0)),
        HyperParams(gamma=0.99, pi_lr=1e-3, q_lr=1e-3, epochs=60),
    ),
    "HalfCheetah-v4": Config(
        CMORL(partial(reward_fns.mujoco_multi_dim_reward_joints_x_velocity, speed_multiplier=0.2)),
        HyperParams(gamma=0.99, epochs=20, pi_lr=1e-3, q_lr=1e-3),
    ),
    "Pendulum-v1": Config(
        CMORL(partial(reward_fns.multi_dim_pendulum, setpoint=0.0))
    ),
    "LunarLanderContinuous-v2": Config(
        CMORL(reward_fns.lunar_lander_rw),
        HyperParams(
            ac_kwargs={
                "obs_normalizer": LunarLander().observation_space.high
            },
            gamma=0.99,
            max_ep_len=300,
            epochs=60,
            pi_lr=1e-4,
            q_lr=1e-4,
            p_Q_batch=1.0,
        ),
    ),
}

def get_config(env_name: str) -> Config:
    return env_configs.get(env_name, Config())