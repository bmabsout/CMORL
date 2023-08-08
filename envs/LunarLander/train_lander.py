import argparse
import json
import os
import time
import pickle
from anchored_rl.rl_algs.ddpg.ddpg import ddpg, HyperParams
from anchored_rl.utils import args_utils
from anchored_rl.utils import train_utils
import lunar_lander
import tensorflow as tf

lander_serializer = lambda: args_utils.Arg_Serializer.join(args_utils.Arg_Serializer(
    abbrev_to_args= {
        # 'd': args_utils.Serialized_Argument(name='--distance', type=float, default=0.2, help='radius of points from the center'),
        # 'b': args_utils.Serialized_Argument(name='--bias', type=float, default=0.0, help='bias of points from the center'),
    }), args_utils.default_serializer())

class WithStrPolyDecay(tf.optimizers.schedules.PolynomialDecay):
    def __str__(self):
        return f"({self.initial_learning_rate},{self.end_learning_rate})"

def train(cmd_args, serializer):
    steps_per_epoch=1000
    total_steps = cmd_args.epochs*steps_per_epoch
    cmd_args.learning_rate = WithStrPolyDecay(
        1e-3,
        total_steps,
        end_learning_rate=1e-4
    )
    # cmd_args.learning_rate = 1e-3

    hp = HyperParams(
        seed=cmd_args.seed,
        steps_per_epoch=steps_per_epoch,
        ac_kwargs={
            "actor_hidden_sizes": (32, 32),
            "critic_hidden_sizes": (512, 512),
            "obs_normalizer": lunar_lander.LunarLander().observation_space.high
        },
        start_steps=1000,
        replay_size=int(1e6),
        gamma=0.99,
        polyak=0.5,
        # pi_lr=tf.optimizers.schedules.PolynomialDecay(1e-3, 50000, end_learning_rate=1e-5),
        # q_lr=tf.optimizers.schedules.PolynomialDecay(1e-3, 50000, end_learning_rate=1e-5),
        pi_lr=cmd_args.learning_rate,
        q_lr=cmd_args.learning_rate,
        batch_size=128,
        act_noise=0.1,
        max_ep_len=300,
        epochs=cmd_args.epochs,
        train_every=50,
        train_steps=50,
    )
    generated_params = train_utils.create_train_folder_and_params("lander-custom", hp, cmd_args, serializer)
    env_fn = lambda: lunar_lander.LunarLander()
    ddpg(env_fn, save_freq=4, **generated_params)

if __name__ == '__main__':
    serializer = lander_serializer()
    cmd_args = args_utils.parse_arguments(serializer)
    train(cmd_args, serializer)
