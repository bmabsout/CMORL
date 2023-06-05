import argparse
import json
import os
import time
import pickle
from spinup.utils.args_utils import Arg_Serializer, Serialized_Argument
from spinup.utils import train_utils, save_utils, args_utils


reacher_serializer = Arg_Serializer(
    abbrev_to_args= {
        'd': Serialized_Argument(name='--distance', type=float, default=0.2, help='radius of points from the center'),
        'b': Serialized_Argument(name='--bias', type=float, default=0.0, help='bias of points from the center'),
    }
)


def train_reacher(cmd_args):
    import spinup.algos.ddpg.ddpg as rl_alg
    import reacher
    import tensorflow as tf

    hp = rl_alg.HyperParams(
        seed=cmd_args.seed,
        steps_per_epoch=1000,
        ac_kwargs={
            "actor_hidden_sizes": (32, 32),
            "critic_hidden_sizes": (256, 256),
            "obs_normalizer": [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 2.0, 2.0, 2.0]
        },
        pi_bar_variance=[0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        start_steps=1000,
        replay_size=int(1e5),
        gamma=0.9,
        polyak=0.995,
        # pi_lr=tf.optimizers.schedules.PolynomialDecay(1e-3, 50000, end_learning_rate=1e-5),
        # q_lr=tf.optimizers.schedules.PolynomialDecay(1e-3, 50000, end_learning_rate=1e-5),
        pi_lr=cmd_args.learning_rate,
        q_lr=cmd_args.learning_rate,
        batch_size=200,
        act_noise=0.05,
        max_ep_len=200,
        epochs=cmd_args.epochs,
        train_every=50,
        train_steps=30,
    )

    save_path = train_utils.save_hypers(hp, cmd_args, reacher_serializer)
    train_utils.build_training_params(hyperparams, args= None, reacher_serializer):


if __name__ == "__main__":
    train_reacher(args_utils.parse_arguments(None, reacher_serializer)