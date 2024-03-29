import argparse
import json
import os
import time
import pickle
from cmorl.rl_algs.ddpg.ddpg import ddpg, HyperParams
from cmorl.utils import args_utils
from cmorl.utils import train_utils
from envs.Reacher import reacher

reacher_serializer = lambda: args_utils.Arg_Serializer.join(
    args_utils.Arg_Serializer(
        abbrev_to_args={
            "d": args_utils.Serialized_Argument(
                name="--distance",
                type=float,
                default=0.2,
                help="radius of points from the center",
            ),
            "b": args_utils.Serialized_Argument(
                name="--bias",
                type=float,
                default=0.0,
                help="bias of points from the center",
            ),
        }
    ),
    args_utils.default_serializer(),
)


def train(cmd_args, serializer):
    hp = HyperParams(
        seed=cmd_args.seed,
        steps_per_epoch=1000,
        ac_kwargs={
            "actor_hidden_sizes": (32, 32),
            "critic_hidden_sizes": (256, 256),
            "obs_normalizer": [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 2.0, 2.0, 2.0],
        },
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
    generated_params = train_utils.create_train_folder_and_params(
        "Reacher-custom", hp, cmd_args, serializer
    )
    env_fn = lambda: reacher.ReacherEnv(
        goal_distance=cmd_args.distance,
        bias=cmd_args.bias,
        reward_fn=reacher.multi_dim_reward,
    )
    ddpg(env_fn, **generated_params)


if __name__ == "__main__":
    serializer = reacher_serializer()
    cmd_args = args_utils.parse_arguments(serializer)
    cmd_args.epochs = 50
    train(cmd_args, serializer)
