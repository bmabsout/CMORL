from cmorl.rl_algs.ddpg.ddpg import ddpg, HyperParams
from cmorl.utils import args_utils
from cmorl.utils import train_utils
from cmorl.utils.reward_utils import CMORL
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
    args_utils.default_serializer(epochs=50),
)


def train(cmd_args, serializer):
    hp = HyperParams.from_cmd_args(cmd_args)
    hp.steps_per_epoch=1000
    hp.ac_kwargs={
            "actor_hidden_sizes": (32, 32),
            "critic_hidden_sizes": (256, 256),
            "obs_normalizer": [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 2.0, 2.0, 2.0],
        }
    hp.start_steps=1000
    hp.replay_size=int(1e5)
    hp.gamma=0.9
    hp.max_ep_len=400

    generated_params = train_utils.create_train_folder_and_params(
        "Reacher-custom", hp, cmd_args, serializer
    )
    env_fn = lambda: reacher.ReacherEnv(
        goal_distance=cmd_args.distance,
        bias=cmd_args.bias,
    )
    ddpg(env_fn, cmorl=CMORL(reacher.multi_dim_reward), **generated_params)


if __name__ == "__main__":
    serializer = reacher_serializer()
    cmd_args = args_utils.parse_arguments(serializer)
    train(cmd_args, serializer)
