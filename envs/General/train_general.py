import gymnasium
from cmorl.rl_algs.ddpg.ddpg import ddpg, HyperParams
from cmorl.utils import args_utils
from cmorl.utils.reward_utils import CMORL
import reward_fns

forward_serializer = args_utils.Arg_Serializer.join(
    args_utils.Arg_Serializer(
        abbrev_to_args={
            "env": args_utils.Serialized_Argument(
                name="--env_name",
                type=str,
                default="HalfCheetah-v4",
            ),
        }
    ),
    args_utils.default_serializer(epochs=100, learning_rate=1e-2),
)

def parse_args_and_train(args=None):
    import cmorl.utils.train_utils as train_utils
    import cmorl.utils.args_utils as args_utils

    cmd_args = args_utils.parse_arguments(forward_serializer)
    hp = HyperParams.from_cmd_args(cmd_args)
    hp.ac_kwargs = { # actor-critic kwargs
        "actor_hidden_sizes": (32, 32),
        "critic_hidden_sizes": (256, 256),
    }
    hp.max_ep_len=400
    # hp.start_steps=10000
    # hp.start_steps=1000
    # hp.replay_size=int(1e5)
    generated_params = train_utils.create_train_folder_and_params(
        cmd_args.env_name, hp, cmd_args, forward_serializer
    )
    env_fn = lambda: gymnasium.make(cmd_args.env_name)
    ddpg(
        env_fn,
        cmorl=CMORL(reward_fns.reward_fns[cmd_args.env_name]),
        **generated_params
    )


if __name__ == "__main__":
    parse_args_and_train()