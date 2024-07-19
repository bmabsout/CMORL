import argparse
from cmorl.rl_algs.ddpg.ddpg import ddpg
from cmorl.rl_algs.ddpg import hyperparams
from cmorl.configs import get_env_and_config
import cmorl.utils.train_utils as train_utils
import envs # for the gym registrations

def parse_env_name(args=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "env_name", type=str, help="environment name (used in gym.make)"
    )
    return parser.parse_known_args(args)

def parse_args_and_train(env_name, args=None):
    env_fn, config = get_env_and_config(env_name)
    serializer = hyperparams.default_serializer(hypers=config.hypers)
    cmd_args = serializer.parse_arguments(args)
    generated_params = train_utils.create_train_folder_and_params(
        env_name, cmd_args, serializer
    )
    ddpg(
        env_fn,
        cmorl=config.cmorl,
        **generated_params
    )

if __name__ == "__main__":
    cmd, rest_of_args = parse_env_name()
    parse_args_and_train(cmd.env_name, rest_of_args)