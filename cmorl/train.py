import argparse
import gymnasium
import gymnasium.envs.mujoco
import gymnasium.envs.mujoco.ant_v4
from cmorl.rl_algs.ddpg.ddpg import ddpg
from cmorl.rl_algs.ddpg.hyperparams import default_serializer
from cmorl.configs import get_env_and_config


def parse_env_name(args=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "env_name", type=str, help="environment name (used in gym.make)"
    )
    return parser.parse_known_args(args)

def parse_args_and_train(args=None):
    import cmorl.utils.train_utils as train_utils
    env_name_cmd, rest_of_args = parse_env_name()
    env_name = env_name_cmd.env_name
    env_fn, config = get_env_and_config(env_name)
    serializer = default_serializer(hypers=config.hypers)
    cmd_args = serializer.parse_arguments(rest_of_args)
    generated_params = train_utils.create_train_folder_and_params(
        env_name, cmd_args, serializer
    )
    ddpg(
        env_fn,
        cmorl=config.cmorl,
        **generated_params
    )

if __name__ == "__main__":
    parse_args_and_train()