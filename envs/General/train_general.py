import argparse
import gymnasium
import gymnasium.envs.mujoco
import gymnasium.envs.mujoco.ant_v4
from cmorl.rl_algs.ddpg.ddpg import ddpg
from cmorl.rl_algs.ddpg.hyperparams import default_serializer
from cmorl.utils import args_utils
from envs.General.configs import get_config


def parse_args_and_train(args=None):
    import cmorl.utils.train_utils as train_utils
    env_name_serializer = args_utils.Arg_Serializer(
        args_utils.Serialized_Argument(
            name="env_name",
            abbrev="env",
            type=str,
            default="HalfCheetah-v4",
        ),
    )
    env_name_parser = argparse.ArgumentParser()
    env_name_serializer.add_serialized_args_to_parser(env_name_parser)
    env_name_cmd, rest_of_args = env_name_parser.parse_known_args(args)
    env_name = env_name_cmd.env_name
    config = get_config(env_name)
    serializer = default_serializer(hypers=config.hypers)
    cmd_args = serializer.parse_arguments(rest_of_args)
    generated_params = train_utils.create_train_folder_and_params(
        env_name, cmd_args, serializer
    )
    env_fn = lambda: gymnasium.make(env_name, **cmd_args.env_args) 
    ddpg(
        env_fn,
        cmorl=config.cmorl,
        **generated_params
    )

if __name__ == "__main__":
    parse_args_and_train()