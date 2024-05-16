import argparse
import gymnasium
import numpy as np
from cmorl.utils import test_utils
from configs import env_configs, get_config


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "save_folders", nargs="+", type=str, help="location of the training run"
    )
    parser.add_argument(
        "-r", "--render", action="store_true", help="render the env as it evaluates"
    )
    parser.add_argument("-env", "--env_name", type=str, default="HalfCheetah-v4")
    parser.add_argument("-n", "--num_tests", type=int, default=20)
    # group = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "-l",
        "--use_latest",
        action="store_true",
        help="use the latest training run from the save_folder",
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    cmd_args = parse_args()
    config = get_config(cmd_args.env_name)
    env = gymnasium.make(cmd_args.env_name, render_mode="human" if cmd_args.render else None, **config.hypers.env_args)
    runs = test_utils.run_tests(
        env,
        cmd_args,
        cmorl=config.cmorl,
    )
    print(f"{np.mean(runs):.4f}+-{np.std(runs):.4f}")
