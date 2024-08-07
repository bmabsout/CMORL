import numpy as np
import Pendulum
import argparse
from cmorl.utils import test_utils, reward_utils

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "save_folders", nargs="+", type=str, help="location of the training run"
    )
    parser.add_argument(
        "-r", "--render", action="store_true", help="render the env as it evaluates"
    )
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
    cmd_args.steps = 100
    runs = test_utils.run_tests(
        Pendulum.PendulumEnv(render_mode="human" if cmd_args.render else None), cmd_args, cmorl=reward_utils.CMORL(Pendulum.multi_dim_reward)
    )
    print(f"{np.mean(runs):.4f}+-{np.std(runs):.4f}")
