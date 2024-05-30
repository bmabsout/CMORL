import argparse
import numpy as np
from cmorl.utils import save_utils, test_utils
from cmorl.configs import get_env_and_config


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "save_folders", nargs="*", type=str, default=["trained"], help="location of the training run"
    )
    parser.add_argument(
        "-r", "--render", action="store_true", help="render the env as it evaluates"
    )
    parser.add_argument("-env", "--env_name", type=str, default=None)
    parser.add_argument("-n", "--num_tests", type=int, default=20)
    parser.add_argument("-f", "--force_truncate_at", type=int, default=None)
    return parser.parse_args(args)


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    cmd_args = parse_args()
    
    folders = list(map(save_utils.latest_train_folder, cmd_args.save_folders))
    if cmd_args.env_name is None:
        cmd_args.env_name = folders[0].parents[5].name
        print(cmd_args.env_name)
    env_fn, config = get_env_and_config(cmd_args.env_name)
    env = env_fn(render_mode="human" if cmd_args.render else None)
    runs = test_utils.run_tests(
        env,
        cmd_args,
        folders=folders,
        max_ep_len=config.hypers.max_ep_len,
        cmorl=config.cmorl,
    )
    print(f"{np.mean(runs):.4f}+-{np.std(runs):.4f}")
