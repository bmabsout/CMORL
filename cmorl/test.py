import argparse
import numpy as np
from cmorl.utils import save_utils, test_utils
from cmorl.configs import get_env_and_config
import glob
import envs # for the gym registrations

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "save_folders", nargs="*", type=str, default=["trained"], help="location of the training run, can be a glob, can be multiple, each glob is considered a group of files to test"
    )
    parser.add_argument(
        "-r", "--render", action="store_true", help="render the env as it evaluates"
    )
    parser.add_argument("-env", "--env_name", type=str, default=None)
    parser.add_argument("-n", "--num_tests", type=int, default=6)
    parser.add_argument("-f", "--force_truncate_at", type=int, default=None)
    parser.add_argument("-a", "--act_noise", type=float, default=0.0)
    return parser.parse_args(args)


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    cmd_args = parse_args()
    # expect some strings in save_folders that have * in them and glob them:
    folder_groups = test_utils.folder_groups_from_globs(*cmd_args.save_folders)
    print(folder_groups)
    if cmd_args.env_name is None:
        # assume the first folder is the same env as the rest
        group, folders_in_first_group = list(folder_groups.items())[0]
        cmd_args.env_name = save_utils.get_env_name_from_folder(folders_in_first_group[0])
    env_fn, config = get_env_and_config(cmd_args.env_name)
    env = env_fn(render_mode="human" if cmd_args.render else None)
    run_groups = test_utils.run_folder_group_tests(
        env,
        cmd_args,
        folder_groups,
        max_ep_len=config.hypers.max_ep_len,
        cmorl=config.cmorl,
    )
    print("results:")
    for group_name, results in run_groups.items():
        print(f"Group: {group_name}")
        # pretty print results
        for key, (mean, std) in results.items():
            print(f"\t{key}: {mean} +- {std}")
