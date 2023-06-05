import argparse
import numpy as np
import pathlib
import os

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('save_folders', nargs='+', type=str, help='location of the training run')
    parser.add_argument('-r', '--render', action="store_true", help="render the env as it evaluates")
    parser.add_argument('-d', '--distance', type=float, default=0.2, help='radius of points from the center')
    parser.add_argument('-b', '--bias', type=float, default=0.0, help='bias of points from the center')
    parser.add_argument('-n', '--num_tests', type=int, default=20)
    # group = parser.add_mutually_exclusive_group()
    parser.add_argument('-l', '--use_latest', action="store_true", help="use the latest training run from the save_folder")
    return parser.parse_args(args)


def test(actor, env, seed=123, render=True, num_steps=100):
    o,i = env.reset(seed=seed)
    # print(o)
    # actor(o)
    # print("no")
    high = env.action_space.high
    low = env.action_space.low
    os = []
    rs = []
    sumr = 0.0;
    for _ in range(num_steps):
        o, r, d, _, i, = env.step(actor(o)*(high - low)/2.0 + (high + low)/2.0)
        # print(o)
        os.append(o)
        rs.append(r)
        if render:
            env.render()
        sumr+=r
    print("reward sum:", sumr)
    return np.array(os), np.array(rs)


def latest_subdir(path='.'):
    return max(pathlib.Path(path).glob('*/'), key=os.path.getctime)

# def latest_train_folder(path):
#     latest_seed_path = latest_subdir(f"{path}/seeds")
#     latest_epoch_path = latest_subdir(f"{latest_seed_path}/epochs")
#     return latest_epoch_path

# def all_seed_folders()

def get_last_epoch_path_for_each_seed_folder(path):
    return [latest_subdir(str(d)) for d in pathlib.Path(path).glob('seeds/*/epochs/')]


def find_folders(dirname, name_to_find) -> list[str]:
    subfolders = [f.path for f in os.scandir(
        dirname) if f.is_dir()]
    subfolders_with_the_right_name = [ subfolder for subfolder in subfolders if pathlib.Path(subfolder).name == name_to_find]
    for dirname in list(subfolders):
        subfolders_with_the_right_name.extend(find_folders(dirname, name_to_find))
    return subfolders_with_the_right_name


def find_all_train_paths(path):
    return [pathlib.Path(folder).parent for folder in find_folders(path, "actor")]


def latest_train_folder(path):
    return max(find_all_train_paths(path), key=os.path.getctime)


def concatenate_lists(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def folder_to_results(render, distance, bias, num_tests, folder_path, **kwargs):
    import tensorflow as tf
    import reacher
    saved = tf.saved_model.load(f"{folder_path}/actor")
    def actor(x): return saved(np.array([x], dtype=np.float32))[0]
    env = reacher.ReacherEnv(
        render_mode="human" if render else None,
        goal_distance=distance,
        bias=bias
    )
    runs = np.array(list(map(lambda i: test(actor, env, seed=17+i,
                    render=False)[1], range(num_tests))))
    return runs


def run_tests(cmd_args):
    print(cmd_args.save_folders)
    if cmd_args.use_latest:
        folders = [latest_train_folder(folder) for folder in cmd_args.save_folders]
    else:
        folders = concatenate_lists([find_all_train_paths(folder) for folder in cmd_args.save_folders])
    print("################################")
    print("################################")
    print("################################")
    for folder in folders:
        print("using folder:", folder)
    print("################################")
    print("################################")
    print("################################")
    runs = [np.mean(folder_to_results(folder_path=folder, **vars(cmd_args))) for folder in folders]
    return runs

if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    runs = run_tests(parse_args())
    print(f"{np.mean(runs):.4f}+-{np.std(runs):.4f}")
