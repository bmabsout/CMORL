from pathlib import Path
from anchored_rl.utils import save_utils
import numpy as np

def test(actor, env, seed=123, render=True, num_steps=100):
    o,i = env.reset(seed=seed)
    high = env.action_space.high
    low = env.action_space.low
    os = []
    rs = []
    sumr = 0.0;
    for _ in range(num_steps):
        o, r, d, _, i, = env.step(actor(o)*(high - low)/2.0 + (high + low)/2.0)
        if d:
            break
        os.append(o)
        rs.append(r)
        if render:
            env.render()
        sumr+=r
    print("reward sum:", sumr)
    return np.array(os), np.array(rs)

def folder_to_results(env, render, num_tests, folder_path, steps=100,  **kwargs):
    import tensorflow as tf
    saved = tf.saved_model.load(Path(folder_path, "actor"))
    def actor(x):
        return saved(np.array([x], dtype=np.float32))[0]
    runs = np.array(list(map(lambda i: test(actor, env, seed=17+i,
                    render=render, num_steps=steps)[1], range(num_tests))))
    return runs

def run_tests(env, cmd_args):
    print(cmd_args.save_folders)
    if cmd_args.use_latest:
        folders = [save_utils.latest_train_folder(folder) for folder in cmd_args.save_folders]
    else:
        folders = save_utils.concatenate_lists([save_utils.find_all_train_paths(folder) for folder in cmd_args.save_folders])
    print("################################")
    print("################################")
    print("################################")
    for folder in folders:
        print("using folder:", folder)
    print("################################")
    print("################################")
    print("################################")
    runs = [np.mean(folder_to_results(env, folder_path=folder, **vars(cmd_args))) for folder in folders]
    return runs