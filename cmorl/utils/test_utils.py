from cmorl.utils import save_utils
import numpy as np

from cmorl.utils.reward_utils import Transition


def test(actor, env, seed=123, render=True, num_steps=400, cmorl=None):
    o, _ = env.reset(seed=seed)
    high = env.action_space.high
    low = env.action_space.low
    os = []
    rs = []
    sumr = 0.0
    for _ in range(num_steps):
        action = actor(o) * (high - low) / 2.0 + (high + low) / 2.0
        o2, r, d, t, i = env.step(action)
        if cmorl:
            transition = Transition(o, action, o2, d, i)
            r = cmorl(transition, env)
        if d:
            break
        o = o2
        os.append(o)
        rs.append(r)
        if render:
            env.render()
        sumr += r
    print(sumr.shape)
    print("reward sum:", sumr)
    return np.array(os), np.array(rs)


def folder_to_results(env, render, num_tests, folder_path, steps=400, cmorl=None, **kwargs):
    saved = save_utils.load_actor(folder_path)

    def actor(x):
        return saved(np.array([x], dtype=np.float32))[0]

    runs = np.array(
        list(
            map(
                lambda i: test(actor, env, seed=17 + i, render=render, num_steps=steps, cmorl=cmorl)[1],
                range(num_tests),
            )
        )
    )
    return runs


def run_tests(env, cmd_args, folders = [], cmorl=None):
    print(cmd_args.save_folders)
    print("################################")
    print("################################")
    print("################################")
    for folder in folders:
        print("using folder:", folder)
    print("################################")
    print("################################")
    print("################################")
    runs = [
        np.mean(folder_to_results(env, folder_path=folder, cmorl=cmorl, **vars(cmd_args)))
        for folder in folders
    ]
    return runs
