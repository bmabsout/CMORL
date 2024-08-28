from collections import deque
import glob
from cmorl.configs import ForcedTimeLimit
from cmorl.rl_algs.ddpg.ddpg import add_noise_to_weights
from cmorl.utils import save_utils
import numpy as np
import tensorflow as tf

from cmorl.utils.reward_utils import CMORL, Transition, discounted_window, estimated_value_fn, values


def test(actor, critic, env, seed=123, render=True, force_truncate_at=None, cmorl=None, max_ep_len=None, gamma=0.99):
    if force_truncate_at is not None:
        env = ForcedTimeLimit(env, max_episode_steps=force_truncate_at)
    o, _ = env.reset(seed=seed)
    np_random = np.random.default_rng(seed)
    os = deque()
    rs = deque()
    cmorl_rs = deque()
    actions = deque()
    while(True):
        action = actor(o, np_random)
        # print(action)
        actions.append(action)
        # print(o)
        os.append(o)
        o2, r, d, t, i = env.step(action)
        rs.append(r)
        if cmorl:
            transition = Transition(o, action, o2, d, i)
            cmorl_r = cmorl(transition, env)
            cmorl_rs.append(cmorl_r)
        if d or t or max_ep_len == len(os):
            print("done" if d else "truncated")
            break
        o = o2
        if render:
            env.render()
    actions = np.array(actions)
    rs = np.array(rs)
    os = np.array(os)
    cmorl_rs = np.array(cmorl_rs)
    qs = np.array(critic(os, actions))
    np.set_printoptions(precision=2)
    print("ep len:", len(os))
    rsum = np.sum(rs)
    print("reward sum:", rsum)
    print("cmorl sum:", np.sum(cmorl_rs, axis=0))
    estimated_value = estimated_value_fn(cmorl_rs, gamma, done=d)
    print(f"estimated avg value (gamma ${gamma}):", estimated_value)
    print("check:", np.mean(values(cmorl_rs, gamma, done=d), axis=0))
    print("mean q:", np.mean(qs, axis=0))
    print("first:", qs[0], np.sum(discounted_window(rs, gamma, done=d,axis=0)))
    print("last:", qs[-1])
    print("max:", np.max(qs, axis=0))
    print("min:", np.min(qs, axis=0))
    vals = values(cmorl_rs, gamma, done=d)
    print("offness:", np.mean(np.abs(qs - vals), axis=0))
    return os, rs, cmorl_rs, rsum, vals


def folder_to_results(env, render, num_tests, folder_path, force_truncate_at=None, cmorl=None, max_ep_len=None, **kwargs):
    saved_actor = save_utils.load_actor(folder_path)
    saved_critic = save_utils.load_critic(folder_path)

    def actor(x, np_random):
        return add_noise_to_weights(x, saved_actor, env.action_space, 0.0, np_random)
    def critic(o, a):
        return saved_critic(np.hstack([o, a], dtype=np.float32))
    runs = [
        test(actor, critic, env, seed=17 + i, render=render, force_truncate_at=force_truncate_at, cmorl=cmorl, max_ep_len=max_ep_len)
        for i in range(num_tests)
    ]
    return runs


def run_tests(env, cmd_args, folders, cmorl: CMORL=None, max_ep_len=None):
    # a deque so we can effiently append
    q_cs = deque()
    qs_cs = deque()
    rsums_means = deque()
    for folder in folders:
        print("using folder:", folder)
        _, _, _, rsums, valss = zip(*folder_to_results(env, folder_path=folder, cmorl=cmorl, max_ep_len=max_ep_len, **vars(cmd_args)))
        qs_c, q_c = cmorl.q_composer(np.concatenate(valss, axis=0))
        q_cs.append(q_c.numpy())
        qs_cs.append(qs_c.numpy())
        rsums_means.append(np.mean(rsums))

    results = {
        "q_c": (np.mean(q_cs, axis=0), np.std(q_cs, axis=0)),
        "qs_c": (np.mean(qs_cs, axis=0), np.std(qs_cs, axis=0)),
        "rsums": (np.mean(rsums_means), np.std(rsums_means))
    }

    return results

def folder_groups_from_globs(*globs: str):
    folder_groups = {}
    for unglobbed in globs:
        latest_folders = map(save_utils.latest_train_folder, glob.glob(unglobbed))
        folder_groups[unglobbed] = [ folder for folder in latest_folders if folder is not None]
    return folder_groups

def run_folder_group_tests(env, cmd_args, folder_groups, cmorl=None, max_ep_len=None):
    group_results = {}
    for folder_group_name, folders in folder_groups.items():
        print("using folder group:", folder_group_name)
        run_stats = run_tests(env, cmd_args, folders=folders, cmorl=cmorl, max_ep_len=max_ep_len)
            
        group_results[folder_group_name] = run_stats
    return group_results