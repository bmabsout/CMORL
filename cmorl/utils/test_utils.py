from cmorl.configs import ForcedTimeLimit
from cmorl.rl_algs.ddpg.ddpg import add_noise_to_weights
from cmorl.utils import save_utils
import numpy as np

from cmorl.utils.reward_utils import Transition, discounted_window, estimated_value_fn, values


def test(actor, critic, env, seed=123, render=True, force_truncate_at=None, cmorl=None, max_ep_len=None):
    if force_truncate_at is not None:
        env = ForcedTimeLimit(env, max_episode_steps=force_truncate_at)
    o, _ = env.reset(seed=seed)
    np_random = np.random.default_rng(seed)
    os = []
    rs = []
    cmorl_rs = []
    actions = []
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
    print("reward sum:", np.sum(rs))
    print("cmorl sum:", np.sum(cmorl_rs, axis=0))
    print("estimated avg value (gamma 0.999):", estimated_value_fn(cmorl_rs, 0.999, done=d))
    print("estimated avg value (gamma 0.99):", estimated_value_fn(cmorl_rs, 0.99, done=d))
    print("check:", np.mean(values(cmorl_rs, 0.99, done=d), axis=0))
    print("estimated avg value (gamma 0.9):", estimated_value_fn(cmorl_rs, 0.9, done=d))
    print("mean q:", np.mean(qs, axis=0))
    print("first:", qs[0], np.sum(discounted_window(rs, 0.99, done=d,axis=0)))
    print("last:", qs[-1])
    print("max:", np.max(qs, axis=0))
    print("min:", np.min(qs, axis=0))
    print(discounted_window(cmorl_rs, 0.99, window_size=np.inf, done=d)[0])
    print(qs[0])
    print("offness:", np.mean(np.abs(qs - values(cmorl_rs, 0.99, done=d)), axis=0))
    return os, rs, cmorl_rs


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


def run_tests(env, cmd_args, folders = [], cmorl=None, max_ep_len=None):
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
        np.mean(folder_to_results(env, folder_path=folder, cmorl=cmorl, max_ep_len=max_ep_len, **vars(cmd_args)))
        for folder in folders
    ]
    return runs
