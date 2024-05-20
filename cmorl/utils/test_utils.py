from cmorl.rl_algs.ddpg.ddpg import add_noise_to_weights
from cmorl.utils import save_utils
import numpy as np

from cmorl.utils.reward_utils import Transition, avg_rolling_discount


def test(actor, critic, env, seed=123, render=True, num_steps=400, cmorl=None):
    o, _ = env.reset(seed=seed)
    np_random = np.random.default_rng(seed)
    os = []
    rs = []
    cmorl_rs = []
    qs = []
    while(True):
        action = actor(o, np_random)
        qs.append(critic(o, action))
        os.append(o)
        o2, r, d, t, i = env.step(action)
        rs.append(r)
        if cmorl:
            transition = Transition(o, action, o2, d, i)
            cmorl_r = cmorl(transition, env)
            cmorl_rs.append(cmorl_r)
        if d or t:
            print("done")
            break
        o = o2
        if render:
            env.render()
    rs = np.array(rs)
    cmorl_rs = np.array(cmorl_rs)
    np.set_printoptions(precision=3)
    print("ep len:", len(os))
    print("reward sum:", np.sum(rs))
    print("cmorl sum:", np.sum(cmorl_rs, axis=0))
    print("dicounted sum (gamma 0.999):", avg_rolling_discount(cmorl_rs, 0.999))
    print("dicounted sum (gamma 0.99):", avg_rolling_discount(cmorl_rs, 0.99))
    print("dicounted sum (gamma 0.9):", avg_rolling_discount(cmorl_rs, 0.9))
    print("first:", qs[0])
    print("last:", qs[-1])
    print("max:", np.max(qs, axis=0))
    print("mean:", np.mean(qs, axis=0))
    print("min:", np.min(qs, axis=0))
    print("offness:", np.mean(np.abs(qs - avg_rolling_discount(cmorl_rs, 0.99)), axis=0))
    return np.array(os), rs, cmorl_rs


def folder_to_results(env, render, num_tests, folder_path, steps=400, cmorl=None, **kwargs):
    saved_actor = save_utils.load_actor(folder_path)
    saved_critic = save_utils.load_critic(folder_path)

    def actor(x, np_random):
        return add_noise_to_weights(x, saved_actor, env.action_space, 0.0, np_random)
    def critic(o, a):
        return saved_critic(np.array([np.concatenate([o, a], dtype=np.float32)]))[0]
    runs = np.array([
        test(actor, critic, env, seed=17 + i, render=render, num_steps=steps, cmorl=cmorl)
        for i in range(num_tests)
    ])
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
