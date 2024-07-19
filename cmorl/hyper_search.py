from gymnasium.utils import seeding
from cmorl import train

def random_args_generator(n = 10):
    for seed in range(n):
        np_random ,_ = seeding.np_random(seed)
        yield [
            "-p_b", str(np_random.normal(0.0, 5)),
            "-p_o", str(np_random.normal(0.0, 5)),
            "-q_b", str(np_random.normal(0.0, 5)),
            "-q_o", str(np_random.normal(0.0, 5)),
            "--qd_power", str(np_random.uniform(0.0, 2.0)),
        ]


if __name__ == "__main__":
    cmd_args, rest_of_args = train.parse_env_name()
    for random_args in random_args_generator():
        train.parse_args_and_train(cmd_args.env_name, args=["-n", "hypersearch", "--seed", "1"]+random_args)