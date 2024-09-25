import argparse
import multiprocessing as mp
from gymnasium.utils import seeding
from cmorl import train

def parse_hypersearch_args(args=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "env_name", type=str, help="environment name (used in gym.make)"
    )
    parser.add_argument(
        "--hyperseed", type=int, default=1, help="seed for random number generator"
    )
    parser.add_argument(
        "--num_searches", type=int, default=10, help="number of random searches to perform"
    )

    parser.add_argument(
        "--num_seeds", type=int, default=1, help="number of seeds to search per random hyperparam selection"
    )

    parser.add_argument(
        "--experiment_name", "-n", type=str, default="hypersearch", help="experiment name"
    )
    return parser.parse_known_args(args)

def random_args_generator(hyperseed, num_searches, num_seeds):
    for hyper_search_seed in range(num_searches):
        np_random, _ = seeding.np_random(hyperseed)
        np_random, _ = seeding.np_random(int(np_random.integers(1e10) + hyper_search_seed))
        random_hypers = [
            # "-p_b", str(np_random.normal(0.0, 1)),
            # "-p_o", str(np_random.normal(0.0, 1)),
            # "-q_b", str(np_random.normal(0.0, 1)),
            # "-q_o", str(np_random.normal(0.0, 1)),
            # "-q_d", str(np_random.uniform(0.2, 5.0)),
            # "--act_noise", str(2**np_random.uniform(-5, -0.5)),
            # "--batch_size", str(int(np_random.uniform(30, 500))),
            # "--gamma", str(1.0 - 10**np_random.uniform(-3, -1)),
            # "--replay_size", str(int(10**np_random.uniform(4, 6))),
            # "--polyak", str(1.0 - 10**np_random.uniform(-3, -1)),
        ]
        for seed in range(num_seeds):
            yield (random_hypers + ["--seed", str(seed)])

def run_training(env_name, args):
    try:
        train.parse_args_and_train(env_name, args=args)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    cmd_args, rest_of_args = parse_hypersearch_args()
    
    # Prepare arguments for each process
    process_args = [
        (cmd_args.env_name, ["-n", cmd_args.experiment_name] + random_args + rest_of_args)
        for random_args in random_args_generator(cmd_args.hyperseed, cmd_args.num_searches, cmd_args.num_seeds)
    ]
    
    # Create a pool of worker processes
    with mp.Pool(processes=10) as pool:
        # Map the run_training function to the process arguments
        pool.starmap(run_training, process_args)