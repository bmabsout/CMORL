import multiprocessing as mp
from gymnasium.utils import seeding
from cmorl import train

def random_args_generator(n=10):
    for seed in range(n):
        np_random, _ = seeding.np_random(seed)
        yield [
            "-p_b", str(np_random.normal(0.0, 5)),
            "-p_o", str(np_random.normal(0.0, 5)),
            "-q_b", str(np_random.normal(0.0, 5)),
            "-q_o", str(np_random.normal(0.0, 5)),
            "-q_d", str(np_random.uniform(0.0, 2.0)),
            "--act_noise", str(np_random.uniform(0.01, 0.2)),
        ]

def run_training(env_name, args):
    try:
        train.parse_args_and_train(env_name, args=args)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    cmd_args, rest_of_args = train.parse_env_name()
    
    # Prepare arguments for each process
    process_args = [
        (cmd_args.env_name, ["-n", "hypersearch", "--seed", "1"] + random_args + rest_of_args)
        for random_args in random_args_generator()
    ]
    
    # Create a pool of worker processes
    with mp.Pool(processes=6) as pool:
        # Map the run_training function to the process arguments
        pool.starmap(run_training, process_args)