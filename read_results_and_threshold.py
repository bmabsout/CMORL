from collections import defaultdict
import pickle
from pathlib import Path
import numpy as np
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Read results and calculate steps before threshold.')

# Add a command line argument for the threshold
parser.add_argument('-t', '--threshold', type=float, help='Threshold value')

# Add a command line argument for the results file
parser.add_argument('-r', '--results-file', type=str, help='Path to the results file')

# Parse the command line arguments
args = parser.parse_args()

results = pickle.load(open(args.results_file, "rb"))

# keys of results look like this : trained/Hopper-v4/default_seeds/x:LP5CB4DX2EBXWZF/seeds/{seed}/epochs/{epoch}/ we want to extract the seed and epoch
seed_to_epoch_to_results = defaultdict(dict)
for key, value in results.items():
    seed = int(Path(key).parts[-3])
    epoch = int(Path(key).parts[-1])
    seed_to_epoch_to_results[seed][epoch] = value
    print(f"seed: {seed}, epoch: {epoch}, value: {value}")
print(seed_to_epoch_to_results)
# turn seed_to_epoch_to_results into a dict of lists where the lists are of the same length

seed_to_results_list = defaultdict(list)
for seed, epoch_to_results in seed_to_epoch_to_results.items():
    for epoch, value in sorted(epoch_to_results.items()):
        seed_to_results_list[seed].append(value)

def results_list_to_steps(rl):
    rsums = np.array([float(result["rsums"][0]) for result in rl])
    print(rsums)
    return (np.argmax(rsums >= args.threshold) + 1) * 2000

seed_to_steps_before_threshold = {seed: results_list_to_steps(rl) for seed, rl in seed_to_results_list.items()}

for seed, steps_before_threshold in seed_to_steps_before_threshold.items():
    print("seed:", seed, "steps:", steps_before_threshold)

print("mean steps:", np.mean(list(seed_to_steps_before_threshold.values())))
