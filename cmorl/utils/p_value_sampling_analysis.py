# Sample 20 different 4 p-values list combinations of the form 
# [p_loss_batch, p_loss_objectives, p_Q_batch, p_Q_objectives]
# where each value is randomly selected from the discrete range(-50, 50, 1).
# loop over 20 random p-values lists and sample from a normal distribution
# with mean 0 and standard deviation 1 over a discrete range of -50 to 50.

import numpy as np
import matplotlib.pyplot as plt

def sample_p_values(n_samples=50, mean=0, std=15, low=-50, high=50):
    p_values_list = []
    for i in range(n_samples):
        p_values = [np.random.normal(mean, std) for _ in range(4)]
        p_values = [int(p) for p in p_values]
        p_values = [max(low, min(high, p)) for p in p_values]
        p_values_list.append(p_values)
    return p_values_list

if __name__ == "__main__":
    p_values_list = sample_p_values(50)
    p_values_flattened = np.array(p_values_list).flatten()
    print(p_values_flattened)
    plt.hist(p_values_flattened, bins=50)
    plt.xlabel('p-values')
    plt.ylabel('Frequency')
    plt.title('Histogram of p-values')
    plt.show()


