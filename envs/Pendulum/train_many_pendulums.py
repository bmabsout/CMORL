from cmorl.rl_algs.ddpg.hyperparams import HyperParams, default_serializer
from itertools import product
from . import train_pendulum

def many_serializer():
    return default_serializer(hypers=HyperParams(epochs = 20, pi_lr=1e-2, q_lr=1e-2)).remove_names({
        "p_Q_batch",
        "p_Q_objectives",
    })

if __name__ == "__main__":
    # p_values_list = p_value_sampling_analysis.sample_p_values(
    #     n_samples=50, mean=0, std=15, low=-50, high=50
    # )
    # Make all possible combinations of p_values using the following list:
    serializer = many_serializer()
    cmd_args = serializer.parse_arguments()
    p_values_batch = [-32, -16, -8, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 8, 16, 32]
    p_values_objectives = [-32, -16, -8, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 8, 16, 32]
    p_values_list = list(product(p_values_batch, p_values_objectives))

    len(p_values_list)
    print("\n\nNumber of p_values combinations: ", len(p_values_list), "\n\n")

    for p_Q_batch, p_Q_objectives in p_values_list:
        cmd_args.p_Q_batch = p_Q_batch
        cmd_args.p_Q_objectives = p_Q_objectives
        train_pendulum.parse_args_and_train(namespace=cmd_args)
