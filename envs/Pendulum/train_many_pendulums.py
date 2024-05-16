from cmorl.utils import args_utils
from itertools import product
import train_pendulum

def many_serializer(epochs=20, learning_rate=1e-2):
    return args_utils.default_serializer(epochs=epochs, learning_rate=learning_rate).remove_args({
        "p_Q_batch",
        "p_Q_objectives",
    })

if __name__ == "__main__":
    # p_values_list = p_value_sampling_analysis.sample_p_values(
    #     n_samples=50, mean=0, std=15, low=-50, high=50
    # )
    # Make all possible combinations of p_values using the following list:
    cmd_args = args_utils.parse_arguments(many_serializer())
    p_values_batch = [-32, -16, -8, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 8, 16, 32]
    p_values_objectives = [-32, -16, -8, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 8, 16, 32]
    p_values_list = list(product(p_values_batch, p_values_objectives))

    len(p_values_list)
    print("\n\nNumber of p_values combinations: ", len(p_values_list), "\n\n")

    for p_Q_batch, p_Q_objectives in p_values_list:
        cmd_args.p_Q_batch = p_Q_batch
        cmd_args.p_Q_objectives = p_Q_objectives
        train_pendulum.parse_args_and_train(namespace=cmd_args)
