from cmorl.rl_algs.ddpg.ddpg import ddpg, HyperParams
from cmorl.utils import args_utils, p_value_sampling_analysis
from Pendulum import PendulumEnv
import Pendulum


def parse_args_and_train(args=None, p_values_list=[0, -4, 0, -4]):
    import cmorl.utils.train_utils as train_utils
    import cmorl.utils.args_utils as args_utils

    serializer = args_utils.default_serializer(epochs=10, learning_rate=1e-2)
    cmd_args = args_utils.parse_arguments(serializer)
    hp = HyperParams(
        start_steps=500,
        epochs=cmd_args.epochs,
        q_lr=cmd_args.learning_rate,
        pi_lr=cmd_args.learning_rate,
        seed=cmd_args.seed,
        max_ep_len=200,
        steps_per_epoch=1000,
        p_Q_batch=p_values_list[0],
        p_Q_objectives=p_values_list[1],
    )
    generated_params = train_utils.create_train_folder_and_params(
        "Pendulum-custom", hp, cmd_args, serializer
    )
    env_fn = lambda: PendulumEnv(
        g=10.0, setpoint=0.0, reward_fn=Pendulum.multi_dim_reward
    )
    ddpg(
        env_fn,
        experiment_description="Testing the variance with resect to p-value when composing Q-values.",
        **generated_params,
    )


if __name__ == "__main__":
    # p_values_list = p_value_sampling_analysis.sample_p_values(
    #     n_samples=50, mean=0, std=15, low=-50, high=50
    # )
    # Make all possible combinations of p_values using the following list:
    p_values_batch = [-32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32]
    p_values_objectives = [-32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32]
    p_values_list = []
    for p_Q_batch in p_values_batch:
        for p_Q_objectives in p_values_objectives:
            p_values_list.append([p_Q_batch, p_Q_objectives])
    len(p_values_list)
    print("\n\nNumber of p_values combinations: ", len(p_values_list), "\n\n")

    for p_values in p_values_list:
        parse_args_and_train(p_values_list=p_values)
