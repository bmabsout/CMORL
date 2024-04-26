from cmorl.rl_algs.ddpg.ddpg import ddpg, HyperParams
from cmorl.utils import args_utils
from Pendulum import PendulumEnv
import Pendulum


def parse_args_and_train(args=None):
    import cmorl.utils.train_utils as train_utils
    import cmorl.utils.args_utils as args_utils

    serializer = args_utils.default_serializer(epochs=10, learning_rate=1e-4)
    cmd_args = args_utils.parse_arguments(serializer)
    hp = HyperParams(
        epochs=cmd_args.epochs,
        q_lr=cmd_args.learning_rate,
        pi_lr=cmd_args.learning_rate,
        seed=cmd_args.seed,
        max_ep_len=200,
        steps_per_epoch=200,
    )
    generated_params = train_utils.create_train_folder_and_params(
        "Pendulum_custom", hp, cmd_args, serializer
    )
    env_fn = lambda: PendulumEnv(
        g=10.0, setpoint=0.0, reward_fn=Pendulum.multi_dim_reward
    )
    ddpg(
        env_fn,
        run_description="Testing the variance with resect to p-value when composing Q-values.",
        **generated_params
    )


if __name__ == "__main__":
    parse_args_and_train()
