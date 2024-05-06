from cmorl.rl_algs.ddpg.ddpg import ddpg, HyperParams
from cmorl.utils import args_utils
from Pendulum import PendulumEnv
import Pendulum


def parse_args_and_train(args=None):
    import cmorl.utils.train_utils as train_utils
    import cmorl.utils.args_utils as args_utils

    serializer = args_utils.default_serializer(epochs=20, learning_rate=1e-2)
    cmd_args = args_utils.parse_arguments(serializer)
    hp = HyperParams(
        ac_kwargs={"actor_hidden_sizes": (16, 16), "critic_hidden_sizes": (64, 64)},
        start_steps=500,
        epochs=cmd_args.epochs,
        q_lr=cmd_args.learning_rate,
        pi_lr=cmd_args.learning_rate,
        p_Q_batch=cmd_args.p_Q_batch,
        p_Q_objectives=cmd_args.p_Q_objectives,
        seed=cmd_args.seed,
        max_ep_len=200,
        steps_per_epoch=1000,
    )
    generated_params = train_utils.create_train_folder_and_params(
        "Pendulum-custom", hp, cmd_args, serializer
    )
    env_fn = lambda: PendulumEnv(
        g=10.0, setpoint=0.0, reward_fn=Pendulum.multi_dim_reward
    )
    ddpg(
        env_fn,
        experiment_description="Testing the variance with respect to p-value when composing Q-values.",
        **generated_params
    )


if __name__ == "__main__":
    parse_args_and_train()
