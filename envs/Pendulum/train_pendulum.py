from cmorl.rl_algs.ddpg.hyperparams import default_serializer, HyperParams
import cmorl.utils.train_utils as train_utils
from cmorl.utils.reward_utils import CMORL
from cmorl.rl_algs.ddpg.ddpg import ddpg
import Pendulum


def parse_args_and_train(**kwargs):
    hypers = HyperParams(
        ac_kwargs={"actor_hidden_sizes": (16, 16), "critic_hidden_sizes": (64, 64)},
        start_steps=500,
        max_ep_len=400,
        steps_per_epoch=1000,
        epochs=100,
        q_lr=1e-2,
        pi_lr=1e-2,
    )
    serializer = default_serializer(hypers=hypers)
    cmd_args = serializer.parse_arguments(**kwargs)

    generated_params = train_utils.create_train_folder_and_params(
        "Pendulum-custom", cmd_args, serializer
    )
    env_fn = lambda: Pendulum.PendulumEnv(
        g=10.0, setpoint=0.0
    )
    ddpg(
        env_fn,
        experiment_description="Testing the variance with respect to p-value when composing Q-values.",
        cmorl=CMORL(Pendulum.multi_dim_reward),
        **generated_params
    )


if __name__ == "__main__":
    parse_args_and_train()
