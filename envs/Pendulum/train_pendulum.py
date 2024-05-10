import cmorl.utils.train_utils as train_utils
from cmorl.utils.reward_utils import CMORL
from cmorl.rl_algs.ddpg.ddpg import ddpg, HyperParams
import Pendulum
import cmorl.utils.args_utils as args_utils


def parse_args_and_train(**kwargs):

    serializer = args_utils.default_serializer(epochs=20, learning_rate=1e-2)
    cmd_args = args_utils.parse_arguments(serializer, **kwargs)
    hp = HyperParams.from_cmd_args(cmd_args)
    hp.ac_kwargs={"actor_hidden_sizes": (16, 16), "critic_hidden_sizes": (64, 64)}
    hp.start_steps=500
    hp.max_ep_len=400
    hp.steps_per_epoch=1000
    
    generated_params = train_utils.create_train_folder_and_params(
        "Pendulum-custom", hp, cmd_args, serializer
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
