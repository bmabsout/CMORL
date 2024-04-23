from cmorl.rl_algs.ddpg.ddpg import ddpg, HyperParams
from cmorl.utils import args_utils
from Boids import BoidsEnv
import Boids


def parse_args_and_train(args=None):
    import cmorl.utils.train_utils as train_utils
    import cmorl.utils.args_utils as args_utils

    serializer = args_utils.default_serializer(epochs=50, learning_rate=1e-3)
    cmd_args = args_utils.parse_arguments(serializer)
    hp = HyperParams(
        ac_kwargs={"actor_hidden_sizes": (256, 256), "critic_hidden_sizes": (512, 512)},
        epochs=cmd_args.epochs,
        q_lr=cmd_args.learning_rate,
        pi_lr=cmd_args.learning_rate,
        seed=cmd_args.seed,
        max_ep_len=400,
        gamma=0.99,
        steps_per_epoch=1000,
    )
    generated_params = train_utils.create_train_folder_and_params(
        "Boids-custom", hp, cmd_args, serializer
    )
    env_fn = lambda: BoidsEnv(
        reward_fn=Boids.multi_dim_reward
    )
    ddpg(
        env_fn,
        **generated_params
    )


if __name__ == "__main__":
    parse_args_and_train()
