from cmorl.rl_algs.ddpg.ddpg import ddpg, HyperParams
from ant import AntEnv
import ant


def parse_args_and_train(args=None):
    import cmorl.utils.train_utils as train_utils
    import cmorl.utils.args_utils as args_utils

    serializer = args_utils.default_serializer(epochs=1000, learning_rate=1e-4)
    cmd_args = args_utils.parse_arguments(serializer)
    hp = HyperParams(
        epochs=cmd_args.epochs,
        q_lr=cmd_args.learning_rate,
        pi_lr=cmd_args.learning_rate,
        seed=cmd_args.seed,
        max_ep_len=400,
        steps_per_epoch=1000,
    )
    generated_params = train_utils.create_train_folder_and_params("Ant-custom", hp, cmd_args, serializer)
    env_fn = lambda: AntEnv(
        reward_fn=ant.multi_dim_reward,
        # render_mode="human",
    )
    ddpg(
        env_fn,
        run_description="""In this run, we use DDPG with tanh for calculating the velocity reward
        and the combination of all joints for the actuation reward. The reward is a vector of 2 elements,
	where the first is the velocity reward and second is the actuation reward calculated from the geometric
	mean of all actuation rewards for all joints using p=0. The q-values are then composed all together in a single step
	with p_mean=-4.0 after reducing the q-values across the batch.""",
        **generated_params
    )


if __name__ == "__main__":
    parse_args_and_train()
