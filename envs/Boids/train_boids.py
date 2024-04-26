import cmorl.utils.args_utils as args_utils

boids_serializer = lambda: args_utils.Arg_Serializer.join(
    args_utils.Arg_Serializer(
        abbrev_to_args={
            "numboids": args_utils.Serialized_Argument(
                name="--numboids",
                type=int,
                default=5,
                help="number of boids",
            ),
        }
    ),
    args_utils.default_serializer(epochs=200),
)

def parse_args_and_train(args=None):
    serializer = boids_serializer()
    cmd_args = args_utils.parse_arguments(serializer)

    import cmorl.utils.train_utils as train_utils
    from cmorl.rl_algs.ddpg.ddpg import ddpg, HyperParams
    import Boids
    # import tensorflow as tf
    # tf.debugging.experimental.enable_dump_debug_info(
    #     "./chu7",
    #     tensor_debug_mode="FULL_HEALTH",
    #     circular_buffer_size=-1
    # )

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
    env_fn = lambda: Boids.BoidsEnv(
        reward_fn=Boids.multi_dim_reward,
        numBoids=cmd_args.numboids,
    )
    ddpg(
        env_fn,
        **generated_params
    )

if __name__ == "__main__":
    parse_args_and_train()
    
