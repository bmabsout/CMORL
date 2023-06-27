from anchored_rl.rl_algs.ddpg.ddpg import ddpg, HyperParams
from anchored_rl.utils import args_utils
import Pendulum

def parse_args_and_train(args=None):
    import anchored_rl.utils.train_utils as train_utils
    import anchored_rl.utils.args_utils as args_utils
    serializer = args_utils.default_serializer()
    serializer.abbrev_to_args["e"] = args_utils.Serialized_Argument(name='--epochs', type=int, default=10, help='number of epochs'),
    cmd_args = args_utils.parse_arguments(serializer)
    hp = HyperParams(q_lr=cmd_args.learning_rate, pi_lr=cmd_args.learning_rate, seed=cmd_args.seed, max_ep_len=200)
    generated_params = train_utils.create_train_folder_and_params("Pendulum-custom", hp, cmd_args, serializer)
    env_fn = lambda: Pendulum.PendulumEnv(g=10.0, setpoint=0.0)
    ddpg(env_fn, **generated_params)

if __name__ == '__main__':
    parse_args_and_train()
