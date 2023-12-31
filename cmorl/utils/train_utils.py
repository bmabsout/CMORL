from functools import partial
from cmorl.utils import save_utils
from cmorl.utils.args_utils import Arg_Serializer



def create_train_folder_and_params(experiment_name, hyperparams, cmd_args, serializer: Arg_Serializer):
    """
    Sets up the folders for the experiment and trains the agent.
    """
    # Create the folders
    save_path = save_utils.save_hypers(experiment_name, hyperparams, cmd_args, serializer)
    generated_params = {
        "hp": hyperparams,
        "on_save": partial(save_utils.on_save, replay_save=cmd_args.replay_save, save_path=save_path),
        "logger_kwargs": {"output_dir": save_path},
        "extra_hyperparameters": vars(cmd_args)
    }
    if cmd_args.prev_folder:
        def existing_actor_critic(*args, **kwargs):
            return save_utils.load_actor(cmd_args.prev_folder), save_utils.load_critic(cmd_args.prev_folder)
        generated_params["actor_critic"] = existing_actor_critic

    return generated_params