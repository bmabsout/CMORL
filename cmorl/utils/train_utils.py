from functools import partial
from cmorl.utils import save_utils
from cmorl.utils.args_utils import Arg_Serializer


def create_train_folder_and_params(env_name, cmd_args, serializer: Arg_Serializer):
    """
    Sets up the folders for the experiment and trains the agent.
    """

    # Create the folders
    save_path, semantic_name = save_utils.save_hypers(f"{env_name}/{cmd_args.experiment_name}", cmd_args, serializer)
    generated_params = {
        "env_name": env_name,
        "experiment_name": f"{cmd_args.experiment_name}({semantic_name})",
        "hp": cmd_args,
        "on_save": partial(save_utils.on_save, save_path=save_path),
        "logger_kwargs": {"output_dir": save_path},
    }
    if cmd_args.prev_folder:
        def existing_actor_critic(*args, **kwargs):
            return save_utils.load_actor(cmd_args.prev_folder), save_utils.load_critic(cmd_args.prev_folder)
        generated_params["actor_critic"] = existing_actor_critic

    return generated_params