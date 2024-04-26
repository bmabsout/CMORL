import argparse
import json
import os
import time
from keras import models, Model, config
import pickle
from cmorl.utils.args_utils import Arg_Serializer
from pathlib import Path
from cmorl.utils.serialization_utils import ExtraTypesEncoder
config.enable_unsafe_deserialization()


def save_hypers(experiment_name, hypers, cmd_args, serializer:Arg_Serializer):
    """ Saves the hyperparameters to a json file in the experiment folder. Uses semantic naming for the folder."""
    all_hypers = {**vars(cmd_args), **vars(hypers)}

    save_path = Path(serializer.get_seed_folder_path(experiment_name, all_hypers), "epochs")
    semantic_name = serializer.get_semantic_folder_name(all_hypers)
    common_output_path = Path("trained", experiment_name, semantic_name)
    os.makedirs(common_output_path, exist_ok=True)
    with open(f"{common_output_path}/hypers.json", "w") as f:
        json.dump(serializer.remove_ignored(all_hypers), f, indent=4, cls=ExtraTypesEncoder)
    return save_path, semantic_name


def latest_subdir(path='.'):
    return max(Path(path).glob('*/'), key=os.path.getctime)


def get_last_epoch_path_for_each_seed_folder(path):
    return [latest_subdir(str(d)) for d in Path(path).glob('seeds/*/epochs/')]


def find_files(dirname, name_to_find) -> list[str]:
    files_with_the_right_name = []
    for dir_entry in os.scandir(dirname):
        if dir_entry.is_dir():
            files_with_the_right_name.extend(find_files(dir_entry.path, name_to_find))
        elif Path(dir_entry.path).name == name_to_find:
            files_with_the_right_name.append(Path(dir_entry.path))
    return files_with_the_right_name


def find_all_train_paths(path):
    return [Path(folder).parent for folder in find_files(path, "actor.keras")]


def latest_train_folder(path):
    return max(find_all_train_paths(path), key=os.path.getctime)


def concatenate_lists(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def on_save(actor: Model, q_network: Model, epoch:int, replay_buffer, replay_save:bool, save_path:str):
    epoch_path = Path(save_path, str(epoch))
    os.makedirs(epoch_path, exist_ok=True)
    actor.save(epoch_path / "actor.keras")
    q_network.save(epoch_path / "critic.keras")
    if replay_save:
        with open( Path(save_path, "replay.p"), "wb" ) as replay_file:
            pickle.dump( replay_buffer, replay_file)

def load_critic(folder):
    return models.load_model(Path(folder, "critic.keras"))

def load_actor(folder):
    return models.load_model(Path(folder, "actor.keras"))

def load_replay(folder):
    return pickle.load(open(Path(folder, "replay.p"), "rb"))
