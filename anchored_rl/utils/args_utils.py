from typing import Any, Optional, Iterator
import copy
from anchored_rl.utils.serialization_utils import serialize_dict
from functools import reduce
import argparse
import time

class Serialized_Argument:
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.kwargs = kwargs

def ignore_some_keys(hypers: dict[str, Any], keys = []):
    copied_hypers = copy.deepcopy(hypers)
    for ignore_me in keys:
        try:
            copied_hypers.pop(ignore_me)
        except:
            pass
    return copied_hypers


class Arg_Serializer:
    def __init__(self, abbrev_to_args: dict[str, Serialized_Argument], ignored: set[str] = set()) -> None:
        self.abbrev_to_args = abbrev_to_args
        self.ignored = ignored
        self.name_to_abbrev = {
            v.name.removeprefix("--"): k for k, v in abbrev_to_args.items()
        }

    def add_serialized_args_to_parser(self, parser):
        for abbreviation, ser_arg in self.abbrev_to_args.items():
            parser.add_argument("-"+abbreviation, ser_arg.name, **ser_arg.kwargs)


    @staticmethod
    def join(*serializers: Optional["Arg_Serializer"]):
        def join_two(as1: Arg_Serializer, as2: Arg_Serializer) -> Arg_Serializer:
            overlapping_names = as1.name_to_abbrev.keys() & as2.name_to_abbrev.keys()
            overlapping_abbrevs = as1.abbrev_to_args.keys() & as2.abbrev_to_args.keys()
            if overlapping_names or overlapping_abbrevs:
                raise Exception(
                    f"Cannot join two Arg_Serializers with overlapping names {overlapping_names} or abbreviations {overlapping_abbrevs}")
            return Arg_Serializer({**as1.abbrev_to_args, **as2.abbrev_to_args}, as1.ignored | as2.ignored)

        return reduce(join_two, [ s for s in serializers if s is not None])

    def get_minified_args_dict(self, hypers: dict[str, Any]):
        extra_args = {}
        serialize_me: dict[str, Any] = {}
        for name, value in hypers.items():
            abbrev = self.name_to_abbrev.get(name)
            if abbrev:
                serialize_me[abbrev] = value
            else:
                extra_args[name] = value
        serialize_me["x"] = extra_args
        return serialize_me

    def remove_ignored(self, hypers: dict[str, Any]):
        return ignore_some_keys(hypers, self.ignored)

    def get_semantic_folder_name(self, hypers: dict[str, Any]) -> str:
        return serialize_dict(self.get_minified_args_dict(self.remove_ignored(hypers)))

    def get_seed_folder_path(self, experiment_name: str, hypers: dict[str, Any]) -> str:
        return f"trained/{experiment_name}/{self.get_semantic_folder_name(hypers)}/seeds/{hypers['seed']}"


def rl_alg_serializer(epochs=50, learning_rate=3e-3):
    return Arg_Serializer(
        abbrev_to_args={
            'e': Serialized_Argument(name='--epochs', type=int, default=epochs, help='number of epochs'),
            's': Serialized_Argument(name='--seed', type=int, default=int(time.time() * 1e5) % int(1e6)),
            'l': Serialized_Argument(name='--learning_rate', type=float, default=learning_rate),
        },
        ignored={'save_path', 'seed'}
    )

def anchor_serializer():
    return Arg_Serializer(
        abbrev_to_args={
            'a': Serialized_Argument(name='--anchored', action='store_true', help='whether to enable anchors or not'),
            'p': Serialized_Argument(name='--prev_folder', type=str, help='folder location for a previous training run with initialized critics and actors'),
            'r': Serialized_Argument(name='--replay_save', action='store_true', help='whether to save the replay buffer')
        },
        ignored={'replay_save'}
    )


def default_serializer(epochs=50, learning_rate=3e-3):
    return Arg_Serializer.join(
        # ArgsSerializer({'n': Serialized_Argument(name='--experiment_name', type=str, required=True)}, ignored={'experiment_name'}),
        rl_alg_serializer(epochs, learning_rate), anchor_serializer())

def parse_arguments(serializer:Arg_Serializer, args=None, parser = None):
    if parser is None:
        parser = argparse.ArgumentParser()
    serializer.add_serialized_args_to_parser(parser)

    cmd_args = parser.parse_args(args)
    if cmd_args.anchored and not cmd_args.prev_folder:
        parser.error("cannot enable anchors without specifying --prev_folder")
    return cmd_args