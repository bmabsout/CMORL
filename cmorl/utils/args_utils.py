from dataclasses import dataclass
from typing import Any, Optional, Iterator
import copy
from cmorl.utils.serialization_utils import serialize_dict
from functools import reduce
import argparse


@dataclass
class Serialized_Argument:
    name: str
    abbrev: Optional[str]
    default: Any
    kwargs: dict[str, Any]
    def __init__(self, name: str, abbrev: Optional[str] = None, default=False,  **kwargs):
        self.abbrev = abbrev
        self.name = name
        self.default = default
        self.kwargs = kwargs

    def id(self):
        return self.abbrev or self.name


class Arg_Serializer:
    args: list[Serialized_Argument]
    ignored: set[str]

    def __init__(
        self, *args: Serialized_Argument, ignored: set[str] = set()
    ) -> None:
        Arg_Serializer.assertions(*args)
        self.args=list(args)
        self.ignored=ignored
    
    def add_serialized_args_to_parser(self, parser):
        for ser_arg in self.args:
            parser.add_argument(*(["-" + ser_arg.abbrev] if ser_arg.abbrev else []), "--" + ser_arg.name, default=ser_arg.default, **ser_arg.kwargs)

    def __iter__(self) -> Iterator[Serialized_Argument]:
        return iter(self.args)
    
    def name_to_args(self):
        return {arg.name: arg for arg in self.args}

    @staticmethod
    def assertions(*args: Serialized_Argument):
        names = [arg.name for arg in args]
        non_empty_abbrevs = [arg.abbrev for arg in args if arg.abbrev is not None]
        assert "x" not in names, "Cannot use 'x' as a name, it's a reserved word"
        assert "x" not in non_empty_abbrevs, "Cannot use 'x' as an abbreviation, it's a reserved_word"
        assert len(names) == len(set(names)), f"Names overlap: {names}"
        assert len(non_empty_abbrevs) == len(set(non_empty_abbrevs)), f"Abrreviations overlap: {non_empty_abbrevs}"

    @staticmethod
    def join(*serializers: "Arg_Serializer"):
        return reduce(lambda as1, as2: Arg_Serializer(
            *as1.args, *as2.args, ignored=as1.ignored | as2.ignored
        ), serializers)
    
    def remove_names(self, names: set[str]):
        args_without_names = [arg for arg in self.args if arg.name not in names]
        return Arg_Serializer(*args_without_names, ignored=self.ignored - names)

    def get_semantic_folder_name(self, hypers: dict[str, Any]) -> str:
        return serialize_dict(**get_minified_args_dict(self, hypers))

    def get_seed_folder_path(self, experiment_name: str, hypers: dict[str, Any]) -> str:
        return f"trained/{experiment_name}/{self.get_semantic_folder_name(hypers)}/seeds/{hypers['seed']}"

    def parse_arguments(self, args=None, parser=argparse.ArgumentParser(), namespace=None):
        self.add_serialized_args_to_parser(parser)
        cmd_args = parser.parse_args(args, namespace=namespace)
        return cmd_args

def get_minified_args_dict(serializer: Arg_Serializer, hypers: dict[str, Any], show_defaults=False):
    ignored_args = {}
    extra_args = {}
    serialize_me: dict[str, Any] = {}
    name_to_args = serializer.name_to_args()
    for name, value in hypers.items():
        arg = name_to_args[name]
        if name in serializer.ignored:
            ignored_args[name] = value
            continue
        if arg is None: # not in the serializer
            extra_args[name] = value
            continue
        if show_defaults:
            serialize_me[arg.id()] = value
            continue
        if value == arg.default:
            ignored_args[name] = value
            continue

        serialize_me[arg.id()] = value
    return {"args":serialize_me,"ignored": ignored_args,"extra": extra_args}

def namespace_serializer(namespace: argparse.Namespace, ignored:set[str]=set(), descriptions:dict[str, str]={}, abbrevs:dict[str, str]={}):
    return Arg_Serializer(
        *(Serialized_Argument(name=k, type=type(v), abbrev=abbrevs.get(k), default=v, help=descriptions.get(k)) for k, v in vars(namespace).items())
        , ignored=ignored
    )

