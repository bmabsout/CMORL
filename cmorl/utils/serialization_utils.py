import json
from json import JSONEncoder
import numpy
import hashlib
import base64
from typing import Any
import copy
from pathlib import Path

class ExtraTypesEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        if callable(obj):
            return str(obj)
        return JSONEncoder.default(self, obj)


def hash_it(s: str, length=15):
    m = hashlib.shake_256()
    m.update(s.encode("ASCII"))
    return base64.b32encode(m.digest(int(length))).decode()[0:length]


def serialize_leaf(o: Any) -> str:
    if type(o) == dict:
        return hash_it(json.dumps(o, sort_keys=True, cls=ExtraTypesEncoder))
    if isinstance(o, Path):
        return hash_it(str(o), length=7)
    if type(o) == str:
        return hash_it(o, length=7)
    if type(o) == float:
        return format(o, ".4g")

    return str(o)


def serialize_dict(args: dict) -> str:
    return ",".join(f"{k}:{serialize_leaf(v)}" for k, v in sorted(args.items(), key=lambda kv: kv[0]) if v)
