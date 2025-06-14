import cloudpickle
import os
import time
import numpy as np
import dask
import argparse
import uuid
from functools import wraps
import sys
from collections import defaultdict, deque
from rich.progress import Progress, SpinnerColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn
import hashlib
import json
import os
import collections
import shutil
import types
from dask.utils import ensure_dict
from dask.highlevelgraph import HighLevelGraph, MaterializedLayer
from dask.optimization import SubgraphCallable

def create_progress_bar():
    return Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        MofNCompleteColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        refresh_per_second=10,
    )

def timed(label=""):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = fn(*args, **kwargs)
            duration = time.time() - start
            print(f"[{label or fn.__name__}] took {round(duration, 4)} seconds")
            return result
        return wrapper
    return decorator

@timed("Load compute keys")
def load_compute_keys(file_path="compute_keys.pkl"):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            compute_keys = cloudpickle.load(f)
    else:
        raise FileNotFoundError("compute_keys.pkl not found")
    return compute_keys

@timed("Load hlg")
def load_hlg(file_path="expanded_hlg.pkl"):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            hlg = cloudpickle.load(f)
    else:
        raise FileNotFoundError("expanded_hlg.pkl not found")
    return hlg

def checkpoint_to(obj, file_path):
    with open(file_path, "wb") as f:
        cloudpickle.dump(obj, f)

def load_from(file_path):
    with open(file_path, "rb") as f:
        return cloudpickle.load(f)
    
def print_size_of(obj, label="Object"):
    from pympler import asizeof
    size = asizeof.asizeof(obj)
    if size < 1024:
        print(f"{label} size: {size} Bytes")
    elif size < 1024 ** 2:
        print(f"{label} size: {size / 1024:.2f} KB")
    elif size < 1024 ** 3:
        print(f"{label} size: {size / (1024 ** 2):.2f} MB")
    else:
        print(f"{label} size: {size / (1024 ** 3):.2f} GB")

def hash_name(*args):
    out_str = ""
    for arg in args:
        out_str += str(arg)
    return hashlib.sha256(out_str.encode('utf-8')).hexdigest()[:12]

def stringify_callable(fn):
    if hasattr(fn, "__module__") and hasattr(fn, "__name__"):
        return f"{fn.__module__}.{fn.__name__}"
    else:
        pickled = cloudpickle.dumps(fn)
        return f"func_{hashlib.md5(pickled).hexdigest()}"

def serialize_obj(obj, valid_keys):
    if isinstance(obj, tuple):
        if obj in valid_keys:
            return str(obj)
        else:
            return [serialize_obj(i, valid_keys) for i in obj]
    elif isinstance(obj, list):
        return [serialize_obj(i, valid_keys) for i in obj]
    elif isinstance(obj, dict):
        return {str(serialize_obj(k, valid_keys)): serialize_obj(v, valid_keys) for k, v in obj.items()}
    elif callable(obj):
        return stringify_callable(obj)
    else:
        return obj

def hashable(s):
    try:
        hash(s)
        return True
    except TypeError:
        return False

def flatten_hlg(hlg):
    assert isinstance(hlg, HighLevelGraph)
    task_dict = {}
    for k, sexpr in hlg.items():
        if isinstance(sexpr, SubgraphCallable):
            args = sexpr.args
            outkey = sexpr.outkey
            task_dict[k] = hash_name(k, outkey)
            for subkey, subexpr in sexpr.dsk.items():
                unique_key = hash_name(k, subkey)
                converted = TaskGraph.convert_expr_to_task_args(None, sexpr.dsk, subexpr, args)
                task_dict[unique_key] = converted
        elif isinstance(sexpr, tuple):
            task_dict[k] = sexpr
        elif isinstance(sexpr, (int, float, str)):
            task_dict[k] = sexpr
        else:
            raise ValueError(f"Unexpected type in HLG: {type(sexpr)}")
    return task_dict