import cloudpickle
import os
import time
import numpy as np
import dask
import argparse
import uuid
from functools import wraps
import sys
from pympler import asizeof
from graphviz import Digraph
from collections import defaultdict, deque
from rich.progress import Progress, SpinnerColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn
import hashlib
import json
import os
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

@timed("Save task dict to json")
def save_task_dict_to_json(task_dict, output_path="task_dict.json"):
    valid_keys = set(task_dict.keys())
    result = {}
    for k, v in task_dict.items():
        k_str = str(k)
        if isinstance(v, tuple) and callable(v[0]):
            fn_str = stringify_callable(v[0])
            args_serialized = [serialize_obj(arg, valid_keys) for arg in v[1:]]
            result[k_str] = [fn_str] + args_serialized
        else:
            result[k_str] = serialize_obj(v, valid_keys)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

@timed("Save task dependencies")
def save_task_dependencies(task_dict_path="task_dict.json", output_path="task_dependencies.txt"):
    with open(task_dict_path) as f:
        task_dict = json.load(f)

    keys = list(task_dict.keys())
    key_to_id = {k: str(i + 1) for i, k in enumerate(keys)}
    key_set = set(keys)

    edges = []
    has_parent = set()

    for k, v in task_dict.items():
        def find_parents(val):
            if isinstance(val, (list, tuple)):
                for x in val:
                    find_parents(x)
            elif isinstance(val, str) and val in key_set and val != k:
                edges.append(f"{key_to_id[val]} -> {key_to_id[k]}")
                has_parent.add(k)

        find_parents(v)

    for k in task_dict:
        if k not in has_parent:
            edges.append(f"None -> {key_to_id[k]}")

    with open(output_path, "w") as f:
        f.write("\n".join(edges))

@timed("Visualize task dependencies")
def visualize_task_dependencies(input_file="task_dependencies.txt", output_file="task_graph"):
    dot = Digraph(format='svg')
    with open(input_file) as f:
        for line in f:
            parts = line.strip().split("->")
            if len(parts) != 2:
                continue
            parent = parts[0].strip()
            child = parts[1].strip()
            if parent != "None":
                dot.edge(parent, child)
            else:
                dot.node(child)
    dot.render(output_file, cleanup=True)

def hashable(s):
    try:
        hash(s)
        return True
    except TypeError:
        return False