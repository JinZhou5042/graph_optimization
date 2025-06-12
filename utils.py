import cloudpickle
import os
import time
import numpy as np
import argparse
import uuid
from functools import wraps
import sys
from pympler import asizeof
from collections import defaultdict, deque
from rich.progress import Progress, SpinnerColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn

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
            print(f"[{label or fn.__name__}] loaded in {round(duration, 4)} seconds")
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