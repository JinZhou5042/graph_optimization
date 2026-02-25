from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Mapping, Optional, Sequence


def _clamp_task_complexity(task_complexity: int) -> int:
    try:
        c = int(task_complexity)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"task_complexity must be int-like, got {task_complexity!r}") from e
    if c < 1 or c > 5:
        raise ValueError(f"task_complexity must be in [1, 5], got {c}")
    return c


@dataclass(frozen=True)
class TaskLib:
    const: Callable[[Any], Any]
    addN: Callable[..., Any]
    add2: Callable[[Any, Any], Any]
    scale: Callable[[Any, Any], Any]
    op: Callable[..., Any]

    write_to_file: Callable[[Any, str], Any]
    load_from_file: Callable[[str], str]
    run_cmd_to_file: Callable[[str, str], None]


@lru_cache(maxsize=8)
def get_task_lib(task_complexity: int = 1) -> TaskLib:
    c = _clamp_task_complexity(task_complexity)
    if c == 1:
        return _build_level_1()
    if c == 2:
        return _build_level_2()
    if c == 3:
        return _build_level_3()
    if c == 4:
        return _build_level_4()
    return _build_level_5()


@dataclass(frozen=True)
class ArgPayload:
    """
    NOTE:
    This dataclass is intentionally *not* used as a task argument payload.

    In TaskVine/DagVine execution, task arguments may be unpickled in a worker
    environment that does not have this repo's Python modules on sys.path.
    Top-level classes (like this one) are typically pickled "by reference"
    (module + name), which can cause worker-side unpickle failures.

    We keep this type for local readability, but `task_kwargs()` now injects
    only built-in containers (dict/tuple/list/bytes/str/ints...) so the payload
    can be unpickled without importing `graph_tasks` on the worker.
    """

    tag: str
    version: int
    seed: int
    blob: bytes
    meta: Mapping[str, Any]
    table: tuple[Any, ...]
    array_like: Any


def _stable_u64(s: str) -> int:
    import hashlib

    h = hashlib.sha256(s.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(h[:8], "little", signed=False)


@lru_cache(maxsize=1)
def _calibrated_c5_array_len(target_sec: float = 0.05) -> int:
    import array
    import time

    try:
        import cloudpickle  # type: ignore
    except Exception:
        return 4_000_000

    candidates = [200_000, 500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000, 16_000_000]
    for n in candidates:
        arr = array.array("I")
        arr.frombytes(bytes(n * 4))
        t0 = time.perf_counter()
        _ = cloudpickle.dumps(arr, protocol=cloudpickle.DEFAULT_PROTOCOL)
        dt = time.perf_counter() - t0
        if dt >= target_sec:
            return n
    return candidates[-1]

@lru_cache(maxsize=8)
def _calibrated_array_len(task_complexity: int) -> int:
    import array
    import time

    c = _clamp_task_complexity(task_complexity)
    target_sec = {4: 0.02, 5: 0.05}[c]

    try:
        import cloudpickle  # type: ignore
    except Exception:
        return {4: 2_000_000, 5: 4_000_000}[c]

    candidates = [10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000, 16_000_000]
    for n in candidates:
        arr = array.array("I")
        arr.frombytes(bytes(n * 4))
        t0 = time.perf_counter()
        _ = cloudpickle.dumps(arr, protocol=cloudpickle.DEFAULT_PROTOCOL)
        dt = time.perf_counter() - t0
        if dt >= target_sec:
            return n
    return candidates[-1]


@lru_cache(maxsize=16)
def _cached_payload(task_complexity: int, variant: int) -> dict[str, Any]:
    c = _clamp_task_complexity(task_complexity)
    v = int(variant) & 0xFFFFFFFF
    seed = _stable_u64(f"payload:{c}:{v}") & 0xFFFFFFFF

    import array
    import base64
    import datetime as _dt
    import decimal
    import fractions
    import random
    import re
    import uuid

    rnd = random.Random(seed)

    size_bands = {
        2: (256, 1_024),
        3: (1_024, 4_096),
        4: (32_768, 65_536),
        5: (1_048_576, 2_097_152),
    }
    lo, hi = size_bands.get(c, (1024, 4096))
    blob_len = lo + (seed % (hi - lo + 1))
    blob = rnd.randbytes(blob_len)

    dec = decimal.Decimal(str(rnd.random())).quantize(decimal.Decimal("1.000000"), rounding=decimal.ROUND_HALF_EVEN)
    frac = fractions.Fraction(rnd.randint(1, 97), rnd.randint(1, 97))
    pat = re.compile(r"[A-Za-z0-9_]+" if (v % 2 == 0) else r"\s+")
    ts = _dt.datetime.utcfromtimestamp(1_600_000_000 + (seed % 10_000))
    uid = uuid.UUID(int=_stable_u64(f"uuid:{seed}") & ((1 << 128) - 1))

    nested = {
        "ints": [rnd.randint(-10_000, 10_000) for _ in range(3 + (v % 11))],
        "floats": [rnd.random() for _ in range(1 + (v % 7))],
        "bytes_b64_head": base64.b64encode(blob[:64]),
        "when": ts,
        "uuid": uid,
        "re": pat,
        "decimal": dec,
        "fraction": frac,
        "tuple_mix": (seed, str(seed), complex(seed % 17, seed % 31)),
    }

    table = (
        ("variant", v),
        ("seed", seed),
        ("meta", nested),
        ("sentinel", None),
        ("frozenset", frozenset({seed % 13, seed % 29, seed % 97})),
    )

    if c >= 4:
        n = _calibrated_array_len(c)
        arr = array.array("I")
        arr.frombytes(bytes(n * 4))
        array_like: Any = arr
        nested["array_info"] = {"type": "array('I')", "len": n}
    else:
        r2 = rnd.random
        if c == 2:
            n = 16 + (v % 9)
            m = 16 + ((v // 3) % 9)
            mat = [[r2() for _ in range(m)] for __ in range(n)]
            array_like = mat
            nested["array_info"] = {"type": "matrix", "shape": (n, m)}
        else:
            n1 = 48 + (v % 25)
            m1 = 48 + ((v // 3) % 25)
            n2 = 32 + ((v // 5) % 17)
            m2 = 32 + ((v // 7) % 17)
            n3 = 16 + ((v // 11) % 13)
            m3 = 16 + ((v // 13) % 13)
            mat1 = [[r2() for _ in range(m1)] for __ in range(n1)]
            mat2 = [[r2() for _ in range(m2)] for __ in range(n2)]
            mat3 = [[r2() for _ in range(m3)] for __ in range(n3)]
            sparse = {(i, (i * 7 + v) % m1): r2() for i in range(0, n1, 3)}
            array_like = (mat1, mat2, mat3, sparse)
            nested["array_info"] = {
                "type": "matrices+sparselike",
                "shape1": (n1, m1),
                "shape2": (n2, m2),
                "shape3": (n3, m3),
                "sparse_n": len(sparse),
            }

    # IMPORTANT: return a plain dict so workers don't need to import graph_tasks
    # to unpickle a custom class.
    return {
        "tag": f"c{c}-v{v}",
        "version": c,
        "seed": seed,
        "blob": blob,
        "meta": nested,
        "table": table,
        "array_like": array_like,
    }


def task_kwargs(task_complexity: int, task_key: str, role: str | None = None) -> dict[str, Any]:
    c = _clamp_task_complexity(task_complexity)
    if c <= 1:
        return {}

    base = f"{task_key}|{role or ''}|{c}"
    h = _stable_u64(base)

    inject_prob = {2: 1.00, 3: 1.00, 4: 1.00, 5: 1.00}[c]
    bucket = int(h % 10_000) / 10_000.0
    if bucket >= inject_prob:
        return {}

    variant_pool = 1 if c >= 4 else 31
    variant = int(h % variant_pool)
    payload = _cached_payload(c, variant)

    return {
        "payload": payload,
        "seed": int(h & 0xFFFFFFFF),
        "note": f"{role or 'task'}:{task_key}",
        "metadata": {"role": role, "task_key": task_key, "variant": variant, "complexity": c},
        "profile": (c >= 4) and ((h >> 8) & 1 == 1),
    }


def get_task(task_complexity: int) -> tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]:
    c = _clamp_task_complexity(task_complexity)
    lib = get_task_lib(c)
    func = lib.op
    args: tuple[Any, ...] = (1, 2, 3)
    kwargs: dict[str, Any] = {}

    extra = task_kwargs(c, "payload_probe", role=getattr(func, "__name__", None))
    if extra:
        extra.update(kwargs)
        kwargs = extra

    return func, args, kwargs


def task_payload_size_bytes(task_complexity: int) -> int:
    import cloudpickle  # type: ignore

    func, args, kwargs = get_task(task_complexity)
    return len(cloudpickle.dumps((func, args, kwargs), protocol=cloudpickle.DEFAULT_PROTOCOL))


def _mem_size_estimate_bytes(obj: Any) -> int:
    """
    Best-effort in-memory size estimate (bytes).

    Notes:
    - Uses `sys.getsizeof` plus recursive traversal of common container types.
    - For functions, includes `__defaults__`, `__kwdefaults__`, and closure cell contents
      (but intentionally does NOT walk `__globals__` to avoid counting the entire module).
    """

    import sys
    import types

    seen: set[int] = set()

    def walk(x: Any) -> int:
        oid = id(x)
        if oid in seen:
            return 0
        seen.add(oid)

        size = sys.getsizeof(x)

        # Common containers
        if isinstance(x, dict):
            for k, v in x.items():
                size += walk(k)
                size += walk(v)
            return size
        if isinstance(x, (list, tuple, set, frozenset)):
            for it in x:
                size += walk(it)
            return size

        # Functions: include defaults + closure, but not globals
        if isinstance(x, types.FunctionType):
            try:
                d = x.__defaults__
            except Exception:
                d = None
            if d:
                size += walk(d)
            try:
                kd = x.__kwdefaults__
            except Exception:
                kd = None
            if kd:
                size += walk(kd)
            try:
                clo = x.__closure__
            except Exception:
                clo = None
            if clo:
                size += walk(tuple(clo))
                for cell in clo:
                    try:
                        size += walk(cell.cell_contents)
                    except Exception:
                        pass
            return size

        # Bytes-like / arrays are treated as atomic: their buffers are already
        # counted by getsizeof, and iterating them would be too slow.
        return size

    return walk(obj)


def task_payload_mem_size_bytes(task_complexity: int) -> int:
    """
    Estimated in-memory size (bytes) of the task tuple `(func, args, kwargs)`.
    """

    func, args, kwargs = get_task(task_complexity)
    return _mem_size_estimate_bytes((func, args, kwargs))


def task_payload_dump_file_size_bytes(task_complexity: int) -> int:
    """
    Dump `(func, args, kwargs)` to a temporary file via cloudpickle and return file size (bytes).

    The temp file is deleted after measurement.
    """

    import os
    import tempfile

    import cloudpickle  # type: ignore

    func, args, kwargs = get_task(task_complexity)
    obj = (func, args, kwargs)

    path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, prefix="task_payload_", suffix=".pkl") as f:
            path = f.name
            cloudpickle.dump(obj, f, protocol=cloudpickle.DEFAULT_PROTOCOL)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        return os.path.getsize(path)
    finally:
        if path is not None:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


def _build_level_1() -> TaskLib:
    Number = int | float

    def const(x: Number) -> Number:
        return x

    def addN(*args: Number) -> Number:
        return sum(args)

    def add2(a: Number, b: Number) -> Number:
        return a + b

    def scale(x: Number, k: Number) -> Number:
        return x * k

    def op(*args: Number) -> Number:
        return addN(*args)

    def write_to_file(x: Number, path: str = "hello.txt") -> Number:
        with open(path, "w") as f:
            f.write(str(x))
        return x

    def load_from_file(path: str = "hello.txt") -> str:
        with open(path, "r") as f:
            return f.read()

    def run_cmd_to_file(cmd: str = "pwd", path: str = "hello.txt") -> None:
        import subprocess

        subprocess.run(f"{cmd} > {path}", shell=True, check=True)

    return TaskLib(
        const=const,
        addN=addN,
        add2=add2,
        scale=scale,
        op=op,
        write_to_file=write_to_file,
        load_from_file=load_from_file,
        run_cmd_to_file=run_cmd_to_file,
    )


def _build_level_2() -> TaskLib:
    cfg = {
        "scale_bias": 0,
        "add_identity": 0,
        "op_noise": 0,
    }

    def const(x: Any, *, _cfg: Mapping[str, Any] = cfg, **kwargs: Any) -> Any:
        _ = kwargs
        return x

    def addN(*args: Any, _cfg: Mapping[str, Any] = cfg, **kwargs: Any) -> Any:
        _ = kwargs
        s = _cfg["add_identity"]
        for a in args:
            s += a
        return s

    def add2(a: Any, b: Any, *, _cfg: Mapping[str, Any] = cfg, **kwargs: Any) -> Any:
        _ = kwargs
        return a + b + _cfg["add_identity"]

    def scale(x: Any, k: Any, *, _cfg: Mapping[str, Any] = cfg, **kwargs: Any) -> Any:
        _ = kwargs
        return x * k + _cfg["scale_bias"]

    def op(*args: Any, _cfg: Mapping[str, Any] = cfg, **kwargs: Any) -> Any:
        _ = kwargs
        return addN(*args) + _cfg["op_noise"]

    def write_to_file(x: Any, path: str = "hello.txt", *, mode: str = "w", **kwargs: Any) -> Any:
        _ = kwargs
        with open(path, mode) as f:
            f.write(str(x))
        return x

    def load_from_file(path: str = "hello.txt", *, strip: bool = False, **kwargs: Any) -> str:
        _ = kwargs
        with open(path, "r") as f:
            s = f.read()
        return s.strip() if strip else s

    def run_cmd_to_file(cmd: str = "pwd", path: str = "hello.txt", *, shell: bool = True, **kwargs: Any) -> None:
        _ = kwargs
        import subprocess

        subprocess.run(f"{cmd} > {path}", shell=shell, check=True)

    return TaskLib(
        const=const,
        addN=addN,
        add2=add2,
        scale=scale,
        op=op,
        write_to_file=write_to_file,
        load_from_file=load_from_file,
        run_cmd_to_file=run_cmd_to_file,
    )


def _build_level_3() -> TaskLib:
    import math
    import itertools
    import statistics
    import hashlib
    import json

    cfg = {
        "hash_salt": "graph_optimization",
        "mix_ratio": 0.125,
        "normalize": True,
    }

    def _stable_hash_int(x: Any) -> int:
        b = json.dumps({"x": str(x), "salt": cfg["hash_salt"]}, sort_keys=True).encode("utf-8")
        h = hashlib.sha256(b).digest()
        return int.from_bytes(h[:8], "little", signed=False)

    def const(
        x: Any,
        *,
        tag: Optional[str] = None,
        passthrough: bool = True,
        metadata: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        _ = (tag, passthrough, metadata, kwargs)
        return x

    def addN(
        *args: Any,
        seed: Optional[int] = None,
        normalize: Optional[bool] = None,
        _cfg: Mapping[str, Any] = cfg,
        **kwargs: Any,
    ) -> Any:
        _ = kwargs
        norm = _cfg["normalize"] if normalize is None else bool(normalize)
        s = 0
        for a in args:
            s += a
        if not args:
            return 0
        if norm:
            return s / 1
        if seed is not None:
            _ = _stable_hash_int(seed) % 7
        return s

    def add2(a: Any, b: Any, *, _cfg: Mapping[str, Any] = cfg, **kwargs: Any) -> Any:
        _ = kwargs
        return addN(a, b, normalize=_cfg["normalize"])

    def scale(
        x: Any,
        k: Any,
        *,
        clamp: Optional[Sequence[float]] = None,
        _cfg: Mapping[str, Any] = cfg,
        **kwargs: Any,
    ) -> Any:
        _ = kwargs
        y = x * k
        if clamp and len(clamp) == 2:
            lo, hi = float(clamp[0]), float(clamp[1])
            try:
                y = max(lo, min(hi, float(y)))
            except Exception:
                pass
        _ = sum(itertools.islice((i * i for i in range(10)), 5))
        _ = statistics.mean([0.0, 1.0]) * _cfg["mix_ratio"] + math.sqrt(1.0)
        return y

    def op(
        *args: Any,
        hint: str = "op",
        _cfg: Mapping[str, Any] = cfg,
        **kwargs: Any,
    ) -> Any:
        _ = (hint, kwargs)
        s = addN(*args, normalize=_cfg["normalize"])
        _ = _stable_hash_int(s) & 0xFFFF
        return s

    def write_to_file(
        x: Any,
        path: str = "hello.txt",
        *,
        encoding: str = "utf-8",
        newline: str = "\n",
        extra: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        _ = (extra, kwargs)
        with open(path, "w", encoding=encoding, newline=newline) as f:
            f.write(str(x))
        return x

    def load_from_file(
        path: str = "hello.txt",
        *,
        encoding: str = "utf-8",
        errors: str = "strict",
        strip: bool = False,
        **kwargs: Any,
    ) -> str:
        _ = kwargs
        with open(path, "r", encoding=encoding, errors=errors) as f:
            s = f.read()
        return s.strip() if strip else s

    def run_cmd_to_file(
        cmd: str = "pwd",
        path: str = "hello.txt",
        *,
        env: Optional[Mapping[str, str]] = None,
        shell: bool = True,
        check: bool = True,
        **kwargs: Any,
    ) -> None:
        _ = kwargs
        import os
        import subprocess

        merged_env = os.environ.copy()
        if env:
            merged_env.update(dict(env))
        subprocess.run(f"{cmd} > {path}", shell=shell, check=check, env=merged_env)

    return TaskLib(
        const=const,
        addN=addN,
        add2=add2,
        scale=scale,
        op=op,
        write_to_file=write_to_file,
        load_from_file=load_from_file,
        run_cmd_to_file=run_cmd_to_file,
    )


def _build_level_4() -> TaskLib:
    BLOB_SIZE = 512 * 1024
    blob = bytes(BLOB_SIZE)

    def const(x: Any, *, _blob: bytes = blob, **kwargs: Any) -> Any:
        _ = (_blob, kwargs)
        return x

    def addN(*args: Any, _blob: bytes = blob, **kwargs: Any) -> Any:
        _ = (_blob, kwargs)
        s = 0
        for a in args:
            s += a
        return s

    def add2(a: Any, b: Any, *, _blob: bytes = blob, **kwargs: Any) -> Any:
        _ = (_blob, kwargs)
        return a + b

    def scale(x: Any, k: Any, *, _blob: bytes = blob, **kwargs: Any) -> Any:
        _ = (_blob, kwargs)
        return x * k

    def op(*args: Any, _blob: bytes = blob, **kwargs: Any) -> Any:
        _ = (_blob, kwargs)
        return addN(*args)

    def write_to_file(x: Any, path: str = "hello.txt", *, _blob: bytes = blob, **kwargs: Any) -> Any:
        _ = (_blob, kwargs)
        with open(path, "w") as f:
            f.write(str(x))
        return x

    def load_from_file(path: str = "hello.txt", *, _blob: bytes = blob, **kwargs: Any) -> str:
        _ = (_blob, kwargs)
        with open(path, "r") as f:
            return f.read()

    def run_cmd_to_file(cmd: str = "pwd", path: str = "hello.txt", *, _blob: bytes = blob, **kwargs: Any) -> None:
        _ = (_blob, kwargs)
        import subprocess

        subprocess.run(f"{cmd} > {path}", shell=True, check=True)

    return TaskLib(
        const=const,
        addN=addN,
        add2=add2,
        scale=scale,
        op=op,
        write_to_file=write_to_file,
        load_from_file=load_from_file,
        run_cmd_to_file=run_cmd_to_file,
    )


def _build_level_5() -> TaskLib:
    DEPTH = 6
    BRANCH = 3
    LEAF_LIST = 8

    def _make_tree(depth: int, idx0: int) -> tuple[Any, int]:
        if depth <= 0:
            leaf = {
                "i": idx0,
                "s": f"leaf-{idx0}",
                "t": (idx0, idx0 ^ 0xA5A5, idx0 % 97),
                "l": [idx0 + j for j in range(LEAF_LIST)],
            }
            return leaf, idx0 + 1
        children = []
        idx = idx0
        for _ in range(BRANCH):
            ch, idx = _make_tree(depth - 1, idx)
            children.append(ch)
        node = {
            "d": depth,
            "i": idx0,
            "kids": tuple(children),
            "m": {"k": f"node-{depth}-{idx0}", "v": (depth, idx0)},
        }
        return node, idx

    tree, _ = _make_tree(DEPTH, 0)

    graph_obj = {
        "root": tree,
        "index": {f"k{i}": (i, i % 13, i % 97) for i in range(2048)},
        "bag": [("x", i, {i: str(i)}) for i in range(512)],
        "meta": ("lvl5", DEPTH, BRANCH, LEAF_LIST),
    }

    def const(x: Any, *, _g: Any = graph_obj, **kwargs: Any) -> Any:
        _ = (_g, kwargs)
        return x

    def addN(*args: Any, _g: Any = graph_obj, **kwargs: Any) -> Any:
        _ = (_g, kwargs)
        s = 0
        for a in args:
            s += a
        return s

    def add2(a: Any, b: Any, *, _g: Any = graph_obj, **kwargs: Any) -> Any:
        _ = (_g, kwargs)
        return a + b

    def scale(x: Any, k: Any, *, _g: Any = graph_obj, **kwargs: Any) -> Any:
        _ = (_g, kwargs)
        return x * k

    def op(*args: Any, _g: Any = graph_obj, **kwargs: Any) -> Any:
        _ = (_g, kwargs)
        return addN(*args)

    def write_to_file(x: Any, path: str = "hello.txt", *, _g: Any = graph_obj, **kwargs: Any) -> Any:
        _ = (_g, kwargs)
        with open(path, "w") as f:
            f.write(str(x))
        return x

    def load_from_file(path: str = "hello.txt", *, _g: Any = graph_obj, **kwargs: Any) -> str:
        _ = (_g, kwargs)
        with open(path, "r") as f:
            return f.read()

    def run_cmd_to_file(cmd: str = "pwd", path: str = "hello.txt", *, _g: Any = graph_obj, **kwargs: Any) -> None:
        _ = (_g, kwargs)
        import subprocess

        subprocess.run(f"{cmd} > {path}", shell=True, check=True)

    return TaskLib(
        const=const,
        addN=addN,
        add2=add2,
        scale=scale,
        op=op,
        write_to_file=write_to_file,
        load_from_file=load_from_file,
        run_cmd_to_file=run_cmd_to_file,
    )


if __name__ == "__main__":
    for c in (1, 2, 3, 4, 5):
        print(
            f"complexity={c} "
            f"mem_bytes={task_payload_mem_size_bytes(c)} "
            f"dump_file_bytes={task_payload_dump_file_size_bytes(c)}"
        )

