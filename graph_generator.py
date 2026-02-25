import re
from collections import defaultdict, deque
import argparse
import random
import cloudpickle
import math
import os
import numpy as np
from ndcctools.taskvine.dagvine.blueprint_graph.blueprint_graph import BlueprintGraph, TaskOutputRef

import graph_tasks


MAX_TASK_PARENTS = 1000
MAX_TASK_CHILDREN = 1000
ENABLE_DEGREE_ASSERTS = os.environ.get("GO_DEGREE_ASSERTS", "1") != "0"
MAX_SUBGRAPH_TASKS = 10_000
ENABLE_SUBGRAPH_ASSERTS = os.environ.get("GO_SUBGRAPH_ASSERTS", "1") != "0"


def _task_parent_keys_from_args(args):
    """
    Extract dependency task keys from a BlueprintGraph task's args tuple.

    BlueprintGraph stores each task as (callable_id, args_tuple, kwargs_dict) in `bg.task_dict`.
    Dependencies are represented by TaskOutputRef objects inside args_tuple.
    """
    out = []
    for a in args:
        if isinstance(a, TaskOutputRef):
            out.append(a.task_key)
    return out


def _reduction_internal_nodes_count(n_leaves: int, fanin: int) -> int:
    """
    Number of internal aggregation nodes needed to reduce `n_leaves` to 1 using a k-ary tree
    with arity `fanin`.

    Example: n=10, fanin=1000 -> 1
             n=1001, fanin=1000 -> 3  (2 + 1)
    """
    assert fanin >= 2
    if n_leaves <= 1:
        return 0
    total = 0
    n = n_leaves
    while n > 1:
        n = (n + fanin - 1) // fanin
        total += n
    return total


def _add_kary_reduction(
    bg: BlueprintGraph,
    lib: graph_tasks.TaskLib,
    keys: list[str],
    *,
    root_key: str | None = None,
    prefix: str = "RED",
    fanin: int = MAX_TASK_PARENTS,
):
    """
    Add a k-ary reduction tree (fanin <= MAX_TASK_PARENTS) over `keys`.

    Returns the root task key (either `root_key` if provided, otherwise an auto-generated one).
    """
    assert fanin >= 2
    if not keys:
        rk = root_key or f"{prefix}0"
        bg.add_task(rk, lib.addN, 0)
        return rk
    if len(keys) == 1:
        rk = root_key or f"{prefix}0"
        # Keep addN arity >= 2 to match other generators' behavior.
        bg.add_task(rk, lib.addN, TaskOutputRef(keys[0]), 0)
        return rk

    cur = list(keys)
    level = 0
    idx = 0
    while len(cur) > 1:
        nxt = []
        for i in range(0, len(cur), fanin):
            grp = cur[i : i + fanin]
            if len(grp) == 1:
                nxt.append(grp[0])
                continue
            if root_key is not None and len(cur) <= fanin and len(grp) == len(cur):
                rk = root_key
            else:
                rk = f"{prefix}{level}_{idx}"
                idx += 1
            bg.add_task(rk, lib.addN, *[TaskOutputRef(k) for k in grp])
            nxt.append(rk)
        cur = nxt
        level += 1

    return cur[0]


def _assert_degree_constraints(bg: BlueprintGraph, *, max_parents: int = MAX_TASK_PARENTS, max_children: int = MAX_TASK_CHILDREN):
    """
    Hard validation: every task has <= max_parents parents and every task has <= max_children children.

    Notes:
    - Parents are counted as TaskOutputRef deps in the args tuple.
    - Children are computed by scanning all edges once.
    """
    if not ENABLE_DEGREE_ASSERTS:
        return
    child_counts: dict[str, int] = defaultdict(int)

    for task_key, (_call_id, args, _kwargs) in bg.task_dict.items():
        parents = _task_parent_keys_from_args(args)
        if len(parents) > max_parents:
            fn = None
            try:
                fn = bg.callables[_call_id].__name__
            except Exception:
                fn = str(_call_id)
            raise AssertionError(f"task {task_key!r} has {len(parents)} parents > {max_parents} (func={fn})")
        for p in parents:
            child_counts[p] += 1

    for p, c in child_counts.items():
        if c > max_children:
            raise AssertionError(f"task {p!r} has {c} children > {max_children}")


def _connected_component_sizes(bg: BlueprintGraph) -> list[int]:
    """
    Return sizes of weakly-connected components (treat edges as undirected).

    Components are defined over task keys using TaskOutputRef dependencies as edges.
    """
    adj: dict[str, set[str]] = {k: set() for k in bg.task_dict.keys()}

    for task_key, (_call_id, args, _kwargs) in bg.task_dict.items():
        parents = _task_parent_keys_from_args(args)
        for p in parents:
            if p not in adj:
                # In practice parents should always be tasks, but be defensive.
                adj[p] = set()
            adj[task_key].add(p)
            adj[p].add(task_key)

    seen: set[str] = set()
    sizes: list[int] = []

    for start in adj.keys():
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        n = 0
        while q:
            u = q.popleft()
            n += 1
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        sizes.append(n)

    sizes.sort(reverse=True)
    return sizes


def _assert_subgraph_size_constraints(bg: BlueprintGraph, *, max_tasks_per_subgraph: int = MAX_SUBGRAPH_TASKS):
    """
    Hard validation: every connected subgraph (weakly-connected component) has <= max_tasks_per_subgraph tasks.
    """
    if not ENABLE_SUBGRAPH_ASSERTS:
        return
    sizes = _connected_component_sizes(bg)
    if sizes and sizes[0] > max_tasks_per_subgraph:
        raise AssertionError(f"largest subgraph has {sizes[0]} tasks > {max_tasks_per_subgraph} (all_sizes_top5={sizes[:5]})")


def _clone_blueprint_graph_into(dst: BlueprintGraph, src: BlueprintGraph, *, prefix: str):
    """
    Clone all tasks + file deps from `src` into `dst`, renaming keys by `prefix`.

    This is used to build a disjoint union of multiple subgraphs while avoiding key collisions.
    """
    # Tasks
    for task_key, (call_id, args, kwargs) in src.task_dict.items():
        func = src.callables[call_id]
        new_key = f"{prefix}{task_key}"
        new_args = []
        for a in args:
            if isinstance(a, TaskOutputRef):
                new_args.append(TaskOutputRef(f"{prefix}{a.task_key}"))
            else:
                new_args.append(a)
        dst.add_task(new_key, func, *new_args, **kwargs)

    # File deps (best-effort): ensure filenames don't collide across subgraphs.
    if hasattr(src, "producer_of") and hasattr(dst, "producer_of"):
        for fname, producer in getattr(src, "producer_of", {}).items():
            dst.producer_of[f"{prefix}{fname}"] = f"{prefix}{producer}"
    if hasattr(src, "consumers_of") and hasattr(dst, "consumers_of"):
        for fname, consumers in getattr(src, "consumers_of", {}).items():
            dst.consumers_of[f"{prefix}{fname}"].update({f"{prefix}{c}" for c in consumers})


def _split_into_disjoint_subgraphs_by_total_tasks(
    build_one,
    total_tasks: int,
    *,
    task_complexity: int,
    max_tasks_per_subgraph: int = MAX_SUBGRAPH_TASKS,
    prefix_base: str,
):
    """
    Build a disjoint union graph by repeatedly calling `build_one(n)` with n <= max_tasks_per_subgraph.

    Important: all cloned keys are prefixed, so the resulting graph will NOT contain a raw key named "output".
    This ensures the runner does not accidentally target only a single component.
    """
    assert total_tasks >= 1
    if total_tasks <= max_tasks_per_subgraph:
        return build_one(total_tasks, task_complexity=task_complexity)

    dst = BlueprintGraph()
    dst = _install_task_kwargs_injector(dst, task_complexity)

    remaining = total_tasks
    i = 0
    while remaining > 0:
        n = min(remaining, max_tasks_per_subgraph)
        part = build_one(n, task_complexity=task_complexity)
        _clone_blueprint_graph_into(dst, part, prefix=f"{prefix_base}{i}_")
        remaining -= n
        i += 1

    _assert_degree_constraints(dst)
    _assert_subgraph_size_constraints(dst, max_tasks_per_subgraph=max_tasks_per_subgraph)
    return dst


def _install_task_kwargs_injector(bg: BlueprintGraph, task_complexity: int) -> BlueprintGraph:
    if task_complexity <= 1:
        return bg

    orig_add_task = bg.add_task

    def add_task(key, func, *args, **kwargs):
        role = getattr(func, "__name__", None)
        extra = graph_tasks.task_kwargs(task_complexity, str(key), role=role)
        if extra:
            extra.update(kwargs)
            kwargs = extra
        return orig_add_task(key, func, *args, **kwargs)

    bg.add_task = add_task
    return bg


def make_simple_graph(task_complexity: int = 1):
    bg = BlueprintGraph()
    bg = _install_task_kwargs_injector(bg, task_complexity)
    lib = graph_tasks.get_task_lib(task_complexity)

    bg.add_task("t1", lib.addN, 1, 5)
    
    bg.add_task("t2", lib.run_cmd_to_file,)
    bg.task_produces("t2", "hello.txt")
    
    bg.add_task("output", lib.load_from_file,)
    bg.task_consumes("output", "hello.txt")
    _assert_degree_constraints(bg)
    return bg


def make_binary_forest(branches=1, level=1, task_complexity: int = 1):
    assert level >= 1 and branches >= 1

    # If the whole forest would create a >10k connected component, split by building
    # multiple independent forests and union them.
    # Per-branch task count is exactly 2**level in this implementation.
    per_branch = 2 ** int(level)
    est = branches * per_branch + 4  # include a tiny overhead buffer
    if est > MAX_SUBGRAPH_TASKS:
        # Choose a safe group size; adjust downward until estimate fits.
        group = max(1, (MAX_SUBGRAPH_TASKS - 8) // per_branch)
        # Keep at least 1 branch per group.
        group = max(1, group)
        dst = BlueprintGraph()
        dst = _install_task_kwargs_injector(dst, task_complexity)
        remaining = branches
        g = 0
        while remaining > 0:
            b = min(remaining, group)
            part = make_binary_forest(branches=b, level=level, task_complexity=task_complexity)
            _clone_blueprint_graph_into(dst, part, prefix=f"BF{g}_")
            remaining -= b
            g += 1
        _assert_degree_constraints(dst)
        _assert_subgraph_size_constraints(dst)
        return dst

    bg = BlueprintGraph()
    bg = _install_task_kwargs_injector(bg, task_complexity)
    lib = graph_tasks.get_task_lib(task_complexity)

    finals = []

    for b in range(branches):
        leaves = [f"B{b}_L{i}" for i in range(2 ** (level - 1))]
        for leaf in leaves:
            bg.add_task(leaf, lib.const, 1)

        cur = leaves
        for d in range(level - 1):
            nxt = []
            for i in range(0, len(cur), 2):
                l, r = cur[i], cur[i + 1]
                p = f"B{b}_N{d}_{i//2}"
                bg.add_task(
                    p,
                    lib.addN,
                    TaskOutputRef(l),
                    TaskOutputRef(r),
                )
                nxt.append(p)
            cur = nxt

        f = f"B{b}_final"
        bg.add_task(f, lib.addN, TaskOutputRef(cur[0]))
        finals.append(f)

    # fan-in cap: branches may be huge; reduce with <=MAX_TASK_PARENTS arity
    merge_root = _add_kary_reduction(bg, lib, finals, root_key="merge", prefix="BF_M_", fanin=MAX_TASK_PARENTS)
    bg.add_task("output", lib.addN, TaskOutputRef(merge_root))

    _assert_degree_constraints(bg)
    _assert_subgraph_size_constraints(bg)
    return bg


def make_random_heavy_fanin_graph(sources=128, layers=4, min_fanin=6, max_fanin=20, unary_prob=0.2, seed=None, task_complexity: int = 1):
    import random
    rnd = random.Random(seed)
    bg = BlueprintGraph()
    bg = _install_task_kwargs_injector(bg, task_complexity)
    lib = graph_tasks.get_task_lib(task_complexity)

    pool = []
    for i in range(sources):
        k = f"S{i}"
        bg.add_task(k, lib.const, 1)
        pool.append(k)

    a = u = 0
    for L in range(layers):
        new_pool = []
        while len(pool) >= 2:
            k_in = min(rnd.randint(min_fanin, max_fanin), len(pool))
            if k_in < 2:
                break
            group = [pool.pop(rnd.randrange(len(pool))) for _ in range(k_in)]
            ak = f"A{L}_{a}"; a += 1
            bg.add_task(ak, lib.addN, *[TaskOutputRef(g) for g in group])
            new_pool.append(ak)
            if rnd.random() < unary_prob and pool:
                base = pool.pop(rnd.randrange(len(pool)))
                uk = f"U{L}_{u}"; u += 1
                bg.add_task(uk, lib.scale, TaskOutputRef(base), rnd.randint(2, 5))
                new_pool.append(uk)
        if pool:
            new_pool.append(pool.pop())
        pool = new_pool

    finals = pool
    if len(finals) >= 2:
        _add_kary_reduction(bg, lib, finals, root_key="final_merge", prefix="HFIN_", fanin=MAX_TASK_PARENTS)
    elif len(finals) == 1:
        # Keep addN's arity >= 2 to match the old behavior that padded with a 0.
        bg.add_task("final_merge", lib.addN, TaskOutputRef(finals[0]), 0)
    else:
        bg.add_task("final_merge", lib.addN, 0)

    bg.add_task("output", lib.addN, TaskOutputRef("final_merge"))
    _assert_degree_constraints(bg)
    _assert_subgraph_size_constraints(bg)
    return bg


def make_random_heavy_fanout_graph(roots=8, layers=3, min_fanout=5, max_fanout=20, binary_prob=0.3, seed=None, task_complexity: int = 1):
    import random
    rnd = random.Random(seed)
    bg = BlueprintGraph()
    bg = _install_task_kwargs_injector(bg, task_complexity)
    lib = graph_tasks.get_task_lib(task_complexity)

    current = []
    for i in range(roots):
        k = f"R{i}"
        bg.add_task(k, lib.const, 1)
        current.append(k)

    # Track children counts to hard-cap fan-out (esp. for random 'q' in binary edges).
    child_counts: dict[str, int] = defaultdict(int)

    c = m = 0
    for L in range(layers):
        next_layer = []
        for p in current:
            n = min(rnd.randint(min_fanout, max_fanout), MAX_TASK_CHILDREN)
            for _ in range(n):
                if rnd.random() < binary_prob and len(current) >= 2:
                    # Pick a q that won't exceed MAX_TASK_CHILDREN children if possible.
                    # If none available, fall back to unary edge on p.
                    q = None
                    # Avoid O(len(current)) scans: try a few random candidates.
                    for _try in range(12):
                        cand = current[rnd.randrange(len(current))]
                        if child_counts[cand] < MAX_TASK_CHILDREN:
                            q = cand
                            break
                    if q is None:
                        ck = f"C{L}_{c}"; c += 1
                        bg.add_task(ck, lib.scale, TaskOutputRef(p), rnd.randint(2, 5))
                        child_counts[p] += 1
                        next_layer.append(ck)
                        continue
                    ck = f"C{L}_{c}"; c += 1
                    bg.add_task(ck, lib.add2, TaskOutputRef(p), TaskOutputRef(q))
                    child_counts[p] += 1
                    child_counts[q] += 1
                    next_layer.append(ck)
                else:
                    ck = f"C{L}_{c}"; c += 1
                    bg.add_task(ck, lib.scale, TaskOutputRef(p), rnd.randint(2, 5))
                    child_counts[p] += 1
                    next_layer.append(ck)
        current = next_layer

    while len(current) > 1:
        nxt = []
        it = iter(current)
        for a in it:
            b = next(it, None)
            if b is None:
                nxt.append(a)
            else:
                mk = f"M{m}"; m += 1
                bg.add_task(mk, lib.add2, TaskOutputRef(a), TaskOutputRef(b))
                nxt.append(mk)
        current = nxt

    bg.add_task("output", lib.const, TaskOutputRef(current[0]))
    _assert_degree_constraints(bg)
    _assert_subgraph_size_constraints(bg)
    return bg


def make_chain_graph(branches: int = 1, chain_len: int = 1, task_complexity: int = 1):
    assert branches >= 1 and chain_len >= 1

    # Avoid a single huge component: split by grouping branches into multiple independent chains.
    # Rough estimate per branch: chain_len tasks + 1 merge/output overhead amortized.
    est = branches * chain_len + 4
    if est > MAX_SUBGRAPH_TASKS:
        per_branch = max(1, chain_len)
        group = max(1, (MAX_SUBGRAPH_TASKS - 8) // per_branch)
        dst = BlueprintGraph()
        dst = _install_task_kwargs_injector(dst, task_complexity)
        remaining = branches
        g = 0
        while remaining > 0:
            b = min(remaining, group)
            part = make_chain_graph(branches=b, chain_len=chain_len, task_complexity=task_complexity)
            _clone_blueprint_graph_into(dst, part, prefix=f"CH{g}_")
            remaining -= b
            g += 1
        _assert_degree_constraints(dst)
        _assert_subgraph_size_constraints(dst)
        return dst

    bg = BlueprintGraph()
    bg = _install_task_kwargs_injector(bg, task_complexity)
    lib = graph_tasks.get_task_lib(task_complexity)

    finals = []
    for b in range(branches):
        head = f"C{b}_N0"
        bg.add_task(head, lib.op, 1)
        prev = head
        for i in range(1, chain_len):
            node = f"C{b}_N{i}"
            bg.add_task(node, lib.op, TaskOutputRef(prev))
            prev = node
        finals.append(prev)

    if len(finals) == 1:
        bg.add_task("output", lib.op, TaskOutputRef(finals[0]))
    else:
        # Hard cap: merge may have huge fan-in when branches is large.
        merge_root = _add_kary_reduction(bg, lib, finals, root_key="merge", prefix="CHAIN_M_", fanin=MAX_TASK_PARENTS)
        bg.add_task("output", lib.op, TaskOutputRef(merge_root))

    _assert_degree_constraints(bg)
    _assert_subgraph_size_constraints(bg)
    return bg


def make_deep_high_fanin_dag(total_tasks: int = 1_000_000, fanin: int = 64, task_complexity: int = 1):
    # Design: a deep spine of aggregators; each step adds (fanin-1) leaves + 1 aggregator.
    assert total_tasks >= 4 and fanin >= 3
    fanin = min(int(fanin), MAX_TASK_PARENTS)
    bg = BlueprintGraph()
    bg = _install_task_kwargs_injector(bg, task_complexity)
    lib = graph_tasks.get_task_lib(task_complexity)

    # Seed so the first aggregator already has a predecessor
    seed = "agg_-1"
    bg.add_task(seed, lib.const, 0)
    prev = seed

    # total = 1(seed) + D*fanin + r(extras) + 1(output) = total_tasks
    D = (total_tasks - 2) // fanin
    r = (total_tasks - 2) - D * fanin  # 0 <= r < fanin

    for k in range(D):
        leaves = []
        for i in range(fanin - 1):
            leaf = f"s{k}_leaf{i}"
            bg.add_task(leaf, lib.const, 1)
            leaves.append(leaf)
        agg = f"agg_{k}"
        bg.add_task(agg, lib.addN, TaskOutputRef(prev), *[TaskOutputRef(l) for l in leaves])
        prev = agg  # deep chain

    # Pad remaining nodes into the final output's fan-in
    extras = []
    for i in range(r):
        e = f"tail_leaf{i}"
        bg.add_task(e, lib.const, 1)
        extras.append(e)

    if extras:
        bg.add_task("output", lib.addN, TaskOutputRef(prev), *[TaskOutputRef(e) for e in extras])
    else:
        bg.add_task("output", lib.addN, TaskOutputRef(prev))

    # Optional sanity check (enable if you want a hard assert)
    # assert len(task_dict) == total_tasks, (len(task_dict), total_tasks)

    return bg


def make_individuals(num_tasks: int = 1, task_complexity: int = 1):
    assert num_tasks >= 1

    # Default "individuals" shape adds a reduction ("merge") + "output" task, which can push
    # the connected component slightly above num_tasks. If that would exceed the subgraph cap,
    # switch to a grouped multi-component variant.
    if num_tasks > 1:
        internal = _reduction_internal_nodes_count(num_tasks, MAX_TASK_PARENTS)  # merge tree nodes (incl root)
        est_component = num_tasks + internal + 1  # +1 for "output"
    else:
        est_component = 1

    if est_component > MAX_SUBGRAPH_TASKS:
        bg = BlueprintGraph()
        bg = _install_task_kwargs_injector(bg, task_complexity)
        lib = graph_tasks.get_task_lib(task_complexity)

        # Create all leaf tasks (these are partitioned into groups below).
        leaf_keys = []
        for i in range(num_tasks):
            k = f"I{i}"
            bg.add_task(k, lib.const, 1)
            leaf_keys.append(k)

        # Pick maximum group size so: group + reduction_nodes <= MAX_SUBGRAPH_TASKS
        max_group = min(num_tasks, MAX_SUBGRAPH_TASKS)
        while max_group > 1:
            internal_g = _reduction_internal_nodes_count(max_group, MAX_TASK_PARENTS)
            if max_group + internal_g <= MAX_SUBGRAPH_TASKS:
                break
            max_group -= 1
        max_group = max(1, max_group)

        gid = 0
        for s in range(0, num_tasks, max_group):
            grp = leaf_keys[s : s + max_group]
            _add_kary_reduction(bg, lib, grp, root_key=f"merge_{gid}", prefix=f"IND_G{gid}_", fanin=MAX_TASK_PARENTS)
            gid += 1

        _assert_degree_constraints(bg)
        _assert_subgraph_size_constraints(bg)
        return bg

    bg = BlueprintGraph()
    bg = _install_task_kwargs_injector(bg, task_complexity)
    lib = graph_tasks.get_task_lib(task_complexity)

    for i in range(num_tasks):
        k = f"I{i}"
        bg.add_task(k, lib.const, 1)

    # Ensure every individual task is reachable from the target key "output".
    if num_tasks == 1:
        bg.add_task("output", lib.const, TaskOutputRef("I0"))
    else:
        _add_kary_reduction(bg, lib, [f"I{i}" for i in range(num_tasks)], root_key="merge", prefix="IND_M_", fanin=MAX_TASK_PARENTS)
        bg.add_task("output", lib.const, TaskOutputRef("merge"))

    _assert_degree_constraints(bg)
    _assert_subgraph_size_constraints(bg)
    return bg


def make_trivial_workflow(total_tasks=1000, task_complexity: int = 1):
    """
    Build a "trivial" workflow: `total_tasks` completely independent tasks.

    Contract:
    - The returned BlueprintGraph has exactly `total_tasks` tasks in bg.task_dict.
    - No task depends on any other task (fully disconnected).
    """
    assert total_tasks >= 1
    bg = BlueprintGraph()
    bg = _install_task_kwargs_injector(bg, task_complexity)
    lib = graph_tasks.get_task_lib(task_complexity)

    for i in range(total_tasks):
        bg.add_task(f"T{i}", lib.const, 1)

    # Strict task-count guarantee
    assert len(bg.task_dict) == total_tasks, (len(bg.task_dict), total_tasks)
    _assert_degree_constraints(bg)
    _assert_subgraph_size_constraints(bg)
    return bg


def make_stencil(total_tasks=1000, task_complexity: int = 1):
    assert total_tasks >= 1

    def _one(n, *, task_complexity: int):
        return make_stencil(total_tasks=n, task_complexity=task_complexity)

    if total_tasks > MAX_SUBGRAPH_TASKS:
        return _split_into_disjoint_subgraphs_by_total_tasks(
            _one, total_tasks, task_complexity=task_complexity, prefix_base="STEN"
        )

    bg = BlueprintGraph()
    bg = _install_task_kwargs_injector(bg, task_complexity)
    lib = graph_tasks.get_task_lib(task_complexity)

    width = int(math.ceil(math.sqrt(total_tasks)))

    def name(i):
        return f"S{i}"

    for i in range(total_tasks):
        r = i // width
        c = i % width

        deps = []
        if c > 0:
            deps.append(name(i - 1))              # left
        if r > 0:
            up = i - width
            if up < total_tasks:
                deps.append(name(up))             # up
            if c > 0:
                deps.append(name(up - 1))         # up-left

        if deps:
            bg.add_task(name(i), lib.addN, *[TaskOutputRef(d) for d in deps])
        else:
            bg.add_task(name(i), lib.const, 1)

    _assert_degree_constraints(bg)
    _assert_subgraph_size_constraints(bg)
    return bg


def make_fft(total_tasks=1000, task_complexity: int = 1):
    assert total_tasks >= 1

    def _one(n, *, task_complexity: int):
        return make_fft(total_tasks=n, task_complexity=task_complexity)

    if total_tasks > MAX_SUBGRAPH_TASKS:
        return _split_into_disjoint_subgraphs_by_total_tasks(
            _one, total_tasks, task_complexity=task_complexity, prefix_base="FFT"
        )

    bg = BlueprintGraph()
    bg = _install_task_kwargs_injector(bg, task_complexity)
    lib = graph_tasks.get_task_lib(task_complexity)

    # choose W = power of two lanes, and stages = log2(W)
    # each stage creates W tasks, plus input layer W tasks => W*(stages+1)
    # keep W as large as possible without exceeding total_tasks
    W = 1
    while True:
        nxt = W * 2
        stages = int(math.log2(nxt))
        core = nxt * (stages + 1)
        if core <= total_tasks:
            W = nxt
        else:
            break

    stages = int(math.log2(W))
    core_tasks = W * (stages + 1)

    def name(s, i):
        return f"F{s}_{i}"

    # input layer
    for i in range(W):
        bg.add_task(name(0, i), lib.const, 1)

    # butterfly stages: pair (i, i^bit) produces two outputs at this stage
    # We model it as two tasks, but still keep total tasks per stage = W.
    for s in range(1, stages + 1):
        bit = 1 << (s - 1)
        for i in range(W):
            p = i ^ bit
            a = name(s - 1, i)
            b = name(s - 1, p)
            bg.add_task(name(s, i), lib.addN, TaskOutputRef(a), TaskOutputRef(b))

    # pad as extra lanes on the last stage (keep layered feel)
    rem = total_tasks - core_tasks
    for j in range(rem):
        bg.add_task(f"Fpad_{j}", lib.addN, TaskOutputRef(name(stages, j % W)))

    _assert_degree_constraints(bg)
    _assert_subgraph_size_constraints(bg)
    return bg


def make_sweep(total_tasks=1000, task_complexity: int = 1):
    assert total_tasks >= 1

    def _one(n, *, task_complexity: int):
        return make_sweep(total_tasks=n, task_complexity=task_complexity)

    if total_tasks > MAX_SUBGRAPH_TASKS:
        return _split_into_disjoint_subgraphs_by_total_tasks(
            _one, total_tasks, task_complexity=task_complexity, prefix_base="SWP"
        )

    bg = BlueprintGraph()
    bg = _install_task_kwargs_injector(bg, task_complexity)
    lib = graph_tasks.get_task_lib(task_complexity)

    width = int(math.ceil(math.sqrt(total_tasks)))
    height = int(math.ceil(total_tasks / width))

    def idx(r, c):
        return r * width + c

    def name(r, c):
        return f"W{r}_{c}"

    for r in range(height):
        for c in range(width):
            i = idx(r, c)
            if i >= total_tasks:
                return bg

            deps = []
            if r > 0:
                deps.append(name(r - 1, c))          # vertical
            if r > 0 and c > 0:
                deps.append(name(r - 1, c - 1))      # diagonal (wavefront)

            if deps:
                bg.add_task(name(r, c), lib.addN, *[TaskOutputRef(d) for d in deps])
            else:
                bg.add_task(name(r, c), lib.const, 1)

    _assert_degree_constraints(bg)
    _assert_subgraph_size_constraints(bg)
    return bg


def make_binary_tree(total_tasks=1000, task_complexity: int = 1):
    """
    Build a single binary-tree-shaped DAG with exactly `total_tasks` tasks.

    - Single parameter: total_tasks
    - Strict task-count guarantee: len(bg.task_dict) == total_tasks
    - No special "output" task is added.

    Implementation: a complete binary tree in array layout:
      node i has children (2*i+1, 2*i+2) if in range.
    Leaves are const tasks; internal nodes aggregate children.
    Root key is "BT0".
    """
    assert total_tasks >= 1

    def _one(n, *, task_complexity: int):
        return make_binary_tree(total_tasks=n, task_complexity=task_complexity)

    if total_tasks > MAX_SUBGRAPH_TASKS:
        return _split_into_disjoint_subgraphs_by_total_tasks(
            _one, total_tasks, task_complexity=task_complexity, prefix_base="BT"
        )

    bg = BlueprintGraph()
    bg = _install_task_kwargs_injector(bg, task_complexity)
    lib = graph_tasks.get_task_lib(task_complexity)

    # Leaves are indices > last_internal
    last_internal = (total_tasks - 2) // 2  # -1 when total_tasks == 1

    # Create leaves first
    for i in range(last_internal + 1, total_tasks):
        bg.add_task(f"BT{i}", lib.const, 1)

    # Create internal nodes bottom-up so all deps exist
    for i in range(last_internal, -1, -1):
        left = 2 * i + 1
        right = 2 * i + 2
        deps = [TaskOutputRef(f"BT{left}")]
        if right < total_tasks:
            deps.append(TaskOutputRef(f"BT{right}"))
        bg.add_task(f"BT{i}", lib.addN, *deps)

    assert "output" not in bg.task_dict
    assert len(bg.task_dict) == total_tasks, (len(bg.task_dict), total_tasks)
    _assert_degree_constraints(bg)
    _assert_subgraph_size_constraints(bg)
    return bg


def make_random(total_tasks=1000, task_complexity: int = 1):
    """
    Build a fully random DAG with exactly `total_tasks` tasks.

    - Single parameter: total_tasks
    - Strict task-count guarantee: len(bg.task_dict) == total_tasks
    - No special "output" task is added.

    Randomness:
    - Each task i randomly chooses a (small) random set of predecessors from [0, i).
    - If it chooses no predecessors, it's a source const task with a random value.
    - Otherwise it's an aggregation task over its chosen predecessors.
    """
    assert total_tasks >= 1

    def _one(n, *, task_complexity: int):
        return make_random(total_tasks=n, task_complexity=task_complexity)

    if total_tasks > MAX_SUBGRAPH_TASKS:
        return _split_into_disjoint_subgraphs_by_total_tasks(
            _one, total_tasks, task_complexity=task_complexity, prefix_base="RND"
        )

    bg = BlueprintGraph()
    bg = _install_task_kwargs_injector(bg, task_complexity)
    lib = graph_tasks.get_task_lib(task_complexity)

    # Per-call random seed from OS entropy (not reproducible by design).
    seed = int.from_bytes(os.urandom(8), "little")
    rnd = random.Random(seed)

    # Keep fan-in bounded for scalability while staying "random".
    MAX_FANIN = min(32, MAX_TASK_PARENTS)

    # Hard-cap fan-out by tracking how many children each node already has.
    child_counts: dict[int, int] = defaultdict(int)

    def name(i):
        return f"R{i}"

    for i in range(total_tasks):
        if i == 0:
            # allow multiple subgraphs overall; keep the first node a source
            bg.add_task(name(i), lib.const, 1)
            continue

        # Allow multiple subgraphs: some nodes may have 0 parents.
        # Also enforce <= MAX_TASK_CHILDREN children per parent.
        eligible = [p for p in range(i) if child_counts[p] < MAX_TASK_CHILDREN]
        if not eligible or rnd.random() < 0.10:
            bg.add_task(name(i), lib.const, 1)
            continue

        k = rnd.randint(1, min(len(eligible), MAX_FANIN))
        parents = rnd.sample(eligible, k)
        bg.add_task(name(i), lib.addN, *[TaskOutputRef(name(p)) for p in parents])
        for p in parents:
            child_counts[p] += 1

    assert "output" not in bg.task_dict
    assert len(bg.task_dict) == total_tasks, (len(bg.task_dict), total_tasks)
    _assert_degree_constraints(bg)
    _assert_subgraph_size_constraints(bg)
    return bg


def make_blast(total_tasks=1000, task_complexity: int = 1):
    """
    BLAST-like workflow shape:
      sources -> many parallel "search" tasks -> multi-level reductions -> (optional) postprocess
    Contract:
      - len(bg.task_dict) == total_tasks
      - No special "output" task required; root is the last node created.
    """
    assert total_tasks >= 1

    def _one(n, *, task_complexity: int):
        return make_blast(total_tasks=n, task_complexity=task_complexity)

    if total_tasks > MAX_SUBGRAPH_TASKS:
        return _split_into_disjoint_subgraphs_by_total_tasks(
            _one, total_tasks, task_complexity=task_complexity, prefix_base="BLAST"
        )

    bg = BlueprintGraph()
    bg = _install_task_kwargs_injector(bg, task_complexity)
    lib = graph_tasks.get_task_lib(task_complexity)

    def name(prefix, i):
        return f"{prefix}{i}"

    # Minimal cases
    if total_tasks == 1:
        bg.add_task("B0", lib.const, 1)
        return bg

    # 1) Choose counts: a few sources, many searches, then reductions, then maybe post
    # Choose sources so per-source fan-out to searches is <= MAX_TASK_CHILDREN.
    # (search tasks are assigned round-robin to sources.)
    sources = min(MAX_TASK_CHILDREN, max(1, total_tasks // 100))
    # Allocate a large chunk to search stage
    search_budget = total_tasks - sources
    if search_budget < 1:
        # total_tasks == sources, all sources
        for i in range(total_tasks):
            bg.add_task(name("S", i), lib.const, 1)
        return bg

    # We'll reserve some tasks for reductions + post; estimate later, then adjust.
    # Start with an aggressive search count then back off to fit reductions.
    searches = max(1, int(search_budget * 0.80))

    # Ensure source fan-out cap.
    sources_needed = min(MAX_TASK_CHILDREN, max(1, (searches + MAX_TASK_CHILDREN - 1) // MAX_TASK_CHILDREN))
    if sources < sources_needed:
        sources = sources_needed
        search_budget = total_tasks - sources
        searches = max(1, int(search_budget * 0.80))

    # helper: build k-ary reduction over a list of keys, returns list of next keys and task_count used
    def reduce_kary(keys, fanin, level, counter_start):
        c = counter_start
        out = []
        i = 0
        while i < len(keys):
            grp = keys[i:i+fanin]
            if len(grp) == 1:
                out.append(grp[0])
            else:
                rk = f"R{level}_{c}"; c += 1
                bg.add_task(rk, lib.addN, *[TaskOutputRef(x) for x in grp])
                out.append(rk)
            i += fanin
        return out, c

    # 2) Try to fit: we may need several reduction levels + a couple of post tasks
    # We'll iterate a few times to find a feasible searches count.
    for _ in range(8):
        bg2 = BlueprintGraph()
        bg2 = _install_task_kwargs_injector(bg2, task_complexity)

        # Create sources
        src_keys = []
        for i in range(sources):
            k = name("SRC", i)
            bg2.add_task(k, lib.const, 1)
            src_keys.append(k)

        # Create searches: each depends on one source (round-robin)
        search_keys = []
        for i in range(searches):
            sk = name("Q", i)
            dep = src_keys[i % sources]
            bg2.add_task(sk, lib.scale, TaskOutputRef(dep), 2)
            search_keys.append(sk)

        # Reductions (fanin 16 typical-ish)
        fanin = 16
        cur = search_keys
        level = 0
        c = 0
        while len(cur) > 1:
            nxt = []
            i = 0
            while i < len(cur):
                grp = cur[i:i+fanin]
                if len(grp) == 1:
                    nxt.append(grp[0])
                else:
                    rk = f"RED{level}_{c}"; c += 1
                    bg2.add_task(rk, lib.addN, *[TaskOutputRef(x) for x in grp])
                    nxt.append(rk)
                i += fanin
            cur = nxt
            level += 1

        root = cur[0]

        # Optional postprocess chain (2 tasks)
        bg2.add_task("POST0", lib.scale, TaskOutputRef(root), 3)
        bg2.add_task("POST1", lib.addN, TaskOutputRef("POST0"), 1)

        used = len(bg2.task_dict)
        if used <= total_tasks:
            # Accept this plan
            bg = bg2
            break

        # Too many reductions/post relative to searches, reduce searches
        # Back off aggressively
        searches = max(1, int(searches * 0.85))
    else:
        # Fallback: simple chain if somehow can't fit (shouldn't happen)
        bg = BlueprintGraph()
        bg = _install_task_kwargs_injector(bg, task_complexity)
        bg.add_task("B0", lib.const, 1)
        for i in range(1, total_tasks):
            bg.add_task(f"B{i}", lib.addN, TaskOutputRef(f"B{i-1}"), 1)
        return bg

    # 3) Pad remaining tasks, keep them connected to the root-ish node to avoid disconnected junk
    used = len(bg.task_dict)
    rem = total_tasks - used
    anchor = "POST1" if "POST1" in bg.task_dict else list(bg.task_dict.keys())[-1]
    for i in range(rem):
        bg.add_task(f"PAD{i}", lib.addN, TaskOutputRef(anchor), 1)
        anchor = f"PAD{i}"

    assert len(bg.task_dict) == total_tasks, (len(bg.task_dict), total_tasks)
    _assert_degree_constraints(bg)
    _assert_subgraph_size_constraints(bg)
    return bg


def make_sipht(total_tasks=1000, seed_genomes=None, task_complexity: int = 1):
    assert total_tasks >= 1

    def _one(n, *, task_complexity: int):
        return make_sipht(total_tasks=n, seed_genomes=seed_genomes, task_complexity=task_complexity)

    if total_tasks > MAX_SUBGRAPH_TASKS:
        return _split_into_disjoint_subgraphs_by_total_tasks(
            _one, total_tasks, task_complexity=task_complexity, prefix_base="SIPHT"
        )

    bg = BlueprintGraph()
    bg = _install_task_kwargs_injector(bg, task_complexity)
    lib = graph_tasks.get_task_lib(task_complexity)

    if total_tasks == 1:
        bg.add_task("SIPHT0", lib.const, 1)
        return bg

    G0 = seed_genomes if seed_genomes is not None else min(32, max(1, total_tasks // 200))
    # Cap fan-out from PRE{g} -> CAND{g}_* by limiting K.
    K0 = min(MAX_TASK_CHILDREN, max(1, (total_tasks // max(1, G0) - 12) // 4))

    def kary_reduce(keys, fanin, prefix):
        if not keys:
            return None
        if len(keys) == 1:
            return keys[0]
        cur = list(keys)
        lvl = 0
        idx = 0
        while len(cur) > 1:
            nxt = []
            for i in range(0, len(cur), fanin):
                grp = cur[i:i+fanin]
                if len(grp) == 1:
                    nxt.append(grp[0])
                else:
                    rk = f"{prefix}{lvl}_{idx}"
                    idx += 1
                    bg.add_task(rk, lib.addN, *[TaskOutputRef(x) for x in grp])
                    nxt.append(rk)
            cur = nxt
            lvl += 1
        return cur[0]

    for _ in range(10):
        bg = BlueprintGraph()
        bg = _install_task_kwargs_injector(bg, task_complexity)

        def kary_reduce2(keys, fanin, prefix):
            if not keys:
                return None
            if len(keys) == 1:
                return keys[0]
            cur = list(keys)
            lvl = 0
            idx = 0
            while len(cur) > 1:
                nxt = []
                for i in range(0, len(cur), fanin):
                    grp = cur[i:i+fanin]
                    if len(grp) == 1:
                        nxt.append(grp[0])
                    else:
                        rk = f"{prefix}{lvl}_{idx}"
                        idx += 1
                        bg.add_task(rk, lib.addN, *[TaskOutputRef(x) for x in grp])
                        nxt.append(rk)
                cur = nxt
                lvl += 1
            return cur[0]

        G = G0
        K = K0

        genome_roots = []
        for g in range(G):
            pre = f"PRE{g}"
            bg.add_task(pre, lib.const, 1)

            candidates = []
            for j in range(K):
                ck = f"CAND{g}_{j}"
                bg.add_task(ck, lib.scale, TaskOutputRef(pre), 2)
                candidates.append(ck)

            filtered = []
            for j in range(K):
                fk = f"FILT{g}_{j}"
                bg.add_task(fk, lib.addN, TaskOutputRef(candidates[j]), 1)
                filtered.append(fk)

            scored = []
            for j in range(K):
                sk = f"SCORE{g}_{j}"
                bg.add_task(sk, lib.scale, TaskOutputRef(filtered[j]), 3)
                scored.append(sk)

            root = kary_reduce2(scored, 16, f"R{g}_")
            if root is None:
                root = pre

            postg = f"POSTG{g}"
            bg.add_task(postg, lib.addN, TaskOutputRef(root), 1)
            genome_roots.append(postg)

        global_root = kary_reduce2(genome_roots, 16, "G_")
        if global_root is None:
            global_root = genome_roots[0]

        bg.add_task("FINAL0", lib.scale, TaskOutputRef(global_root), 5)
        bg.add_task("FINAL1", lib.addN, TaskOutputRef("FINAL0"), 1)

        used = len(bg.task_dict)
        if used <= total_tasks:
            break

        K0 = max(1, int(K0 * 0.8))
    else:
        bg = BlueprintGraph()
        bg = _install_task_kwargs_injector(bg, task_complexity)
        bg.add_task("SIPHT0", lib.const, 1)
        for i in range(1, total_tasks):
            bg.add_task(f"SIPHT{i}", lib.addN, TaskOutputRef(f"SIPHT{i-1}"), 1)
        return bg

    used = len(bg.task_dict)
    rem = total_tasks - used
    anchor = "FINAL1" if "FINAL1" in bg.task_dict else list(bg.task_dict.keys())[-1]
    for i in range(rem):
        pk = f"PAD{i}"
        bg.add_task(pk, lib.addN, TaskOutputRef(anchor), 1)
        anchor = pk

    assert len(bg.task_dict) == total_tasks, (len(bg.task_dict), total_tasks)
    _assert_degree_constraints(bg)
    _assert_subgraph_size_constraints(bg)
    return bg


def make_synthetic_scec(total_tasks=1000, task_complexity: int = 1):
    """
    Synthetic SCEC-like (CyberShake-ish) shape:
      2 sources -> blue stage -> orange stage -> (red join) -> (green final)
    Contract: len(bg.task_dict) == total_tasks
    """
    assert total_tasks >= 1

    def _one(n, *, task_complexity: int):
        return make_synthetic_scec(total_tasks=n, task_complexity=task_complexity)

    if total_tasks > MAX_SUBGRAPH_TASKS:
        return _split_into_disjoint_subgraphs_by_total_tasks(
            _one, total_tasks, task_complexity=task_complexity, prefix_base="SCEC"
        )

    bg = BlueprintGraph()
    bg = _install_task_kwargs_injector(bg, task_complexity)
    lib = graph_tasks.get_task_lib(task_complexity)

    if total_tasks == 1:
        bg.add_task("G0", lib.const, 1)
        return bg

    # Pick layer sizes to mimic the picture while enforcing hard degree caps:
    # - Top -> Blue fan-out <= MAX_TASK_CHILDREN
    # - Blue -> Orange fan-out (at least 1 edge via modulo) <= MAX_TASK_CHILDREN
    # - Orange join fan-in is reduced with <= MAX_TASK_PARENTS arity (budgeted inside core)
    tail = 2 if total_tasks >= 4 else 1  # (R0 + G0) or just (G0)

    # Initial guess
    top = 2 if total_tasks >= 3 else 1
    blue = 1
    orange = 1
    extra_red = 0

    for _ in range(16):
        core = total_tasks - top - tail
        if core <= 0:
            break

        blue = max(1, int(core * 0.45))
        orange = max(1, core - blue)

        # Enforce Blue -> Orange child cap: orange tasks each depend on blue_keys[i % blue],
        # so each blue gets about orange/blue children via that mandatory edge.
        need_blue = max(1, (orange + MAX_TASK_CHILDREN - 1) // MAX_TASK_CHILDREN)
        if blue < need_blue:
            # steal from orange if needed
            delta = min(need_blue - blue, max(0, orange - 1))
            blue += delta
            orange -= delta

        # Enforce Top -> Blue child cap: distribute blue over `top` sources.
        top_need = max(1, (blue + MAX_TASK_CHILDREN - 1) // MAX_TASK_CHILDREN)
        if top < top_need:
            top = min(MAX_TASK_CHILDREN, top_need)
            continue  # recompute core sizes

        # Budget extra reduction nodes for Orange join so root join has <= MAX_TASK_PARENTS parents.
        # Root join exists (R0 when tail==2, else G0), so we need internal_nodes-1 extra tasks.
        internal = _reduction_internal_nodes_count(orange, MAX_TASK_PARENTS)
        extra_red = max(0, internal - 1)

        used = top + blue + orange + extra_red + tail
        if used > total_tasks:
            # Shrink orange first (it dominates), then recompute.
            shrink = used - total_tasks
            orange = max(1, orange - shrink)
            continue
        break

    core = total_tasks - top - tail
    if core <= 0:
        for i in range(total_tasks):
            bg.add_task(f"N{i}", lib.const, 1)
        return bg

    # 1) Top (yellow)
    top_keys = []
    for i in range(top):
        k = f"Y{i}"
        bg.add_task(k, lib.const, 1)
        top_keys.append(k)

    # 2) Blue (fan-out from top)
    blue_keys = []
    for i in range(blue):
        p = top_keys[i % len(top_keys)]
        k = f"B{i}"
        bg.add_task(k, lib.scale, TaskOutputRef(p), 2)
        blue_keys.append(k)

    # 3) Orange (each orange depends on 1..3 blue nodes)
    orange_keys = []
    for i in range(orange):
        deps = []
        deps.append(blue_keys[i % len(blue_keys)])
        if len(blue_keys) >= 2 and (i % 3 != 0):
            deps.append(blue_keys[(i * 7 + 1) % len(blue_keys)])
        if len(blue_keys) >= 3 and (i % 5 == 0):
            deps.append(blue_keys[(i * 11 + 2) % len(blue_keys)])
        k = f"O{i}"
        bg.add_task(k, lib.addN, *[TaskOutputRef(d) for d in deps])
        orange_keys.append(k)

    # 4) Red join (reduced to satisfy fan-in cap)
    if tail == 2:
        # Create extra reduction nodes if needed, but keep total_tasks exact by having
        # allocated `extra_red` budget above (pads below will absorb any rounding deltas).
        root = _add_kary_reduction(bg, lib, orange_keys, root_key="R0", prefix="SCEC_R_", fanin=MAX_TASK_PARENTS)
        # 5) Green final
        bg.add_task("G0", lib.addN, TaskOutputRef(root), 1)
        anchor = "G0"
    else:
        # total_tasks == 2 or 3 case
        _add_kary_reduction(bg, lib, orange_keys, root_key="G0", prefix="SCEC_G_", fanin=MAX_TASK_PARENTS)
        anchor = "G0"

    # Pad if any rounding caused fewer tasks than requested (should not, but keep it strict).
    rem = total_tasks - len(bg.task_dict)
    for i in range(rem):
        pk = f"PAD{i}"
        bg.add_task(pk, lib.addN, TaskOutputRef(anchor), 1)
        anchor = pk

    assert len(bg.task_dict) == total_tasks, (len(bg.task_dict), total_tasks)
    _assert_degree_constraints(bg)
    _assert_subgraph_size_constraints(bg)
    return bg


def make_epigenomics(total_tasks=1000, task_complexity: int = 1):
    """
    Epigenomics-like layered pipeline (per the common Pegasus diagram):
      seqSplit (1)
        -> filterContams (L1, parallel)
        -> sol2sanger   (L2, parallel)
        -> fastq2bfq    (L3, parallel)
        -> maq          (L4, parallel)
        -> mapMerge     (1)
        -> maqIndex     (1)
        -> pileup       (1)

    Contract: len(bg.task_dict) == total_tasks
    """
    assert total_tasks >= 1

    def _one(n, *, task_complexity: int):
        return make_epigenomics(total_tasks=n, task_complexity=task_complexity)

    if total_tasks > MAX_SUBGRAPH_TASKS:
        return _split_into_disjoint_subgraphs_by_total_tasks(
            _one, total_tasks, task_complexity=task_complexity, prefix_base="EPI"
        )

    bg = BlueprintGraph()
    bg = _install_task_kwargs_injector(bg, task_complexity)
    lib = graph_tasks.get_task_lib(task_complexity)

    if total_tasks == 1:
        bg.add_task("seqSplit", lib.const, 1)
        return bg

    # Fixed singletons: seqSplit, mapMerge, maqIndex, pileup
    fixed = 4
    core_budget = total_tasks - fixed
    if core_budget <= 0:
        for i in range(total_tasks):
            bg.add_task(f"E{i}", lib.const, 1)
        return bg

    # Allocate widths for L1..L4 while enforcing:
    # - seqSplit fan-out <= MAX_TASK_CHILDREN via seqSplit_shards
    # - mapMerge fan-in <= MAX_TASK_PARENTS via reduction nodes (budgeted)
    # - wrap-around pipeline fan-outs <= MAX_TASK_CHILDREN by keeping widths roughly balanced
    w1 = max(1, core_budget // 4)
    w2 = max(1, core_budget // 4)
    w3 = max(1, core_budget // 4)
    w4 = max(1, core_budget - (w1 + w2 + w3))

    for _ in range(32):
        shards = (w1 + MAX_TASK_CHILDREN - 1) // MAX_TASK_CHILDREN
        internal = _reduction_internal_nodes_count(w4, MAX_TASK_PARENTS)
        extra_merge = max(0, internal - 1)  # mapMerge is the root

        used = w1 + w2 + w3 + w4 + shards + extra_merge
        if used <= core_budget:
            break

        # Shrink stages to fit budget, favor shrinking w4 then w3 then w2 then w1.
        shrink = used - core_budget
        if shrink > 0 and w4 > 1:
            dec = min(shrink, w4 - 1)
            w4 -= dec
            shrink -= dec
        if shrink > 0 and w3 > 1:
            dec = min(shrink, w3 - 1)
            w3 -= dec
            shrink -= dec
        if shrink > 0 and w2 > 1:
            dec = min(shrink, w2 - 1)
            w2 -= dec
            shrink -= dec
        if shrink > 0 and w1 > 1:
            dec = min(shrink, w1 - 1)
            w1 -= dec
            shrink -= dec

        # Keep widths balanced to prevent huge fan-out in wrap-around mapping.
        w2 = min(w2, w1 * MAX_TASK_CHILDREN)
        w3 = min(w3, w2 * MAX_TASK_CHILDREN)
        w4 = min(w4, w3 * MAX_TASK_CHILDREN)

    # seqSplit
    bg.add_task("seqSplit", lib.const, 1)

    # L1 filterContams: fan-out from seqSplit
    L1 = []
    shards = (w1 + MAX_TASK_CHILDREN - 1) // MAX_TASK_CHILDREN
    shard_keys = []
    for s in range(shards):
        sk = f"seqSplit_shard_{s}"
        # identity-ish task to fan-out through shards
        bg.add_task(sk, lib.addN, TaskOutputRef("seqSplit"), 0)
        shard_keys.append(sk)
    for i in range(w1):
        k = f"filterContams_{i}"
        dep = shard_keys[i % len(shard_keys)]
        bg.add_task(k, lib.scale, TaskOutputRef(dep), 2)
        L1.append(k)

    # L2 sol2sanger: 1-to-1 pipeline from L1 (wrap-around if widths differ)
    L2 = []
    for i in range(w2):
        dep = L1[i % len(L1)]
        k = f"sol2sanger_{i}"
        bg.add_task(k, lib.scale, TaskOutputRef(dep), 3)
        L2.append(k)

    # L3 fastq2bfq: 1-to-1 pipeline from L2
    L3 = []
    for i in range(w3):
        dep = L2[i % len(L2)]
        k = f"fastq2bfq_{i}"
        bg.add_task(k, lib.addN, TaskOutputRef(dep), 1)
        L3.append(k)

    # L4 maq: 1-to-1 pipeline from L3
    L4 = []
    for i in range(w4):
        dep = L3[i % len(L3)]
        k = f"maq_{i}"
        bg.add_task(k, lib.scale, TaskOutputRef(dep), 5)
        L4.append(k)

    # mapMerge: merge all maq results (single fan-in)
    # If L4 is huge, reduce in <= MAX_TASK_PARENTS chunks; mapMerge is the root join.
    _add_kary_reduction(bg, lib, L4, root_key="mapMerge", prefix="EPI_M_", fanin=MAX_TASK_PARENTS)

    # maqIndex: depends on mapMerge
    bg.add_task("maqIndex", lib.scale, TaskOutputRef("mapMerge"), 2)

    # pileup: depends on maqIndex
    bg.add_task("pileup", lib.addN, TaskOutputRef("maqIndex"), 1)

    # Pad remaining tasks (if any), keep connected to pileup
    rem = total_tasks - len(bg.task_dict)
    anchor = "pileup"
    for i in range(rem):
        pk = f"PAD{i}"
        bg.add_task(pk, lib.addN, TaskOutputRef(anchor), 1)
        anchor = pk

    assert len(bg.task_dict) == total_tasks, (len(bg.task_dict), total_tasks)
    _assert_degree_constraints(bg)
    _assert_subgraph_size_constraints(bg)
    return bg


def make_ligo(total_tasks=1000, task_complexity: int = 1):
    """
    LIGO-like (as in common Pegasus gallery diagram):
      TempBank (many) + Inspiral (many) -> Thinca (join)
      Thinca -> TrigBank (many) -> Inspiral2 (many) -> Thinca2 (join)
      (repeat-ish; we model 2 blocks like the picture)

    Contract: len(bg.task_dict) >= total_tasks
    """
    assert total_tasks >= 1

    def _one(n, *, task_complexity: int):
        return make_ligo(total_tasks=n, task_complexity=task_complexity)

    if total_tasks > MAX_SUBGRAPH_TASKS:
        return _split_into_disjoint_subgraphs_by_total_tasks(
            _one, total_tasks, task_complexity=task_complexity, prefix_base="LIGO"
        )

    bg = BlueprintGraph()
    bg = _install_task_kwargs_injector(bg, task_complexity)
    lib = graph_tasks.get_task_lib(task_complexity)

    if total_tasks == 1:
        bg.add_task("L0", lib.const, 1)
        return bg

    # We model 2 "coincidence" blocks to match the figure.
    blocks = 2

    # Each block: TB + IN -> TH (join) -> TR (many) -> IN2 (many) -> TH2 (join)
    # That's (tb + in + 1 + tr + in2 + 1) tasks per block.
    # We'll pick widths and then pad.
    fixed_per_block = 2  # the two joins
    # pick a reasonable base width
    base = max(1, (total_tasks - blocks * fixed_per_block) // (blocks * 4))  # split among 4 wide stages
    tb_w = max(1, base)
    in_w = max(1, base)
    tr_w = max(1, base)
    in2_w = max(1, base)

    # If we're too big, shrink widths quickly.
    # Account for extra proxy tasks used to keep TH -> TR fan-out <= MAX_TASK_CHILDREN.
    def est(tb, inn, tr, in2):
        proxies = (tr + MAX_TASK_CHILDREN - 1) // MAX_TASK_CHILDREN
        return blocks * (tb + inn + 1 + proxies + tr + in2 + 1)

    while est(tb_w, in_w, tr_w, in2_w) > total_tasks and (tb_w > 1 or in_w > 1 or tr_w > 1 or in2_w > 1):
        tb_w = max(1, int(tb_w * 0.8))
        in_w  = max(1, int(in_w * 0.8))
        tr_w  = max(1, int(tr_w * 0.8))
        in2_w = max(1, int(in2_w * 0.8))

    anchor = None

    for b in range(blocks):
        # TempBank (yellow)
        tb = []
        for i in range(tb_w):
            k = f"TB{b}_{i}"
            bg.add_task(k, lib.const, 1)
            tb.append(k)

        # Inspiral (blue) fan-out from TempBank (round-robin)
        ins = []
        for i in range(in_w):
            dep = tb[i % len(tb)]
            k = f"IN{b}_{i}"
            bg.add_task(k, lib.scale, TaskOutputRef(dep), 2)
            ins.append(k)

        # Thinca (orange) join of some inspirals (and a couple tempbanks to mimic edges)
        deps = []
        deps.extend(ins)
        # keep deps size bounded-ish if huge
        if len(deps) > 64:
            deps = deps[::max(1, len(deps)//64)]
        # sprinkle a few TB deps for structure richness
        if tb:
            deps.append(tb[0])
        if len(tb) > 1:
            deps.append(tb[len(tb)//2])
        th = f"TH{b}"
        bg.add_task(th, lib.addN, *[TaskOutputRef(x) for x in deps])

        # TrigBank (yellow/orange in figure) fan-out from Thinca
        # Hard cap: TH -> TR fan-out via proxies so each node has <= MAX_TASK_CHILDREN children.
        proxies = (tr_w + MAX_TASK_CHILDREN - 1) // MAX_TASK_CHILDREN
        th_proxies = []
        if proxies <= 1:
            th_proxies = [th]
        else:
            for i in range(proxies):
                pk = f"TH{b}_proxy_{i}"
                bg.add_task(pk, lib.addN, TaskOutputRef(th), 0)
                th_proxies.append(pk)
        tr = []
        for i in range(tr_w):
            k = f"TR{b}_{i}"
            dep = th_proxies[i % len(th_proxies)]
            bg.add_task(k, lib.scale, TaskOutputRef(dep), 3)
            tr.append(k)

        # Inspiral2 (blue) fan-out from TrigBank
        ins2 = []
        for i in range(in2_w):
            dep = tr[i % len(tr)]
            k = f"IN2{b}_{i}"
            bg.add_task(k, lib.scale, TaskOutputRef(dep), 2)
            ins2.append(k)

        # Thinca2 join
        deps2 = ins2
        if len(deps2) > 64:
            deps2 = deps2[::max(1, len(deps2)//64)]
        th2 = f"TH2{b}"
        bg.add_task(th2, lib.addN, *[TaskOutputRef(x) for x in deps2])

        anchor = th2

    # Pad remaining tasks, keep connected to last join
    rem = total_tasks - len(bg.task_dict)
    if anchor is None:
        anchor = list(bg.task_dict.keys())[-1]
    for i in range(rem):
        pk = f"PAD{i}"
        bg.add_task(pk, lib.addN, TaskOutputRef(anchor), 1)
        anchor = pk

    # For small total_tasks, the 2-block structure has a minimum size; allow overshoot.
    assert len(bg.task_dict) >= total_tasks, (len(bg.task_dict), total_tasks)
    _assert_degree_constraints(bg)
    _assert_subgraph_size_constraints(bg)
    return bg


def make_montage(total_tasks=1000, task_complexity: int = 1):
    """
    Montage-like structure (Pegasus-style):
      mProjectPP (many) -> mDiffFit (many) -> mConcatFit (1) -> mBgModel (1)
        -> mBackground (many) -> mImgTbl (1) -> mAdd (1) -> mViewer (1)

    Contract: len(bg.task_dict) == total_tasks
    """
    assert total_tasks >= 1

    def _one(n, *, task_complexity: int):
        return make_montage(total_tasks=n, task_complexity=task_complexity)

    if total_tasks > MAX_SUBGRAPH_TASKS:
        return _split_into_disjoint_subgraphs_by_total_tasks(
            _one, total_tasks, task_complexity=task_complexity, prefix_base="MON"
        )

    bg = BlueprintGraph()
    bg = _install_task_kwargs_injector(bg, task_complexity)
    lib = graph_tasks.get_task_lib(task_complexity)

    if total_tasks == 1:
        bg.add_task("mViewer", lib.const, 1)
        return bg

    tail = 4  # mImgTbl + mAdd + mViewer + (optional extra anchor) but we'll use exactly these 3 + 1 join
    fixed = 4  # mConcatFit, mBgModel, mImgTbl, mAdd, mViewer -> actually 5, but keep mAdd/mViewer in tail handling
    # We'll explicitly create: mConcatFit, mBgModel, mImgTbl, mAdd, mViewer = 5 singletons
    singletons = ["mConcatFit", "mBgModel", "mImgTbl", "mAdd", "mViewer"]
    fixed = len(singletons)

    core = total_tasks - fixed
    if core <= 0:
        for i in range(total_tasks):
            bg.add_task(f"M{i}", lib.const, 1)
        return bg

    # Allocate parallel stages while enforcing:
    # - mBgModel fan-out to mBackground <= MAX_TASK_CHILDREN via sharded models
    # - mConcatFit and mImgTbl fan-in <= MAX_TASK_PARENTS via reductions (budgeted)
    w_pp = max(1, int(core * 0.35))          # mProjectPP
    w_diff = max(1, int(core * 0.25))        # mDiffFit
    w_bg = max(1, core - (w_pp + w_diff))    # mBackground

    for _ in range(32):
        model_shards = (w_bg + MAX_TASK_CHILDREN - 1) // MAX_TASK_CHILDREN
        concat_internal = _reduction_internal_nodes_count(w_diff, MAX_TASK_PARENTS)
        extra_concat = max(0, concat_internal - 1)  # mConcatFit is root
        imgtbl_internal = _reduction_internal_nodes_count(w_bg, MAX_TASK_PARENTS)
        extra_imgtbl = max(0, imgtbl_internal - 1)  # mImgTbl is root
        extra_models = max(0, model_shards - 1)     # mBgModel is base shard

        used = w_pp + w_diff + w_bg + extra_concat + extra_imgtbl + extra_models
        if used <= core:
            break

        shrink = used - core
        # shrink background first, then diff, then pp
        if shrink > 0 and w_bg > 1:
            dec = min(shrink, w_bg - 1)
            w_bg -= dec
            shrink -= dec
        if shrink > 0 and w_diff > 1:
            dec = min(shrink, w_diff - 1)
            w_diff -= dec
            shrink -= dec
        if shrink > 0 and w_pp > 1:
            dec = min(shrink, w_pp - 1)
            w_pp -= dec
            shrink -= dec

    # Stage 1: mProjectPP fan-out from a small input table (implicit)
    pp = []
    for i in range(w_pp):
        k = f"mProjectPP_{i}"
        bg.add_task(k, lib.const, 1)
        pp.append(k)

    # Stage 2: mDiffFit (overlaps/diffs) depends on 1..2 projected images
    diff = []
    for i in range(w_diff):
        a = pp[i % len(pp)]
        deps = [a]
        if len(pp) >= 2 and (i % 2 == 0):
            deps.append(pp[(i * 7 + 1) % len(pp)])
        k = f"mDiffFit_{i}"
        bg.add_task(k, lib.addN, *[TaskOutputRef(d) for d in deps])
        diff.append(k)

    # Stage 3: mConcatFit join over diffs (fit params aggregation)
    if len(diff) == 1:
        bg.add_task("mConcatFit", lib.const, TaskOutputRef(diff[0]))
    else:
        _add_kary_reduction(bg, lib, diff, root_key="mConcatFit", prefix="MON_C_", fanin=MAX_TASK_PARENTS)

    # Stage 4: mBgModel depends on mConcatFit
    model_shards = (w_bg + MAX_TASK_CHILDREN - 1) // MAX_TASK_CHILDREN
    bg.add_task("mBgModel", lib.scale, TaskOutputRef("mConcatFit"), 2)
    model_keys = ["mBgModel"]
    for s in range(1, model_shards):
        mk = f"mBgModel_shard_{s}"
        bg.add_task(mk, lib.scale, TaskOutputRef("mConcatFit"), 2)
        model_keys.append(mk)

    # Stage 5: mBackground fan-out from (mBgModel + each projected image)
    back = []
    for i in range(w_bg):
        a = pp[i % len(pp)]
        k = f"mBackground_{i}"
        model = model_keys[i % len(model_keys)]
        bg.add_task(k, lib.addN, TaskOutputRef(model), TaskOutputRef(a))
        back.append(k)

    # Stage 6: mImgTbl join over corrected images
    if len(back) == 1:
        bg.add_task("mImgTbl", lib.const, TaskOutputRef(back[0]))
    else:
        _add_kary_reduction(bg, lib, back, root_key="mImgTbl", prefix="MON_T_", fanin=MAX_TASK_PARENTS)

    # Stage 7: mAdd depends on mImgTbl (co-add)
    bg.add_task("mAdd", lib.scale, TaskOutputRef("mImgTbl"), 3)

    # Stage 8: mViewer depends on mAdd
    bg.add_task("mViewer", lib.addN, TaskOutputRef("mAdd"), 1)

    # Pad remaining tasks, keep connected to mViewer
    rem = total_tasks - len(bg.task_dict)
    anchor = "mViewer"
    for i in range(rem):
        pk = f"PAD{i}"
        bg.add_task(pk, lib.addN, TaskOutputRef(anchor), 1)
        anchor = pk

    assert len(bg.task_dict) == total_tasks, (len(bg.task_dict), total_tasks)
    _assert_degree_constraints(bg)
    _assert_subgraph_size_constraints(bg)
    return bg

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, default="task-graph-pkls/test-graph.pkl")
    args = parser.parse_args()
    
    graph = make_deep_high_fanin_dag()
    with open(args.save_path, "wb") as f:
        cloudpickle.dump(graph, f)
