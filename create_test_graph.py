# build_spec_collection.py
import argparse
import math
import random
import cloudpickle
import dask._task_spec as dts

# ---- 纯 Task/Alias 用的函数 ----
def const(x):
    return x

def add(x, y):
    return x + y

# ---- 一个最小的 Dask collection 封装 ----
class SpecCollection:
    """
    Minimal Dask collection wrapper around a dict of {key: dts.Task|dts.Alias}
    and a list of output keys.
    """
    def __init__(self, graph: dict, out_keys):
        self._graph = graph
        self._out_keys = list(out_keys)

    # Dask protocols:
    def __dask_graph__(self):
        return self._graph

    def __dask_keys__(self):
        return self._out_keys

    # 这两个可选，给默认 passthrough
    @staticmethod
    def __dask_optimize__(dsk, keys, **kwargs):
        return dsk

    def __dask_postcompute__(self):
        # 返回 (func, args)；默认直接返回结果（恒等）
        return (lambda *xs: xs if len(xs) != 1 else xs[0], ())

    def __dask_postpersist__(self):
        return (SpecCollection, (self._out_keys,))

    def __repr__(self):
        return f"SpecCollection(n_nodes={len(self._graph)}, n_outputs={len(self._out_keys)})"

# ---- 构图工具 ----
def make_leaf(graph, idx):
    k = f"L{idx}"
    graph[k] = dts.Task(k, const, idx)  # 叶子任务返回常量，避免 DataNode
    return k

def make_add(graph, name, left_key, right_key):
    graph[name] = dts.Task(name, add, dts.Alias(left_key), dts.Alias(right_key))
    return name

def make_alias(graph, new_key, target_key):
    graph[new_key] = dts.Alias(target_key)
    return new_key

def build_param_graph(
    n_leaves=64,
    alias_fanout=4,
    alias_chain_len=2,
    cross_links=32,
    outputs=8,
    seed=0,
):
    """
    只包含 dts.Task / dts.Alias 的复杂图：
      1) 叶子：const(i)
      2) 自底向上 pairwise reduce：add(Alias(...), Alias(...))
      3) 每层添加一些 cross-link：从较浅层随机取一个作为右输入
      4) 对若干中间/根节点做 alias fanout
      5) 对一些节点再接 alias 链
      6) 从不同层/链选 outputs 作为集合的键
    """
    rng = random.Random(seed)
    graph = {}

    # 1) 生成叶子
    level_nodes = [make_leaf(graph, i) for i in range(n_leaves)]

    all_levels = [level_nodes[:]]  # 记录每层节点
    # 2) pairwise reduce 到根
    level = 0
    while len(level_nodes) > 1:
        next_level = []
        for i in range(0, len(level_nodes) - 1, 2):
            name = f"A{level}_{i//2}"
            make_add(graph, name, level_nodes[i], level_nodes[i+1])
            next_level.append(name)
        if len(level_nodes) % 2 == 1:
            # 把最后一个抬上来（作为别名传递）
            carry = level_nodes[-1]
            alias_k = f"AL_carry_{level}_{len(next_level)}"
            make_alias(graph, alias_k, carry)
            next_level.append(alias_k)
        level_nodes = next_level
        all_levels.append(level_nodes[:])
        level += 1

    root = level_nodes[0]

    # 3) 跨层交叉边（再造一些 add，右边从较浅层挑一个，左边是 root 的别名链终点）
    for ci in range(cross_links):
        src_level = rng.randrange(max(1, len(all_levels) // 2))  # 浅层
        src_nodes = all_levels[src_level]
        left_anchor = root
        # 给 left_anchor 加一小段别名链再作为 add 左输入
        for c in range(rng.randint(1, max(1, alias_chain_len))):
            left_alias = f"AL_root_chain_{ci}_{c}"
            make_alias(graph, left_alias, left_anchor)
            left_anchor = left_alias

        right = rng.choice(src_nodes)
        add_key = f"XADD_{ci}"
        make_add(graph, add_key, left_anchor, right)
        # 把交叉 add 的结果也放到一个层里（“交叉层”）
        all_levels.append([add_key])

    # 4) 对一些节点做 alias fanout
    fanout_sources = []
    # 选几层的几个节点作为源：root + 若干中层 + 若干交叉节点
    fanout_sources.append(root)
    if len(all_levels) >= 3:
        fanout_sources += rng.sample(all_levels[len(all_levels)//2], k=min(3, len(all_levels[len(all_levels)//2])))
    cross_layer_nodes = [k for ks in all_levels[-cross_links:] for k in ks] if cross_links else []
    if cross_layer_nodes:
        fanout_sources += rng.sample(cross_layer_nodes, k=min(5, len(cross_layer_nodes)))

    for i, src in enumerate(set(fanout_sources)):
        for j in range(alias_fanout):
            make_alias(graph, f"AL_fan_{i}_{j}", src)

    # 5) 在随机节点上附加 alias 链，增加路径深度
    all_nodes = list(graph.keys())
    for i in range(min(50, len(all_nodes)//4)):
        base = rng.choice(all_nodes)
        last = base
        length = rng.randint(1, alias_chain_len)
        for c in range(length):
            last = make_alias(graph, f"AL_chain_{i}_{c}", last)

    # 6) 选择输出 keys：根、一些扇出 alias、一些交叉 add、一些链末尾
    outs = [root]
    outs += [k for k in graph if k.startswith("AL_fan_")][:outputs//3]
    outs += [k for k in graph if k.startswith("XADD_")][:outputs//3]
    outs += [k for k in graph if k.startswith("AL_chain_")][- (outputs - len(outs)) :]
    outs = outs[:outputs] if outs else [root]

    return SpecCollection(graph, outs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-leaves", type=int, default=64)
    ap.add_argument("--alias-fanout", type=int, default=6)
    ap.add_argument("--alias-chain-len", type=int, default=4)
    ap.add_argument("--cross-links", type=int, default=40)
    ap.add_argument("--outputs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-collections", type=int, default=3, help="打包多个 collection 方便你一次性load")
    ap.add_argument("--out", type=str, default="spec_collections.pkl")
    args = ap.parse_args()

    bundles = {}
    for i in range(args.n_collections):
        coll = build_param_graph(
            n_leaves=args.n_leaves,
            alias_fanout=args.alias_fanout,
            alias_chain_len=args.alias_chain_len,
            cross_links=args.cross_links,
            outputs=args.outputs,
            seed=args.seed + i,
        )
        bundles[f"graph_{i}"] = coll

    with open(args.out, "wb") as f:
        cloudpickle.dump(bundles, f)

    print(f"Saved {len(bundles)} Dask collections to {args.out}")
    for name, coll in bundles.items():
        print(name, coll)

if __name__ == "__main__":
    main()
