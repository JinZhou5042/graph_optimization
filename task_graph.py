from graph_tool.all import Graph as GTGraph, GraphView, label_out_component, shortest_distance
from graph_tool.topology import topological_sort
import cloudpickle
import hashlib
from utils import *
from collections import defaultdict
import uuid
import numpy as np


class TaskGraph:
    def __init__(self, task_dict, enable_sexpr_conversion=True, expand_dsk=True, debug=False):
        if expand_dsk:
            self.task_dict = self._expand_dsk(task_dict)
        else:
            self.task_dict = task_dict

        self.map_callable_to_id = {}
        self.map_id_to_callable = {}
        self.task_dict = self._create_callable_mapping(self.task_dict)

        self.key_to_idx = {k: i + 1 for i, k in enumerate(self.task_dict)}

        self.parents_of = defaultdict(set)
        self.children_of = defaultdict(set)
        self._build_dependencies()

        if debug:
            self._write_task_dependencies()
            self._visualize_task_dependencies()

        self.g = GTGraph(directed=True)
        self.key_to_vertex = {}
        self.vertex_to_key = {}
        self.output_filename_of = {key: f"{uuid.uuid4()}.pkl" for key in self.task_dict.keys()}
        self.output_vine_file_of = {key: None for key in self.task_dict.keys()}

        # add all edges
        for key, deps in self.parents_of.items():
            for dep in deps:
                if dep in self.key_to_vertex:
                    self.g.add_edge(self.key_to_vertex[dep], self.key_to_vertex[key])

        # compute depth
        self.depth = self.g.new_vertex_property("int")
        depth_np = np.zeros(self.g.num_vertices(), dtype=np.int32)
        edges = self.g.get_edges()[:, :2]
        src = edges[:, 0]
        dst = edges[:, 1]
        order = np.argsort(dst)
        src_sorted = src[order]
        dst_sorted = dst[order]
        counts = np.bincount(dst_sorted, minlength=self.g.num_vertices())
        offsets = np.zeros(self.g.num_vertices() + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(counts)
        for v_idx in topological_sort(self.g):
            start = offsets[v_idx]
            end = offsets[v_idx + 1]
            in_neighbors = src_sorted[start:end]
            max_d = np.max(depth_np[in_neighbors]) if in_neighbors.size > 0 else -1
            depth_np[v_idx] = max_d + 1
        self.depth.a = depth_np

        # global topo order
        self.global_topo_order = [self.vertex_to_key[int(v)] for v in topological_sort(self.g)]

    @timed("TaskGraph build parent-child relationships")
    def _build_dependencies(self):
        def _find_parents(sexpr):
            if hashable(sexpr) and sexpr in self.task_dict.keys():
                return {sexpr}
            elif isinstance(sexpr, (list, tuple)):
                deps = set()
                for x in sexpr:
                    deps |= _find_parents(x)
                return deps
            elif isinstance(sexpr, dict):
                deps = set()
                for k, v in sexpr.items():
                    deps |= _find_parents(k)
                    deps |= _find_parents(v)
                return deps
            else:
                return set()

        for key, sexpr in self.task_dict.items():
            self.parents_of[key] = _find_parents(sexpr)

        for key, deps in self.parents_of.items():
            for dep in deps:
                self.children_of[dep].add(key)

    @timed("TaskGraph write task dependencies")
    def _write_task_dependencies(self):
        with open("task_dependencies.txt", "w") as f:
            for key, parents in self.parents_of.items():
                if parents:
                    for parent in parents:
                        if parent != key:
                            f.write(f"{self.key_to_idx[parent]} -> {self.key_to_idx[key]}\n")
                else:
                    f.write(f"None -> {self.key_to_idx[key]}\n")

    @timed("TaskGraph visualize task dependencies")
    def _visualize_task_dependencies(self, input_file="task_dependencies.txt", output_file="task_graph"):
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
                    dot.node(child)  # ensure orphan nodes appear
        dot.render(output_file, cleanup=True)

    def convert_expr_to_task_args(self, dsk_key, task_dict, args, blockwise_args):
        try:
            if args in task_dict:
                return hash_name(dsk_key, args)
        except:
            pass
        if isinstance(args, list):
            return [self.convert_expr_to_task_args(dsk_key, task_dict, item, blockwise_args) for item in args]
        elif isinstance(args, tuple):
            # nested tuple is not allowed
            return tuple(self.convert_expr_to_task_args(dsk_key, task_dict, item, blockwise_args) for item in args)
        else:
            if isinstance(args, str) and args.startswith('__dask_blockwise__'):
                blockwise_arg_idx = int(args.split('__')[-1])
                return blockwise_args[blockwise_arg_idx]
            return args

    @timed("TaskGraph expand task_dict")
    def _expand_dsk(self, task_dict, save_dir="./"):
        assert os.path.exists(save_dir)

        expanded_task_dict = {}

        for k, sexpr in task_dict.items():
            callable = sexpr[0]
            args = sexpr[1:]

            if isinstance(callable, dask.optimization.SubgraphCallable):
                expanded_task_dict[k] = hash_name(k, callable.outkey)
                for sub_key, sub_sexpr in callable.dsk.items():
                    unique_key = hash_name(k, sub_key)
                    expanded_task_dict[unique_key] = self.convert_expr_to_task_args(k, callable.dsk, sub_sexpr, args)
            elif isinstance(callable, types.FunctionType):
                expanded_task_dict[k] = sexpr
            else:
                print(f"ERROR: unexpected type: {type(callable)}")
                exit(1)
        
        return expanded_task_dict

    @timed("TaskGraph reduce task_dict callables")
    def _create_callable_mapping(self, task_dict):
        def recurse(obj):
            if isinstance(obj, dict):
                return {k: recurse(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recurse(e) for e in obj]
            elif isinstance(obj, tuple):
                if callable(obj[0]):
                    try:
                        callable_id = f"{obj[0].__module__}.{obj[0].__name__}"
                    except AttributeError:
                        pickled = cloudpickle.dumps(obj[0])
                        callable_id = "callable_" + hashlib.md5(pickled).hexdigest()
                    self.map_callable_to_id[obj[0]] = callable_id
                    self.map_id_to_callable[callable_id] = obj[0]
                    return (callable_id,) + tuple(recurse(e) for e in obj[1:])
                else:
                    return tuple(recurse(e) for e in obj)
            else:
                return obj

        return {k: recurse(v) for k, v in task_dict.items()}

    def get_reachable_from(self, key):
        v = self.key_to_vertex[key]
        reachable = label_out_component(self.g, v)
        return {self.vertex_to_key[int(u)] for u in self.g.vertices() if reachable[u]}

    def get_reaching_to(self, key):
        rev_g = GraphView(self.g, reversed=True)
        v = self.key_to_vertex[key]
        reachable = label_out_component(rev_g, v)
        return {self.vertex_to_key[int(u)] for u in self.g.vertices() if reachable[u]}

    def get_distance_from(self, key):
        v = self.key_to_vertex[key]
        dist = shortest_distance(self.g, source=v)
        return {self.vertex_to_key[int(u)]: dist[u] for u in self.g.vertices()}
    
    def get_input_keys_of_group(self, group_keys):
        group_set = set(group_keys)
        return {
            p for k in group_keys
            for p in self.parents_of[k]
            if p not in group_set
        }

    def get_output_keys_of_group(self, group_keys):
        group_set = set(group_keys)
        return {
            k for k in group_keys
            if any(c not in group_set for c in self.children_of[k])
        }

    def get_sexpr_of_group(self, group_keys):
        return {k: self.task_dict[k] for k in group_keys}
