from graph_tool.all import Graph as GTGraph, GraphView, label_out_component, shortest_distance
from graph_tool.topology import topological_sort
import cloudpickle
import hashlib
from utils import *

FUNCTION_REGISTRY = {}
FUNCTION_REGISTRY_INV = {}

def register_function(fn):
    try:
        fn_id = f"{fn.__module__}.{fn.__name__}"
    except AttributeError:
        pickled = cloudpickle.dumps(fn)
        fn_id = "func_" + hashlib.md5(pickled).hexdigest()

    if fn_id not in FUNCTION_REGISTRY:
        FUNCTION_REGISTRY[fn_id] = fn
        FUNCTION_REGISTRY_INV[fn] = fn_id
    return fn_id


class TaskGraph:
    def __init__(self, hlg_dict, compute_keys, convert_sexpr=True):
        self.sexpr_of = self._create_sexpr(hlg_dict, convert_sexpr)
        self.compute_keys = compute_keys
        self.g = GTGraph(directed=True)
        self.key_to_vertex = {}
        self.vertex_to_key = {}
        self.output_filename_of = {key: f"{uuid.uuid4()}.pkl" for key in self.sexpr_of.keys()}
        self.output_vine_file_of = {key: None for key in self.sexpr_of.keys()}

        # compute dependencies and add all nodes
        self.parents_of = {}
        for key in self.sexpr_of:
            v = self.g.add_vertex()
            self.key_to_vertex[key] = v
            self.vertex_to_key[int(v)] = key
            self.parents_of[key] = self._extract_deps(self.sexpr_of[key])

        # compute children
        self.children_of = defaultdict(set)
        for key, deps in self.parents_of.items():
            for dep in deps:
                self.children_of[dep].add(key)

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

    @timed("TaskGraph create sexpr")
    def _create_sexpr(self, hlg_dict, convert_sexpr=True):
        sexpr_of = {}
        for key, hlg in hlg_dict.items():
            sexpr_of[key] = self._convert_sexpr(hlg) if convert_sexpr else hlg
        return sexpr_of
        
    def _convert_sexpr(self, expr):
        # function call: (callable, arg1, arg2, ...)
        if isinstance(expr, tuple) and callable(expr[0]):
            fn_name = register_function(expr[0])
            return (fn_name,) + tuple(self._convert_sexpr(e) for e in expr[1:])
        
        # plain list or tuple (e.g., tuple passed as a parameter): keep it as-is, don't recurse
        if isinstance(expr, list):
            return [self._convert_sexpr(e) for e in expr]
        
        return expr  # str, int, tuple as atomic argument

    def _extract_deps(self, sexpr):
        if isinstance(sexpr, str):
            return {sexpr} if sexpr in self.sexpr_of else set()
        if isinstance(sexpr, tuple) and isinstance(sexpr[0], str) and sexpr[0] in FUNCTION_REGISTRY:
            # function call: ("module.fn_name", arg1, arg2, ...)
            return set.union(*(self._extract_deps(e) for e in sexpr[1:]))
        if isinstance(sexpr, (list, tuple)):
            return set.union(*(self._extract_deps(e) for e in sexpr))
        return set()

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
        return {k: self.sexpr_of[k] for k in group_keys}
