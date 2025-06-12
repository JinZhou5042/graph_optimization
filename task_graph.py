from graph_tool.all import Graph as GTGraph, GraphView, label_out_component, shortest_distance
from graph_tool.topology import topological_sort
from utils import *

class TaskGraph:
    @timed("TaskGraph init")
    def __init__(self, hlg_dict, compute_keys):
        self.sexpr_of = hlg_dict
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

    def _extract_deps(self, sexpr):
        if isinstance(sexpr, str):
            return {sexpr} if sexpr in self.sexpr_of else set()
        if isinstance(sexpr, tuple) and callable(sexpr[0]):
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

    def get_topo_order_of_group(self, group_keys):
        group_keys_set = set(group_keys)
        return [k for k in self.global_topo_order if k in group_keys_set]
    
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
