import random
from collections import defaultdict
from collections import deque
import json
import hashlib
import sys
from heapq import heappush, heappop
import time
from graphviz import Digraph
import numpy as np
import numpy as np
from copy import deepcopy
import os
import heapq
import cloudpickle
import math
import ndcctools.taskvine as vine
from ndcctools.taskvine import FunctionCall
from pprint import pprint
from tqdm import tqdm
import numpy as np
from rich import print
import queue
import cloudpickle
import libs
from statistics import mean
# from libs import Graph, Node, SCHEDULING_OVERHEAD, Group
import uuid
import sys
import types
import random
from collections import defaultdict
from collections import deque
import json
from heapq import heappush, heappop
import hashlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections.abc import Hashable
from graphviz import Digraph
import numpy as np
from copy import deepcopy
from statistics import mean
import uuid


NUM_WORKERS = 4
NUM_CORES_PER_WORKER = 4


####################################################################################################

def generate_random_colors(num_colors):
    colors = []
    for _ in range(num_colors):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        color = f'#{r:02X}{g:02X}{b:02X}'
        colors.append(color)
    return colors

colors = generate_random_colors(5000)

EXECUTION_TIME_RANGE = (10, 21)
SCHEDULING_OVERHEAD = 4000000
COMMUNICATION_OVERHEAD = 10

def hash_name(*args):
    out_str = ""
    for arg in args:
        out_str += str(arg)
    return hashlib.sha256(out_str.encode('utf-8')).hexdigest()[:12]


class Graph:
    def __init__(self, hlg):
        self.nodes = {}
        self.max_node_id = 0

        self.hlg_keys = set(hlg.keys())
        self.compute_keys = set()

        # initialize each node
        for key, sexpr in hlg.items():
            self.add_node(key, sexpr)
            parent_keys = self.find_hlg_keys(sexpr)
            for parent_key in parent_keys:
                self.add_node(parent_key, hlg[parent_key])
                self.nodes[key].add_parent(parent_key)
                self.nodes[parent_key].add_child(key)

        # initialize the critical time of each node
        topo_order = self.get_topo_order()
        for k in topo_order:
            if not self.nodes[k].parents:
                self.nodes[k].critical_time = self.nodes[k].execution_time
                self.nodes[k].depth = 0
            else:
                self.nodes[k].critical_time = max([self.nodes[parent].critical_time for parent in self.nodes[k].parents]) + self.nodes[k].execution_time
                self.nodes[k].depth = max([self.nodes[parent].depth for parent in self.nodes[k].parents]) + 1

        # initialize the reachable keys
        def dfs(k, visited):
            for child in self.nodes[k].children:
                if child not in visited:
                    visited.add(child)
                    dfs(child, visited)
        for k in self.nodes.keys():
            visited = set()
            dfs(k, visited)

            self.nodes[k].add_can_reach_to([v for v in visited])
            for kk in self.nodes[k].can_reach_to:
                self.nodes[kk].can_be_reached_from(k)

    def get_output_vine_file_of(self, k):
        return self.nodes[k].output_vine_file

    def set_output_vine_file_of(self, k, output_vine_file):
        self.nodes[k].output_vine_file = output_vine_file

    def get_output_filename_of(self, k):
        return self.nodes[k].output_filename

    def set_output_filename_of(self, k, output_filename):
        self.nodes[k].output_filename = output_filename

    def children_of(self, k):
        return self.nodes[k].children

    def parents_of(self, k):
        return self.nodes[k].parents

    def node_of(self, k):
        return self.nodes[k]

    def is_key(self, s):
        try:
            hash(s)
            if s in self.hlg_keys:
                return True
            else:
                return False
        except TypeError:
            return False

    def find_hlg_keys(self, sexpr):
        parent_keys = set()
        if self.is_key(sexpr):
            parent_keys.add(sexpr)

        if isinstance(sexpr, (list, tuple)):
            for item in sexpr:
                parent_keys.update(self.find_hlg_keys(item))
        return parent_keys

    def get_num_nodes(self):
        return len(self.nodes)

    def add_node(self, key, sexpr):
        if isinstance(key, list):
            exit(1)
        if key in self.nodes.keys():
            return self.nodes[key]

        self.max_node_id += 1
        node = Node(key, sexpr, self.max_node_id, random.randint(*EXECUTION_TIME_RANGE), SCHEDULING_OVERHEAD)
        self.nodes[key] = node

        return node

    def get_topo_order(self, start_key=None, start_keys=None, keys=None):
        if keys and start_key:
            raise ValueError("keys and start_key cannot be used together")

        if keys:
            keys_to_sort = keys
        else:
            keys_to_sort = self.nodes.keys()

        visited = set()
        stack = []

        def dfs(u):
            if u in visited:
                return
            visited.add(u)
            for v in self.nodes[u].children:
                dfs(v)
            if u in keys_to_sort:
                stack.append(u)
            return

        if start_key:
            dfs(start_key)
        elif start_keys:
            for k in start_keys:
                dfs(k)
        else:
            for k in keys_to_sort:
                if k not in visited:
                    dfs(k)

        return deque(stack[::-1])

    def visualize(self, filename="graph", label='id', fill_white=False, draw_keys=None, format='svg'):
        print(f"Saving graph to {filename}.{format}")

        dot = Digraph()

        draw_keys = self.nodes.keys() if not draw_keys else draw_keys

        for key in draw_keys:
            if fill_white:
                color = 'white'
            else:
                hash_object = hashlib.sha256(str(self.node_of(key).group).encode())
                hash_int = int(hash_object.hexdigest(), 16)
                color_index = hash_int % len(colors)
                color = colors[color_index]
#            if label == 'id':
#                dot.node(node.hash_name, str(node.id), style="filled", fillcolor=color)
#            elif label == 'key':
#                dot.node(node.hash_name, node.key, style="filled", fillcolor=color)
#            elif label == 'hash_name':
#                dot.node(node.hash_name, node.hash_name, style="filled", fillcolor=color)
#            elif label == 'critical_time':
#                dot.node(node.hash_name, node.critical_time, style="filled", fillcolor=color)
#            elif label == 'execution_time':
#                dot.node(node.hash_name, str(node.execution_time), style="filled", fillcolor=color)
#            else:
#                raise ValueError(f"Unknown label: {label}")
        for k in draw_keys:
            for pk in self.node_of(k).parents:
                dot.edge(pk, k)

        dot.render(filename, format=format, cleanup=True)        


class Group:
    def __init__(self, graph, cores=0, runtime_limit=0, id=None):
        # self.nodes = set()
        self.graph = graph

        self.keys = set()
        
        self.id = id
        self.cores = cores
        self.runtime_limit = runtime_limit
        self.consider_queue = []
        self.pending_keys = set()

    def children_of(self, key):
        return self.graph.children_of(key)
    
    def parents_of(self, key):
        return self.graph.parents_of(key)

    def node_of(self, key):
        return self.graph.node_of(key)

    @property
    def children(self):
        return set.union(*[self.children_of(k) for k in self.keys]) - self.keys

    @property
    def parents(self):
        return set.union(*[self.parents_of(k) for k in self.keys]) - self.keys

    def set_cores(self, cores):
        self.cores = cores

    def add_key(self, key):
        self.keys.add(key)

        if not self.children_of(key):
            return

        for child in self.children_of(key):
            if child in self.keys or child in self.pending_keys:
                continue
            self.push_consider_queue(child)

    def push_consider_queue(self, key):
        child_children_depth = mean([self.graph.nodes[cc].depth for cc in self.children_of(key)]) if self.children_of(key) else 0
        heappush(self.consider_queue, (child_children_depth, id(key), key))

    def pop_consider_queue(self):
        if not self.consider_queue:
            return None
        return heappop(self.consider_queue)[2]

    # the closured nodes cannot be inter-reached from the nodes outside the closure
    def get_pending_keys(self, new_key):
        self.pending_keys = set([new_key])
        combined_keys = self.keys | self.pending_keys

        visited = set()

        def dfs(current_key):
            if current_key in visited:
                return
            visited.add(current_key)

            # skip if this node can not reach one of the end nodes
            if not (self.graph.nodes[current_key].can_reach_to & combined_keys):
                return

            self.pending_keys.add(current_key)
            combined_keys.add(current_key)

            for child_key in self.graph.nodes[current_key].children:
                dfs(child_key)

        # the new node can reach other nodes, while other nodes can reach the new node
        for k in self.keys | set([new_key]):
            dfs(k)

        # don't need it for the following
        combined_keys = None

        self.pending_keys -= self.keys

        # the pending nodes should not depend on ouside nodes
        outside_parents = deque()
        for pending_key in self.pending_keys:
            for parent in self.parents_of(pending_key):
                if not self.graph.nodes[parent].group:
                    outside_parents.append(parent)
        while outside_parents:
            parent = outside_parents.popleft()
            if parent in self.pending_keys:
                continue
            for grandparent in self.parents_of(parent):
                if not self.node_of(grandparent).group:
                    outside_parents.append(grandparent)
            self.pending_keys.add(parent)

    def merge_pending_keys(self):
        for k in self.pending_keys:
            self.add_key(k)
            self.graph.nodes[k].group = self
        self.pending_keys = set()

    def revert_pending_keys(self):
        self.pending_keys = set()
    
    def get_critical_time(self):
        if not self.keys:
            return 0

        self.critical_time = self.schedule_to_cores()

        return round(self.critical_time, 4)
    
    def get_resource_utilization(self):
        if not self.keys:
            return 0
        if not self.get_critical_time():
            return 0

        return round(sum([self.graph.nodes[k].execution_time for k in self.keys | self.pending_keys]) / (self.get_critical_time() * self.cores), 4)

    def calculate_bottom_reach_time(self):
        temp_keys = self.keys.copy() | self.pending_keys.copy()
        bottom_reach_time = {}
        def dfs(k):
            if k not in temp_keys:
                return 0
            if k in bottom_reach_time:
                return bottom_reach_time[k]

            max_time_to_leaf = max(
                (dfs(child) + self.node_of(child).execution_time 
                for child in self.node_of(k).children if child in temp_keys),
                default=0
            )
            bottom_reach_time[k] = max_time_to_leaf
            return max_time_to_leaf

        for k in temp_keys:
            dfs(k)

        return bottom_reach_time

    def schedule_to_cores(self):
        # cannot find the optimal scheduling algorithm
        if not self.keys:
            return

        ready_keys = []
        running_keys = []
        num_available_cores = self.cores

        temp_keys = self.keys.copy() | self.pending_keys.copy()
        
        num_pending_parents = {k: 0 for k in temp_keys}
        for k in temp_keys:
            for pk in self.parents_of(k):
                if pk in temp_keys:
                    num_pending_parents[k] += 1

        waiting_keys = set([k for k in temp_keys if num_pending_parents[k] > 0])
        num_available_cores = self.cores
        bottom_reach_time = self.calculate_bottom_reach_time()

        def enqueue_ready_keys(k):
            heappush(ready_keys, (-bottom_reach_time[k], id(k), k))

        for k in temp_keys:
            if num_pending_parents[k] == 0:
                enqueue_ready_keys(k)

        current_time = 0
        critical_time = 0

        start_time = {}
        end_time = {}

        while ready_keys or running_keys:
            # submit as many tasks as possible
            while ready_keys and num_available_cores:
                ready_key = heappop(ready_keys)[2]
                start_time[ready_key] = current_time
                end_time[ready_key] = current_time + self.node_of(ready_key).execution_time

                heappush(running_keys, (end_time[ready_key], id(ready_key), ready_key))
                num_available_cores -= 1

            # get the earliest finished task
            finished_key = heappop(running_keys)[2]
            current_time = end_time[finished_key]
            num_available_cores += 1
            critical_time = max(critical_time, end_time[finished_key])

            for child in self.children_of(finished_key):
                if child in waiting_keys:
                    num_pending_parents[child] -= 1
                    if num_pending_parents[child] == 0:
                        enqueue_ready_keys(child)
                        waiting_keys.remove(child)

        assert len(ready_keys) == len(running_keys) == len(waiting_keys) == 0
        assert max([end_time[k] for k in temp_keys]) == critical_time

        return critical_time

    def merge_from(self, starting_key):
        if self.node_of(starting_key).group:
            return

        self.push_consider_queue(starting_key)
        new_keys = set()

        while (consider_key := self.pop_consider_queue()):
            # try to merge more nodes
            self.get_pending_keys(consider_key)

            self.critical_time = self.schedule_to_cores()

            if self.get_critical_time() <= self.runtime_limit:
                new_keys.add(consider_key)
                new_keys.update(self.pending_keys)
                self.merge_pending_keys()
            else:
                self.revert_pending_keys()

        self.critical_time = self.schedule_to_cores()
        return new_keys

    def visualize(self, save_to=None, show=False, label=None):
        exit(1)
        self.critical_time = self.schedule_to_cores()

        fig, ax = plt.subplots(figsize=(10, 6))
        group_nodes = sorted(self.nodes, key=lambda task: task.start_time)

        core_end_times = []
        task_core_mapping = []

        for task in group_nodes:
            assigned_core = None
            for core_id, end_time in enumerate(core_end_times):
                if end_time <= task.start_time:
                    assigned_core = core_id
                    core_end_times[core_id] = task.end_time 
                    break

            if assigned_core is None:
                assigned_core = len(core_end_times)
                core_end_times.append(task.end_time)

            task_core_mapping.append((task, assigned_core))

        for task, core_id in task_core_mapping:
            ax.barh(core_id, task.end_time - task.start_time, left=task.start_time, edgecolor='black')
            if label == 'id':
                bar_text = f'{task.id}'
            elif label == 'execution_time':
                bar_text = f'{task.execution_time}'
            else:
                bar_text = ''
            ax.text(task.start_time + (task.end_time - task.start_time) / 2, core_id, bar_text,
                    va='center', ha='center', color='white')

        ax.set_xlabel('Time')
        ax.set_ylabel('Core ID')
        # x range
        ax.set_yticks(range(len(core_end_times)))
        ax.set_yticklabels([f'Core {i}' for i in range(len(core_end_times))])
        plt.title('Task Distribution Across Cores')
        if save_to:
            plt.savefig(save_to)
        if show:
            plt.show()


class Node:
    def __init__(self, key, sexpr, id, execution_time, scheduling_overhead):
        # must be initialized
        self.key = key
        self.sexpr = sexpr
        self.id = id
        self.execution_time = execution_time
        self.scheduling_overhead = scheduling_overhead

        self.hash_name = hash_name(key)

        self.output_vine_file = None
        self.output_filename = f"{uuid.uuid4()}.pkl"

        self.children = set()
        self.parents = set()

        self.pending_parents = set()
        self.temp_pending_parents = set()

        # time taken from the start of the graph to the end of this node
        self.critical_time = 0

        # the nodes that can be reached from this node
        self.can_reach_to = set()
        # the nodes that can reach this node
        self.reachable_from_nodes = set()

        self.group = None

        self.start_time = 0
        self.end_time = 0

        self.depth = 0

        self.completed = False


    def add_parent(self, parent):
        self.parents.add(parent)
        self.pending_parents.add(parent)

    def add_child(self, child):
        self.children.add(child)
    
    def remove_parent(self, parent):
        self.parents.remove(parent)

    def remove_child(self, child):
        self.children.remove(child)
    
    def add_can_reach_to(self, keys):
        self.can_reach_to.update(keys)

    def remove_reachable_to_node(self, key):
        self.can_reach_to.remove(key)

    def can_be_reached_from(self, key):
        self.reachable_from_nodes.add(key)

    def remove_reachable_from_node(self, key):
        self.reachable_from_nodes.remove(key)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('group', None)
        return state


class SimpleGroup:
    def __init__(self, graph, group):
        self.keys = group.keys
        self.sexpr_of = {k: graph.node_of(k).sexpr for k in group.keys}
        self.parents = group.parents
        self.children = group.children
        self.children_of = {k: group.children_of(k) for k in group.keys}
        self.parents_of = {k: group.parents_of(k) for k in group.keys}
        self.pending_parents = self.parents_of.copy()
        self.num_pending_parents = {k: sum(1 for p in self.parents_of[k] if p in self.keys) for k in self.keys}
        self.ready_keys = [k for k in self.keys if self.num_pending_parents[k] == 0]
        self.waiting_keys = [k for k in self.keys if self.num_pending_parents[k] > 0]

        self.output_of = {k: None for k in group.keys}
        # these are parent keys outside of this group, these outputs are used as inputs and should be loaded from pkl
        for k in self.parents:
            self.output_of[k] = graph.get_output_filename_of(k)

        # output of these keys are going to be dumped to files
        self.outkeys = {k for k in self.keys if any(child not in self.keys for child in self.children_of[k])}
        self.output_filenames = {k: graph.get_output_filename_of(k) for k in self.outkeys}


class MyFunctionCall(FunctionCall):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.group = None


####################################################################################################

# with open('simplified_hlg.json', 'r') as json_in:
#     loaded_children_of = json.load(json_in)
#     children_of = {key: set(value) for key, value in loaded_children_of.items()}

# graph = Graph(children_of)



with open("expanded_hlg.pkl", "rb") as f:
    hlg = cloudpickle.load(f)

    all_tasks = {}
    for layer_name, layer in hlg.layers.items():
        for key, sexpr in layer.items():
            all_tasks[key] = sexpr

    graph = Graph(all_tasks)


with open("keys.pkl", "rb") as f:
    keys = cloudpickle.load(f)
    graph.compute_keys = graph.find_hlg_keys(keys)


enqueued_groups = []


def execute_group(group):

    # output: the direct output
    # output_filename: the filename to dump the output
    # output_vine_file: the vine file to load the output

    for k, output in group.output_of.items():
        if not output:
            continue

        try:
            with open(output, 'rb') as f:
                group.output_of[k] = cloudpickle.load(f)

        except Exception as e:
            print(f"Error: {e}")
            raise

    def rec_call(sexpr):
        if hash(sexpr) and sexpr in group.keys | group.parents:
            return group.output_of[sexpr]

        if isinstance(sexpr, list):
            return [rec_call(item) for item in sexpr]
        if isinstance(sexpr, tuple) and callable(sexpr[0]):
            return sexpr[0](*[rec_call(item) for item in sexpr[1:]])
        else:
            return sexpr

    while group.ready_keys:
        # run a task
        k = group.ready_keys.pop(0)
        group.output_of[k] = rec_call(group.sexpr_of[k])
        if k in group.outkeys:
            with open(group.output_filenames[k], 'wb') as f:
                cloudpickle.dump(group.output_of[k], f)
        # submit more tasks
        for child in group.children_of[k]:
            if child in group.waiting_keys:
                group.num_pending_parents[child] -= 1
                if group.num_pending_parents[child] == 0:
                    group.ready_keys.append(child)
                    group.waiting_keys.remove(child)

    return group.keys


# g.visualize('/Users/jinzhou/Downloads/original_hlg', fill_white=True)

def consume_ready_keys(graph, ready_keys, q):

    mergeable_keys = ready_keys.copy()

    while mergeable_keys:

        group = Group(graph, cores=1, runtime_limit=500, id=0)

        for mergeable_key in mergeable_keys:
            if graph.node_of(mergeable_key).group:
                mergeable_keys.remove(mergeable_key)
                continue
            new_keys = group.merge_from(mergeable_key)

            print(f"merging from {mergeable_key}, new keys: {new_keys}")

            for k in new_keys:
                if k in mergeable_keys:
                    mergeable_keys.remove(k)

                for ck in graph.children_of(k):
                    ck_ready = True
                    for pck in graph.parents_of(ck):
                        # skip if the parent is grouped
                        if graph.node_of(pck).group:
                            continue
                        ck_ready = False
                    if ck_ready:
                        mergeable_keys.append(ck)

        if not group.keys:
            continue

        simple_group = SimpleGroup(graph, group)

        t = MyFunctionCall('dask-library', 'execute_group', simple_group)
        t.group = simple_group
    
        t.enable_temp_output()

        for k in simple_group.parents:
            t.add_input(graph.get_output_vine_file_of(k), graph.get_output_filename_of(k))

        for k, output_filename in simple_group.output_filenames.items():
            # f = q.declare_file(os.path.join(staging_directory, "outputs", output_filename))
            f = q.declare_temp()
            t.add_output(f, output_filename)
            graph.set_output_vine_file_of(k, f)

        q.submit(t)

        print(len(group.keys), group.get_critical_time(), group.get_resource_utilization())


def test():
    return 100


q = vine.Manager(9123, name="graph-optimization")
q.tune("watch-library-logfiles", 1)
hoisting_modules=[Graph, Node, Group, uuid, mean, hash_name]
libtask = q.create_library_from_functions('dask-library', execute_group, test, add_env=False, hoisting_modules=hoisting_modules)
q.install_library(libtask)
staging_directory = q.staging_directory



def execute_graph(graph):

    ready_keys = []
    for k, n in graph.nodes.items():
        if not n.pending_parents:
            ready_keys.append(k)

    print(f"all nodes: {len(graph.nodes)}")
    time_start = time.time()

    consume_ready_keys(graph, ready_keys, q)

    pbar = tqdm(total=len(graph.nodes))
    while not q.empty():
        t = q.wait(1)
        if t:
            if t.successful():
                keys = t.group.keys
                pbar.update(len(keys))
                # enqueue the children of the group
                for ck in t.group.children:
                    if graph.node_of(ck).completed:
                        ready_keys.remove(ck)
                    else:
                        ready_keys.append(ck)

                consume_ready_keys(graph, ready_keys, q)
            else:
                print(f"failed")
        else:
            print("waiting for results")

    pbar.close()

    # assert sum([len(group.keys) for group in groups]) == len(graph.nodes)

    print(f"Execution time: {time.time() - time_start}")

execute_graph(graph)

# graph.visualize('merged_hlg')
