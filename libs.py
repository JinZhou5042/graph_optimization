import random
from collections import defaultdict
from collections import deque
import json
from heapq import heappush, heappop
import hashlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections.abc import Hashable
from heapq import heappush, heappop
from graphviz import Digraph
import numpy as np
from copy import deepcopy
from statistics import mean
import uuid

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
            node = self.add_node(key, sexpr)
            parent_keys = self.find_hlg_keys(sexpr)
            for parent_key in parent_keys:
                parent_node = self.add_node(parent_key, sexpr)
                node.add_parent(parent_node)
                parent_node.add_child(node)

        # initialize the critical time of each node
        topo_order = self.get_topo_order()
        for node in topo_order:
            if not node.parents:
                node.critical_time = node.execution_time
                node.depth = 0
            else:
                node.critical_time = max([parent.critical_time for parent in node.parents]) + node.execution_time
                node.depth = max([parent.depth for parent in node.parents]) + 1

        # initialize the reachable nodes from each node
        def dfs(node, visited):
            for child_node in node.children:
                if child_node not in visited:
                    visited.add(child_node)
                    dfs(child_node, visited)
        for node in self.nodes.values():
            visited = set()
            dfs(node, visited)
            node.add_reachable_to_nodes(visited)
            for reachable_node in node.reachable_to_nodes:
                reachable_node.add_reachable_from_node(node)

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

    def remove_node(self, key):
        node = self.nodes[key]

        def dfs(node):
            for parent in node.parents:
                parent.remove_reachable_to_node(node)
                dfs(parent)
        dfs(node)

        def dfs(node):
            for child in node.children:
                child.remove_reachable_from_node(node)
                dfs(child)
        dfs(node)

        # remove the node
        for parent in node.parents:
            parent.remove_child(node)
        for child in node.children:
            child.remove_parent(node)
    
        del self.nodes[key]

    def get_topo_order(self, start_node=None, start_nodes=None, nodes=None):
        # start_node: start from this node and all of its children
        # nodes: only consider these nodes in the graph

        if nodes and start_node:
            raise ValueError("node_names and start_node cannot be used together")

        if nodes:
            if not nodes.issubset(self.nodes.values()):
                raise ValueError("node_names must be subset of the graph")
            nodes_to_sort = nodes
        else:
            nodes_to_sort = self.nodes.values()

        visited = set()
        stack = []

        def dfs(u):
            if u in visited:
                return
            visited.add(u)
            for v in u.children:
                dfs(v)
            if u in nodes_to_sort:
                stack.append(u)
            return

        if start_node:
            dfs(start_node)
        elif start_nodes:
            for node in start_nodes:
                dfs(node)
        else:
            for node in nodes_to_sort:
                if node not in visited:
                    dfs(node)

        return deque(stack[::-1])

    def merge_nodes(self, group_nodes):
        part_topo_order = self.get_topo_order(nodes=group_nodes)
        for i, part_node in enumerate(part_topo_order):
            part_node.belongs_to_part = group_nodes
            if i >= 1:
                part_node.critical_time = part_topo_order[i - 1].critical_time + part_node.execution_time

        if len(group_nodes) >= 3:
            # update the downstream critical time
            downstream_topo_order = self.get_topo_order(start_nodes=group_nodes)
            downstream_topo_order = deque(set(downstream_topo_order) - group_nodes)

            for node in downstream_topo_order:
                node.critical_time = max([parent.critical_time for parent in node.parents]) + node.execution_time

    def visualize(self, filename="graph", label='id', fill_white=False, draw_nodes=None):
        print(f"Saving graph to {filename}.svg")

        dot = Digraph()

        draw_nodes = self.nodes.values() if not draw_nodes else draw_nodes

        for node in draw_nodes:
            if fill_white:
                color = 'white'
            else:
                hash_object = hashlib.sha256(str(node.group).encode())
                hash_int = int(hash_object.hexdigest(), 16)
                color_index = hash_int % len(colors)
                color = colors[color_index]
            if label == 'id':
                dot.node(node.hash_name, str(node.id), style="filled", fillcolor=color)
            elif label == 'key':
                dot.node(node.hash_name, node.key, style="filled", fillcolor=color)
            elif label == 'hash_name':
                dot.node(node.hash_name, node.hash_name, style="filled", fillcolor=color)
            elif label == 'critical_time':
                dot.node(node.hash_name, node.critical_time, style="filled", fillcolor=color)
            elif label == 'execution_time':
                dot.node(node.hash_name, str(node.execution_time), style="filled", fillcolor=color)
            else:
                raise ValueError(f"Unknown label: {label}")
        for node in draw_nodes:
            for parent_node in node.parents:
                dot.edge(parent_node.hash_name, node.hash_name)

        dot.render(filename, format='svg', cleanup=True)        


class Group:
    def __init__(self, graph, cores=0, runtime_limit=0, id=None):
        self.graph = graph
        self.nodes = set()
        self.id = id
        self.cores = cores
        self.runtime_limit = runtime_limit
        self.consider_queue = []
        self.pending_nodes = set()

    def get_parents(self):
        parents = set()
        for node in self.nodes:
            parents.update(node.parents)
        return parents

    def get_children(self):
        children = set()
        for node in self.nodes:
            children.update(node.children)
        return children

    def set_cores(self, cores):
        self.cores = cores

    def add_node(self, node):
        self.nodes.add(node)

        for child in node.children:
            if child in self.nodes or child in self.pending_nodes:
                continue
            self.push_consider_queue(child)

    def push_consider_queue(self, node):
        child_children_depth = mean([cc.depth for cc in node.children]) if node.children else 0
        heappush(self.consider_queue, (child_children_depth, id(node), node))

    def pop_consider_queue(self):
        if not self.consider_queue:
            return None
        return heappop(self.consider_queue)[2]

    def remove_node(self, node):
        self.nodes.remove(node)

    def remove_nodes(self, nodes):
        self.nodes.difference_update(nodes)

    # the closured nodes cannot be inter-reached from the nodes outside the closure
    def get_pending_nodes(self, new_node):
        self.pending_nodes = set([new_node])
        combined_nodes = self.nodes | self.pending_nodes

        visited = set()

        def dfs(current_node):
            if current_node in visited:
                return
            visited.add(current_node)

            # skip if this node can not reach one of the end nodes
            if not (current_node.reachable_to_nodes & combined_nodes):
                return

            self.pending_nodes.add(current_node)
            combined_nodes.add(current_node)

            for child_node in current_node.children:
                dfs(child_node)

        # the new node can reach other nodes, while other nodes can reach the new node
        for n in self.nodes | set([new_node]):
            dfs(n)

        # don't need it anymore
        combined_nodes = None

        self.pending_nodes -= self.nodes

        # the pending nodes should not depend on ouside nodes
        outside_parents = deque()
        for pending_node in self.pending_nodes:
            for parent in pending_node.parents:
                if not parent.group:
                    outside_parents.append(parent)
        while outside_parents:
            parent = outside_parents.popleft()
            if parent in self.pending_nodes:
                continue
            for grandparent in parent.parents:
                if not grandparent.group:
                    outside_parents.append(grandparent)
            self.pending_nodes.add(parent)

    def merge_pending_nodes(self):
        for node in self.pending_nodes:
            self.add_node(node)
            node.group = self
        self.pending_nodes = set()

    def revert_pending_nodes(self):
        self.pending_nodes = set()
    
    def get_critical_time(self):
        if not self.nodes:
            return 0
        return round(max([node.end_time for node in self.nodes | self.pending_nodes]), 4)
    
    def get_resource_utilization(self):
        if not self.nodes:
            return 0
        if not self.get_critical_time():
            return 0
        return round(sum([node.execution_time for node in self.nodes | self.pending_nodes]) / (self.get_critical_time() * self.cores), 4)

    def calculate_bottom_reach_time(self):
        temp_group_nodes = self.nodes.copy() | self.pending_nodes.copy()
        bottom_reach_time = {}
        def dfs(node):
            if node not in temp_group_nodes:
                return 0
            if node in bottom_reach_time:
                return bottom_reach_time[node]
            
            max_time_to_leaf = max(
                (dfs(child) + child.execution_time 
                for child in node.children if child in temp_group_nodes),
                default=0
            )
            bottom_reach_time[node] = max_time_to_leaf
            return max_time_to_leaf

        for node in temp_group_nodes:
            dfs(node)

        return bottom_reach_time

    def schedule_to_cores(self):
        # cannot find the optimal scheduling algorithm
        if not self.nodes:
            return
        
        ready_tasks = []
        running_tasks = []
        num_available_cores = self.cores
        temp_group_nodes = self.nodes.copy() | self.pending_nodes.copy()
        node_num_pending_parents = {node: len(node.pending_parents) for node in temp_group_nodes}
        waiting_tasks = set([node for node in temp_group_nodes if node_num_pending_parents[node] > 0])
        num_available_cores = self.cores
        bottom_reach_time = self.calculate_bottom_reach_time()
        node_num_pending_parents = {node: len(node.pending_parents) for node in temp_group_nodes}

        def enqueue_ready_tasks(node):
            heappush(ready_tasks, (-bottom_reach_time[node], id(node), node))

        for node in temp_group_nodes:
            if node_num_pending_parents[node] == 0:
                enqueue_ready_tasks(node)

        current_time = 0
        critical_time = 0

        while ready_tasks or running_tasks:
            # submit as many tasks as possible
            while ready_tasks and num_available_cores:
                _, _, ready_task = heappop(ready_tasks)
                ready_task.start_time = current_time
                ready_task.end_time = current_time + ready_task.execution_time
                heappush(running_tasks, (ready_task.end_time, id(ready_task), ready_task))
                num_available_cores -= 1

            # get the earliest finished task
            _, _, finished_task = heappop(running_tasks)
            current_time = finished_task.end_time
            num_available_cores += 1
            critical_time = max(critical_time, finished_task.end_time)

            for child in finished_task.children:
                if child in waiting_tasks:
                    node_num_pending_parents[child] -= 1
                    if node_num_pending_parents[child] == 0:
                        enqueue_ready_tasks(child)
                        waiting_tasks.remove(child)

        assert len(ready_tasks) == len(running_tasks) == len(waiting_tasks) == 0
        assert max([task.end_time for task in temp_group_nodes]) == critical_time

        return critical_time

    def merge_from(self, starting_node):
        if starting_node.group:
            return

        self.push_consider_queue(starting_node)
        new_group_nodes = set()

        while (consider_node := self.pop_consider_queue()):
            # try to merge more nodes
            self.get_pending_nodes(consider_node)
            self.schedule_to_cores()

            if self.get_critical_time() <= self.runtime_limit:
                new_group_nodes.add(consider_node)
                new_group_nodes.update(self.pending_nodes)
                self.merge_pending_nodes()
            else:
                self.revert_pending_nodes()

        self.schedule_to_cores()
        return new_group_nodes

    def visualize(self, save_to=None, show=False, label=None):
        self.schedule_to_cores()

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

        self.children = set()
        self.parents = set()

        self.pending_parents = set()
        self.temp_pending_parents = set()

        # time taken from the start of the graph to the end of this node
        self.critical_time = 0

        # the nodes that can be reached from this node
        self.reachable_to_nodes = set()
        # the nodes that can reach this node
        self.reachable_from_nodes = set()

        self.group = None

        self.start_time = 0
        self.end_time = 0

        self.depth = 0

        self.result_file = None

    def add_parent(self, parent):
        self.parents.add(parent)
        self.pending_parents.add(parent)

    def add_child(self, child):
        self.children.add(child)
    
    def remove_parent(self, parent):
        self.parents.remove(parent)

    def remove_child(self, child):
        self.children.remove(child)
    
    def add_reachable_to_nodes(self, nodes):
        self.reachable_to_nodes.update(nodes)

    def remove_reachable_to_node(self, node):
        self.reachable_to_nodes.remove(node)
    
    def remove_reachable_to_nodes(self, nodes):
        self.reachable_to_nodes.difference_update(nodes)

    def add_reachable_from_node(self, node):
        self.reachable_from_nodes.add(node)

    def add_reachable_from_nodes(self, nodes):
        self.reachable_from_nodes.update(nodes)

    def remove_reachable_from_node(self, node):
        self.reachable_from_nodes.remove(node)
    
    def remove_reachable_from_nodes(self, nodes):
        self.reachable_from_nodes.difference_update(nodes)