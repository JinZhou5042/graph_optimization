import random
from collections import defaultdict
from collections import deque
import json
from heapq import heappush, heappop
import hashlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

EXECUTION_TIME_RANGE = (10, 30)
SCHEDULING_OVERHEAD = 4000000
COMMUNICATION_OVERHEAD = 10

def hash_name(*args):
    out_str = ""
    for arg in args:
        out_str += str(arg)
    return hashlib.sha256(out_str.encode('utf-8')).hexdigest()[:12]


class Graph:
    def __init__(self, children_of):
        self.nodes = {}
        self.max_node_id = 0

        # initialize each node
        for node_name, children_names in children_of.items():
            node = self.add_node(node_name)
            for child_name in children_names:
                child_node = self.add_node(child_name)
                node.add_child(child_node)
                child_node.add_parent(node)

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
            node.add_reachable_nodes(visited)

    def get_num_nodes(self):
        return len(self.nodes)

    def add_node(self, node_name):
        if node_name in self.nodes.keys():
            return self.nodes[node_name]
        self.max_node_id += 1
        node = Node(node_name, self.max_node_id, random.randint(*EXECUTION_TIME_RANGE), SCHEDULING_OVERHEAD)
        self.nodes[node_name] = node

        return node

    def remove_node(self, node_name):
        node = self.nodes[node_name]

        def dfs(node):
            for parent in node.parents:
                parent.remove_reachable_node(node)
                dfs(parent)
        dfs(node)

        def dfs(node):
            for child in node.children:
                child.remove_reachable_node(node)
                dfs(child)
        dfs(node)

        # remove the node
        for parent in node.parents:
            parent.remove_child(node)
        for child in node.children:
            child.remove_parent(node)
    
        del self.nodes[node_name]

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
            elif label == 'name':
                dot.node(node.hash_name, node.name, style="filled", fillcolor=color)
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
    def __init__(self, cores=0):
        self.nodes = set()
        self.cores = cores
        self.children = []
        self.pending_nodes = set()

    def add_node(self, node):
        self.nodes.add(node)

        for child in node.children:
            if child in self.nodes or child in self.pending_nodes:
                continue
            child_children_depth = mean([cc.depth for cc in child.children]) if child.children else 0
            heappush(self.children, (child_children_depth, id(child), child))

    def consider_child(self):
        if not self.children:
            return None
        return heappop(self.children)[2]

    def remove_node(self, node):
        self.nodes.remove(node)

    def remove_nodes(self, nodes):
        self.nodes.difference_update(nodes)

    # the closured nodes cannot be inter-reached from the nodes outside the closure
    def get_pending_nodes(self, new_node):
        self.pending_nodes = set([new_node])

        visited = set()

        def dfs(current_node):
            if current_node in visited:
                return
            visited.add(current_node)

            # skip if this node can not reach one of the end nodes
            if not (current_node.reachable_nodes & (self.nodes | self.pending_nodes)):
                return

            self.pending_nodes.add(current_node)

            # consider the parent: the closure should not depend on uncompleted nodes
            #for parent_node in current_node.parents:
            #    if parent_node not in self.nodes and not parent_node.group:
            #        dfs(parent_node)
#
            for child_node in current_node.children:
                dfs(child_node)

        # the new node can reach other nodes, while other nodes can reach the new node
        for n in self.nodes | set([new_node]):
            dfs(n)

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
        self.pending_nodes = set()
        for node in self.nodes:
            node.group = self

    def revert_pending_nodes(self):
        self.pending_nodes = set()
    
    def get_critical_time(self):
        return max([node.end_time for node in self.nodes | self.pending_nodes])
    
    def get_resource_utilization(self):
        return sum([node.execution_time for node in self.nodes | self.pending_nodes]) / (self.get_critical_time() * self.cores)

    def schedule_to_cores(self):
        if not self.nodes:
            return
        
        group_nodes = self.nodes.copy() | self.pending_nodes.copy()

        num_available_cores = self.cores

        for node in group_nodes:
            node.temp_pending_parents = set()
            node.temp_pending_parents.update(node.pending_parents & group_nodes)

        ready_tasks = [node for node in group_nodes if not node.temp_pending_parents]
        waiting_tasks = [node for node in group_nodes if node.temp_pending_parents]
        running_tasks = []

        current_time = 0
        critical_time = 0

        while ready_tasks or running_tasks:
            # submit as many tasks as possible
            while ready_tasks and num_available_cores:
                ready_task = ready_tasks.pop(0)
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
                    child.temp_pending_parents.discard(finished_task)
                    if not child.temp_pending_parents:
                        ready_tasks.append(child)
                        waiting_tasks.remove(child)

        assert len(ready_tasks) == len(running_tasks) == len(waiting_tasks) == 0
        assert max([task.end_time for task in group_nodes]) == critical_time

        return critical_time

    def visualize(self):
        self.schedule_to_cores()

        fig, ax = plt.subplots(figsize=(10, 6))
        group_nodes = sorted(self.nodes, key=lambda task: task.start_time)

        print(f"max time: {max([task.end_time for task in self.nodes])}")

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
            ax.text(task.start_time + (task.end_time - task.start_time) / 2, core_id, f'{task.id}', 
                    va='center', ha='center', color='white')

        ax.set_xlabel('Time')
        ax.set_ylabel('Core ID')
        ax.set_yticks(range(len(core_end_times)))
        ax.set_yticklabels([f'Core {i}' for i in range(len(core_end_times))])
        plt.title('Task Distribution Across Cores')
        plt.show()


class Node:
    def __init__(self, name, id, execution_time, scheduling_overhead):
        # must be initialized
        self.name = name
        self.id = id
        self.execution_time = execution_time
        self.scheduling_overhead = scheduling_overhead

        self.hash_name = hash_name(name)

        self.children = set()
        self.parents = set()

        self.pending_parents = set()
        self.temp_pending_parents = set()

        # time taken from the start of the graph to the end of this node
        self.critical_time = 0

        # the nodes that can be reached from this node
        self.reachable_nodes = set()

        self.group = None

        self.start_time = 0
        self.end_time = 0

        self.depth = 0

    def add_parent(self, parent):
        self.parents.add(parent)
        self.pending_parents.add(parent)

    def add_child(self, child):
        self.children.add(child)
    
    def remove_parent(self, parent):
        self.parents.remove(parent)

    def remove_child(self, child):
        self.children.remove(child)

    def add_reachable_node(self, node):
        self.reachable_nodes.add(node)
    
    def add_reachable_nodes(self, nodes):
        self.reachable_nodes.update(nodes)

    def remove_reachable_node(self, node):
        self.reachable_nodes.remove(node)
    
    def remove_reachable_nodes(self, nodes):
        self.reachable_nodes.difference_update(nodes)
