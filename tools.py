import random
from collections import defaultdict
from collections import deque
import json
import hashlib
from graphviz import Digraph
import numpy as np
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

EXECUTION_TIME_RANGE = (10, 100)
SCHEDULING_OVERHEAD = 400000000
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
            else:
                node.critical_time = max([parent.critical_time for parent in node.parents]) + node.execution_time

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
    
    # the closured nodes cannot be inter-reached from the nodes outside the closure
    def get_part_nodes_closure(self, part_nodes):
        part_nodes_closure = part_nodes.copy()
        visited = set()

        def dfs(current_node):
            if current_node in visited:
                return
            visited.add(current_node)

            # skip if this node can not reach one of the end nodes
            if not current_node.reachable_nodes & part_nodes:
                return

            part_nodes_closure.add(current_node)

            for child_node in current_node.children:
                dfs(child_node)

        for part_node in part_nodes:
            dfs(part_node)

        return part_nodes_closure
    
    def merge_nodes(self, part_nodes):
        part_topo_order = self.get_topo_order(nodes=part_nodes)
        for i, part_node in enumerate(part_topo_order):
            part_node.belongs_to_part = part_nodes
            if i >= 1:
                part_node.critical_time = part_topo_order[i - 1].critical_time + part_node.execution_time

        if len(part_nodes) >= 3:
            # update the downstream critical time
            downstream_topo_order = self.get_topo_order(start_nodes=part_nodes)
            downstream_topo_order = deque(set(downstream_topo_order) - part_nodes)

            for node in downstream_topo_order:
                node.critical_time = max([parent.critical_time for parent in node.parents]) + node.execution_time
    
    def visualize(self, filename="graph", label='id', fill_white=False):
        dot = Digraph()

        for node in self.nodes.values():
            if fill_white:
                color = 'white'
            else:
                hash_object = hashlib.sha256(str(node.belongs_to_part).encode())
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
                dot.node(node.hash_name, node.execution_time, style="filled", fillcolor=color)
            else:
                raise ValueError(f"Unknown label: {label}")
        for node in self.nodes.values():
            for parent_node in node.parents:
                dot.edge(parent_node.hash_name, node.hash_name)

        dot.render(filename, format='svg', cleanup=True)        


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

        # time taken from the start of the graph to the end of this node
        self.critical_time = 0

        # the nodes that can be reached from this node
        self.reachable_nodes = set()

        # this node is merged with other nodes
        self.belongs_to_part = set()

    def set_parents(self, parents):
        self.parents = parents

    def set_children(self, children):
        self.children = children

    def add_parent(self, parent):
        self.parents.add(parent)

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
