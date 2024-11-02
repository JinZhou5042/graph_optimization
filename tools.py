from collections import defaultdict, deque
import random
import json
import hashlib
from graphviz import Digraph
import numpy as np


def initialize_graph(children_of):
    children_of = defaultdict(set, children_of)
    parents_of = defaultdict(set)

    for node, children in children_of.items():
        for child in children:
            parents_of[child].add(node)

    for node, parents in parents_of.items():
        # leaf node
        if node not in children_of:
            children_of[node] = set()
    for node, children in children_of.items():
        # root node
        if node not in parents_of:
            parents_of[node] = set()

    return children_of, parents_of

def get_topo_order(children_of):
    visited = set()
    stack = []
    
    def dfs(u):
        if u in visited:
            return
        visited.add(u)
        for v in children_of[u]:
            dfs(v)
        stack.append(u)
        return
    
    for u in list(children_of.keys()):
        if u not in visited:
            dfs(u)

    return deque(stack[::-1])


def compute_critical_time(children_of, parents_of, execution_time):

    topo_order = get_topo_order(children_of)
    critical_time = {node: 0 for node in children_of}

    for node in topo_order:
        critical_time[node] = execution_time[node]
        if parents_of[node]:
            critical_time[node] += max(critical_time[parent] for parent in parents_of[node])

    return critical_time


def visualize_graph(children_of, filename, label_dict=None):
    print(f"saving graph to {filename}")
    dot = Digraph()
    for node in children_of:
        if not label_dict:
            dot.node(node)
        else:
            dot.node(node, str(label_dict[node]))
    for node, children in children_of.items():
        for child in children:
            dot.edge(node, child)
    dot.render(filename, format='svg', cleanup=True)

def visualize_graph_parent_of(parents_of, filename):
    dot = Digraph()
    for node in parents_of:
        dot.node(node)
    for node, parents in parents_of.items():
        for parent in parents:
            dot.edge(parent, node)
    dot.render(filename, format='svg', cleanup=True)


def hash_name(*args):
    out_str = ""
    for arg in args:
        out_str += str(arg)
    return hashlib.sha256(out_str.encode('utf-8')).hexdigest()[:12]


def connectivity_matrix(children_of):
    def dfs(children_of, start, visited):
        for neighbor in children_of.get(start, []):
            if not visited[neighbor]:
                visited[neighbor] = True
                dfs(children_of, neighbor, visited)

    nodes = list(children_of.keys())
    n = len(nodes)
    node_index = {node: idx for idx, node in enumerate(nodes)}

    reachability = np.zeros((n, n), dtype=bool)

    for node in nodes:
        visited = {n: False for n in nodes}
        dfs(children_of, node, visited)

        for target, can_reach in visited.items():
            reachability[node_index[node]][node_index[target]] = can_reach

    return node_index, reachability


def connectivity_dict(children_of):
    def dfs(children_of, start, visited):
        for neighbor in children_of.get(start, []):
            if neighbor not in visited:
                visited.add(neighbor)
                dfs(children_of, neighbor, visited)

    reachability = {node: {} for node in children_of.keys()}

    for node in children_of.keys():
        visited = set()
        dfs(children_of, node, visited)

        reachability[node] = visited

    return reachability


def replace_part_with_node(new_node, reachability, children_of_part, parents_of_part, parents_of):

    # update reachability
    reachability[new_node] = set()
    
    for child_of_part in children_of_part:
        reachability[new_node].add(child_of_part)
        reachability[new_node].update(reachability[child_of_part])

    visited = set()
    def dfs(node):
        if not node or node in visited:
            return
        visited.add(node)
        reachability[node].add(new_node)
        for parent in parents_of[node]:
            dfs(parent)
    
    for parent_of_part in parents_of_part:
        dfs(parent_of_part)



def get_part_closure1(part, children_of, reachability, execution_time):
    part_closure = part.copy()
    part_critical_time = {node: 0 for node in part}
    part_edges = 0

    visited = set()

    def dfs(start_node, current_node, end_nodes):
        if current_node in visited:
            return
        visited.add(current_node)

        for end_node in end_nodes:
            if end_node not in reachability[current_node]:
                return
            else:

                part_closure.add(current_node)
        for child in children_of[current_node]:
            dfs(start_node, child, end_nodes)

    for start_node in part:
        dfs(start_node, start_node, part - {start_node})
        
    return part_closure, max(part_critical_time.values())


def get_part_closure(part, children_of, reachability, execution_time):
    part_closure = part.copy()
    max_critical_time = 0
    visited = set()

    def dfs(start_node, current_node, end_nodes, current_time):
        if current_node in visited:
            return
        visited.add(current_node)

        # skip if this node can not reach one of the end nodes
        if not reachability[current_node] & end_nodes:
            return
        
        part_closure.add(current_node)

        nonlocal max_critical_time
        max_critical_time = max(max_critical_time, current_time)

        for child in children_of[current_node]:
            dfs(start_node, child, end_nodes, current_time + execution_time[child])

    for start_node in part:
        dfs(start_node, start_node, part, execution_time[start_node])


    print(f"part: {part}, part_closure: {part_closure}, max_critical_time: {max_critical_time}")
    return part_closure, max_critical_time
