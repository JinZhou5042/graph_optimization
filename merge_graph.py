import random
from collections import defaultdict
from collections import deque
import json
import hashlib
from heapq import heappush, heappop
from graphviz import Digraph
import heapq
import numpy as np
import queue
import cloudpickle
from libs import Graph, Node, SCHEDULING_OVERHEAD, Group

NUM_WORKERS = 4
NUM_CORES_PER_WORKER = 4


with open("expanded_hlg.pkl", "rb") as f:
    loaded_dag = cloudpickle.load(f)
    print(loaded_dag)

with open('simplified_hlg.json', 'r') as json_in:
    loaded_children_of = json.load(json_in)
    children_of = {key: set(value) for key, value in loaded_children_of.items()}

graph = Graph(children_of)

# g.visualize('/Users/jinzhou/Downloads/original_hlg', fill_white=True)

def execute_graph(graph):

    ready_nodes = []
    for node in graph.nodes.values():
        if not node.pending_parents:
            ready_nodes.append(node)

    print(f"all nodes: {len(graph.nodes)}")

    group_id = 0
    groups = []

    while ready_nodes:
        group_id += 1
        group = Group(graph, cores=1, runtime_limit=2000, id=group_id)
        groups.append(group)

        for ready_node in ready_nodes:
            if group.get_resource_utilization() > 0.99:
                break

            new_group_nodes = group.merge_from(ready_node)

            if new_group_nodes:
                for new_group_node in new_group_nodes:
                    if new_group_node in ready_nodes:
                        ready_nodes.remove(new_group_node)
                    for new_group_node_child in new_group_node.children:
                        new_group_node_child.pending_parents.remove(new_group_node)
                        if not new_group_node_child.group and not new_group_node_child.pending_parents:
                            ready_nodes.append(new_group_node_child)

        print(len(group.nodes), group.get_critical_time(), group.get_resource_utilization())
        #group.visualize(save_to=f"/Users/jinzhou/Downloads/group_execution_{group.id}")
        #graph.visualize(f"/Users/jinzhou/Downloads/group_merged_{group.id}", label='id', draw_nodes=group.nodes)

    assert sum([len(group.nodes) for group in groups]) == len(graph.nodes)

execute_graph(graph)


# graph.visualize('/Users/jinzhou/Downloads/merged_hlg')
