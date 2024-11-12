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
import ndcctools.taskvine as vine
import cloudpickle
from libs import Graph, Node, SCHEDULING_OVERHEAD, Group

NUM_WORKERS = 4
NUM_CORES_PER_WORKER = 4

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
    input_files = {group_parent.result_file: None for group_parent in group.get_parents()}
    
    for input_file in input_files.keys():
        assert isinstance(input_file, vine.File)

        try:
            input_files[input_file] = input_file.contents(cloudpickle.load)['Result']
        except Exception as e:
            print(f"Error: {e}")
            raise

    def rec_call(sexpr):
        if group.graph.is_key(sexpr):
            pass
    


def enqueue_group(group):
    print(f"group size: {len(group.nodes)}")


    group_parents = group.get_parents()

    
    t = vine.FunctionCall('dask-library', 'execute_group', group)

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
        group = Group(graph, cores=1, runtime_limit=5000, id=group_id)
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
        #enqueue_group(group)

        #exit(1)


    assert sum([len(group.nodes) for group in groups]) == len(graph.nodes)

execute_graph(graph)


# graph.visualize('merged_hlg')
