import random
from collections import defaultdict
from collections import deque
import json
import hashlib
import sys
from heapq import heappush, heappop
import time
from graphviz import Digraph
import heapq
import ndcctools.taskvine as vine
from pprint import pprint
from tqdm import tqdm
import numpy as np
from rich import print
import queue
import cloudpickle
from libs import Graph, Node, SCHEDULING_OVERHEAD, Group, TGroup

import sys


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
    group_parents = group.get_parents()

    def is_key(s, nodes):
        try:
            hash(s)
            if s in [node.key for node in nodes]:
                return True
            else:
                return False
        except TypeError:
            return False
        
    input_files = {group_parent.result_file: None for group_parent in group.get_parents()}
    
    for input_file in input_files.keys():
        if not input_file:
            continue
        assert isinstance(input_file, vine.File)

        try:
            input_files[input_file] = input_file.contents(cloudpickle.load)['Result']
        except Exception as e:
            print(f"Error: {e}")
            raise

    def rec_call(sexpr):
        if is_key(sexpr, group_parents):
            if sexpr in input_files.keys():
                return input_files[sexpr]
        elif is_key(sexpr, group.nodes):
            for n in group.nodes:
                if sexpr == n.key:
                    return n.result_file
                
        if isinstance(sexpr, list):
            return [rec_call(item) for item in sexpr]
        if isinstance(sexpr, tuple) and callable(sexpr[0]):
            return sexpr[0](*[rec_call(item) for item in sexpr[1:]])
        else:
            return sexpr

    num_pending_parents = {task: 0 for task in group.nodes}
    for task in group.nodes:
        for parent in task.parents:
            if parent in group.nodes:
                num_pending_parents[task] += 1

    ready_tasks = [task for task in group.nodes if num_pending_parents[task] == 0]
    waiting_tasks = [task for task in group.nodes if num_pending_parents[task] != 0]
    running_tasks = []

    while ready_tasks or running_tasks:
        # run a task
        task = ready_tasks.pop(0)
        task.result_file = rec_call(task.sexpr)
        # submit more tasks
        for child in task.children:
            if child in waiting_tasks:
                num_pending_parents[child] -= 1
                if num_pending_parents[child] == 0:
                    ready_tasks.append(child)
                    waiting_tasks.remove(child)

    return group


# g.visualize('/Users/jinzhou/Downloads/original_hlg', fill_white=True)

def execute_graph(graph):

    ready_keys = []
    for k, n in graph.nodes.items():
        if not n.pending_parents:
            ready_keys.append(k)

    print(f"all nodes: {len(graph.nodes)}")

    group_id = 0
    groups = []

    q = vine.Manager([9123, 9128], name="graph-optimization")
    libtask = q.create_library_from_functions('dask-library', execute_group, add_env=False)
    q.install_library(libtask)


    while ready_keys:
        group_id += 1
        group = Group(graph, cores=1, runtime_limit=5000, id=group_id)
        groups.append(group)

        for ready_key in ready_keys:
            if group.get_resource_utilization() > 0.99:
                break

            new_keys = group.merge_from(ready_key)

            if new_keys:
                for new_key in new_keys:
                    if new_key in ready_keys:
                        ready_keys.remove(new_key)
                    for new_key_child in graph.nodes[new_key].children:
                        graph.nodes[new_key_child].pending_parents.remove(new_key)
                        if not graph.nodes[new_key_child].group and not graph.nodes[new_key_child].pending_parents:
                            ready_keys.append(new_key_child)

        print(len(group.keys), group.get_critical_time(), group.get_resource_utilization())
        #group.visualize(save_to=f"/Users/jinzhou/Downloads/group_execution_{group.id}")
        #graph.visualize(f"/Users/jinzhou/Downloads/group_merged_{group.id}", label='id', draw_nodes=group.nodes)
        #enqueue_group(group)
        # execute_group(group)

        # print(group.nodes)
        # group.nodes = None



    assert sum([len(group.keys) for group in groups]) == len(graph.nodes)

execute_graph(graph)


# graph.visualize('merged_hlg')
