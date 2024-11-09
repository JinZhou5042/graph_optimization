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
from libs import Graph, Node, SCHEDULING_OVERHEAD, Group

NUM_WORKERS = 4
NUM_CORES_PER_WORKER = 4


with open('hlg_simplified.json', 'r') as json_in:
    loaded_children_of = json.load(json_in)
    children_of = {key: set(value) for key, value in loaded_children_of.items()}

g = Graph(children_of)

# g.visualize('/Users/jinzhou/Downloads/original_hlg', fill_white=True)


def execute_graph(g):

    ready_nodes = deque()
    for node in g.nodes.values():
        if not node.pending_parents:
            ready_nodes.append(node)

    while ready_nodes:
        ready_node = ready_nodes.popleft()
        group = Group(cores=3, runtime_limit=100)

        group.merge_from(ready_node)

        while group.get_resource_utilization() < 0.99 and ready_nodes:
            ready_node = ready_nodes.popleft()
            if not ready_node.group:
                group.merge_from(ready_node)

            print(group.get_critical_time(), group.get_resource_utilization())

        print(group.get_critical_time(), group.get_resource_utilization())

        #g.visualize(f"/Users/jinzhou/Downloads/merged", label='id', draw_nodes=group.nodes)
        #group.visualize()
        exit(1)

execute_graph(g)


def merge_graph(g):
    print(f"size of graph: {g.get_num_nodes()}")

    topo_order = g.get_topo_order()

    while topo_order:
        # can we merge this node and all its parents?
        current_node = topo_order.popleft()

        # skip if already merged: 1. it belongs to a merged part; 2. it is a part itself
        if current_node.belongs_to_part:
            continue

        # consider those children that have not been merged yet
        part_nodes = set([current_node])
        part_children = deque()
        part_children.extendleft(current_node.children)

        part_start_time = current_node.critical_time

        while part_children:
            child_node = part_children.popleft()
            if child_node.belongs_to_part:
                continue

            # when considering a child node, we are considering the closure of the child node and the parts
            temp_part_nodes = g.get_group_closure(set([child_node, *part_nodes]))
            to_be_merged_nodes = set(temp_part_nodes) - part_nodes

            t_keep = max([node.critical_time for node in temp_part_nodes]) - part_start_time + SCHEDULING_OVERHEAD * len(to_be_merged_nodes)
            t_merge = sum([node.execution_time for node in temp_part_nodes])

            if t_merge <= t_keep:
                part_nodes.update(temp_part_nodes)
                g.merge_nodes(part_nodes)

                for added_node in to_be_merged_nodes:
                    part_children.extendleft(added_node.children)

        print(f"part_nodes: {len(part_nodes)}")


#merge_graph(g)

#g.visualize('/Users/jinzhou/Downloads/merged_hlg')
