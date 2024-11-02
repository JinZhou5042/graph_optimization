import random
from collections import defaultdict
from collections import deque
import json
from tools import *


with open('hlg_simplified.json', 'r') as json_in:
    loaded_children_of = json.load(json_in)
    children_of = {key: set(value) for key, value in loaded_children_of.items()}

children_of, parents_of = initialize_graph(children_of)

e_min = 10
e_max = 100
execution_time = defaultdict(lambda: random.randint(e_min, e_max), {node: random.randint(e_min, e_max) for node in children_of})
s = 30    # scheduling overhead
c = 10    # communication overhead


critical_time = compute_critical_time(children_of, parents_of, execution_time)

print(f"Critical Path Length of the entire graph: {max(critical_time.values())}")
merge_mapping = {}


def merge_graph(children_of):
    reachability = connectivity_dict(children_of)

    print(f"size of graph: {len(children_of)}")
    visited = set()
    has_update = False

    topo_order = get_topo_order(children_of)

    while topo_order:
        # can we merge this node and all its parents?
        current_node = topo_order.popleft()

        parents = parents_of[current_node].copy()
        children = children_of[current_node].copy()

        # skip if already visited
        if current_node in visited:
            continue
        visited.add(current_node)

        # skip if this node has no parents
        if not parents:
            continue
        part = parents.copy()
        part.add(current_node)
        part, part_max_critical_time = get_part_closure(part, children_of, reachability, execution_time)

        can_be_parallelized = False   # this node can finish before all parents finish
        for p in parents:
            if critical_time[p] >= critical_time[current_node]:
                can_be_parallelized = True
        if can_be_parallelized:
            t_keep = 0
        else:
            t_keep = max(execution_time[n] for n in parents) + execution_time[current_node] + len(part) * s + len(parents) * c


        t_merge = sum(execution_time[n] for n in part) + s


        if t_merge <= t_keep:
            has_update = True

            visited.update(parents)

            merged_node_name = hash_name(part)
            merge_mapping[merged_node_name] = part

            # print(f"current node: {current_node}, children: {children}, merged_node_name: {merged_node_name}")

            execution_time[merged_node_name] = sum(execution_time[n] for n in part)
            critical_time[merged_node_name] = max(critical_time[n] for n in part)

            # the merging affects the critical time of all downstream nodes

            # redirect the parents and children of the merged part
            parents_of_part = set()
            children_of_part = set()
            for n in part:
                parents_of_part.update(parents_of[n])
                children_of_part.update(children_of[n])
            parents_of_part -= part
            children_of_part -= part

            for parent_of_part in parents_of_part:
                children_of[parent_of_part] -= part
                children_of[parent_of_part].add(merged_node_name)

            for child_of_part in children_of_part:
                parents_of[child_of_part] -= part
                parents_of[child_of_part].add(merged_node_name)

            children_of[merged_node_name] = children_of_part
            parents_of[merged_node_name] = parents_of_part

            # remove the part from the graph
            for n in part:
                del children_of[n]
                del parents_of[n]
                del execution_time[n]
                del critical_time[n]
                del reachability[n]

            print(f"merged_node_name: {merged_node_name}")

            replace_part_with_node(merged_node_name, reachability, children_of_part, parents_of_part, parents_of)

            # update the reachability

            # add the merged part to the graph
            children_of[merged_node_name] = set(children_of_part)
            parents_of[merged_node_name] = set(parents_of_part)

            # add to the topo_order
            topo_order.appendleft(merged_node_name)

        else:
            # keep as is
            pass

    print(f"size of merged graph: {len(children_of)}")
    return has_update


while True:

    if not merge_graph(children_of):
        break

merge_graph(children_of)


output_file = '/Users/jinzhou/Downloads/merged_hlg'
visualize_graph(children_of, output_file, critical_time)