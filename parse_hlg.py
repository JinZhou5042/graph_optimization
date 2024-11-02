import json
from graphviz import Digraph
import re
import hashlib
import tqdm
from collections import defaultdict
from collections import deque
import argparse


def hash_name(*args):
    out_str = ""
    for arg in args:
        out_str += str(arg)
    return hashlib.sha256(out_str.encode('utf-8')).hexdigest()[:12]

def sanitize_label(label):
    if isinstance(label, str):
        clean_label = re.sub(r'[<>]', '', label).replace(' ', '-')
        return clean_label
    else:
        return str(label).replace("<", "").replace(">", "")

def parse_node(json_data, node_key, node_value, parent_of, children_of):
    if isinstance(node_value, str):
        if node_value in json_data:
            parent_of[node_key].add(node_value)
            children_of[node_value].add(node_key)
    elif isinstance(node_value, list):
        for element in node_value:
            parse_node(json_data, node_key, element, parent_of, children_of)


def topological_sort(children_of):
    in_degree = defaultdict(int)
    for node, children in children_of.items():
        for child in children:
            in_degree[child] += 1

    zero_in_degree = deque([node for node in children_of if in_degree[node] == 0])
    topo_order = []

    while zero_in_degree:
        node = zero_in_degree.popleft()
        topo_order.append(node)
        for child in children_of[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                zero_in_degree.append(child)

    return topo_order


def merge_linear_nodes_topological(parent_of, children_of):
    visited_nodes = set()

    topo_order = topological_sort(children_of)

    for node in topo_order:
        if node in visited_nodes:
            continue

        visited_nodes.add(node)

        current_chain = [node]

        while (children_of_chain := list(children_of[current_chain[-1]])) and len(children_of_chain) == 1 and (child := children_of_chain[0]):
            if len(parent_of[child]) == 1:
                current_chain.append(child)
                visited_nodes.add(child)
            else:
                break

        merged_node_name = hash_name(tuple(current_chain))

        first, last = current_chain[0], current_chain[-1]
        for parent in parent_of[first]:
            if first in children_of[parent]:
                children_of[parent].remove(first)
            children_of[parent].add(merged_node_name)
            parent_of[merged_node_name].add(parent)
        for child in children_of[last]:
            if last in parent_of[child]:
                parent_of[child].remove(last)
            parent_of[child].add(merged_node_name)
            children_of[merged_node_name].add(child)

        # remove the nodes
        for n in current_chain:
            if n != merged_node_name:
                parent_of.pop(n, None)
                children_of.pop(n, None)

    return parent_of, children_of



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse HLG to DAG')
    parser.add_argument('--label', type=str, help='Node label')
    args = parser.parse_args()

    json_file = 'expanded_hlg.json'
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    
    # parse the DAG dependencies
    parent_of = defaultdict(lambda: set())
    children_of = defaultdict(lambda: set())

    for node_key, node_value in json_data.items():
        parse_node(json_data, node_key, node_value, parent_of, children_of)

    # original_num_nodes = len(parent_of)

    # eliminate linear dependencies
    # parent_of, children_of = merge_linear_nodes_topological(parent_of, children_of)
    # print(f"eliminated linear dependencies, {len(parent_of)}/{original_num_nodes} nodes left")


    with open('hlg_simplified.json', 'w') as json_out:
        json.dump({key: list(value) for key, value in children_of.items()}, json_out, indent=4)

    # render the DAG
    pbar = tqdm.tqdm(children_of.items(), desc="Rendering DAG")
    dot = Digraph(comment='DAG', format='svg')
    for parent, children in children_of.items():
        pbar.update(1)
        dot.node(parent, sanitize_label(parent))
        for child in children:
            dot.node(child, sanitize_label(child))
            dot.edge(parent, child)
    pbar.close()

    output_file = '/Users/jinzhou/Downloads/expanded_hlg'
    dot.render(output_file, format='svg', view=False, cleanup=True)
    print(f"Saved DAG to {output_file}.svg")


    exit(1)
    # =================================================================

    # duplicate the nodes with multiple children
    n21_children_of = defaultdict(lambda: set())
    pbar = tqdm.tqdm(children_of.items(), desc="Duplicating nodes")
    for task, children in children_of.items():
        pbar.update(1)
        if len(children) > 1:        # if one task has multiple children, duplicate the task
            for index, child in enumerate(children):
                copied_task = f"{task}_copy_{index}"
                n21_children_of[copied_task] = [child]
        else:
            n21_children_of[task] = children
    pbar.close()

    #parent_of, n21_children_of = merge_linear_nodes_topological(parent_of, n21_children_of)

    # render the DAG
    pbar = tqdm.tqdm(n21_children_of.items(), desc="Rendering DAG")
    dot_2 = Digraph(comment='DAG', format='svg')
    for parent, children in n21_children_of.items():
        pbar.update(1)
        dot_2.node(parent, sanitize_label(parent))
        for child in children:
            dot_2.node(child, sanitize_label(child))
            dot_2.edge(parent, child)
        if len(children) == 1:
            dot_2.node(parent, sanitize_label(parent))
            print(f"Node {parent} has no children")
    pbar.close()



    output_file = '/Users/jinzhou/Downloads/expanded_hlg_duplicated'
    dot_2.render(output_file, format='svg', view=False, cleanup=True)
    print(f"Saved DAG to {output_file}.svg")
