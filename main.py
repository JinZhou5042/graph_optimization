from task_graph_definition import TaskGraph
import ndcctools.taskvine as vine
from utils import *
import dask
import cloudpickle
from dask.base import collections_to_dsk
import collections
from ndcctools.taskvine import FunctionCall
from ndcctools.taskvine.utils import load_variable_from_library

def load_task_graph():
    if not os.path.exists("task_graph.pkl"):
        raise FileNotFoundError("task_graph.pkl not found")

    with open("task_graph.pkl", 'rb') as f:
        task_graph = cloudpickle.load(f)
    return task_graph

def init_task_graph_context():
    import cloudpickle

    with open("task_graph.pkl", 'rb') as f:
        task_graph = cloudpickle.load(f)

    return {
        'task_graph': task_graph,
    }

def compute_group_keys(keys):
    task_graph = load_variable_from_library('task_graph')

    input_paths = task_graph.get_group_input_paths(keys)
    output_paths = task_graph.get_group_output_paths(keys)
    values = {}

    for k, path in input_paths.items():
        try:
            with open(path, 'rb') as f:
                values[k] = cloudpickle.load(f)
        except Exception as e:
            raise Exception(f"Error loading input {k} from {path}: {e}")

    def rec_call(expr):
        try:
            if expr in values:
                return values[expr]
        except TypeError:
            pass
        if isinstance(expr, list):
            return [rec_call(e) for e in expr]
        if isinstance(expr, tuple) and isinstance(expr[0], str) and expr[0] in task_graph.map_id_to_callable:
            fn = task_graph.map_id_to_callable[expr[0]]
            return fn(*[rec_call(a) for a in expr[1:]])
        return expr

    for k in keys:
        result = rec_call(task_graph.task_dict[k])
        values[k] = result

    for k, path in output_paths.items():
        with open(path, 'wb') as f:
            cloudpickle.dump(values[k], f)

    return True

class TaskGraphExecutor():
    def __init__(self, task_graph, lib_resources=None, prune_depth=1):
        self.prune_depth = prune_depth

        self.manager = vine.Manager(
            9126,
            name="graph-optimization",
            # run_info_template="test2",
        )

        # self.manager.tune("watch-library-logfiles", 1)
        # self.manager.tune("temp-replica-count", 3)
        self.manager.tune("worker-source-max-transfers", 10000)
        self.manager.tune("max-retrievals", -1)
        self.manager.tune("prefer-dispatch", 1)
        self.manager.tune("wait-for-workers", 20)

        hoisting_modules = [timed, os, uuid, types, cloudpickle, hashlib, collections, TaskGraph, load_variable_from_library]
        self.libtask = self.manager.create_library_from_functions('library',
                                                                  compute_group_keys,
                                                                  library_context_info=[init_task_graph_context, [], {}],
                                                                  add_env=False,
                                                                  hoisting_modules=hoisting_modules)
        self.libtask.add_input(self.manager.declare_file("task_graph.pkl"), "task_graph.pkl")
        self.libtask.add_input(self.manager.declare_file("task_graph_definition.py"), "task_graph_definition.py")
        self.libtask.add_input(self.manager.declare_file("utils.py"), "utils.py")
        self.libtask.add_input(self.manager.declare_file("function_group.py"), "function_group.py")

        if lib_resources:
            if 'cores' in lib_resources:
                self.libtask.set_cores(lib_resources['cores'])
                self.libtask.set_function_slots(lib_resources['cores'])
            if 'memory' in lib_resources:
                self.libtask.set_memory(lib_resources['memory'])
            if 'disk' in lib_resources:
                self.libtask.set_disk(lib_resources['disk'])
            if 'slots' in lib_resources:
                self.libtask.set_function_slots(lib_resources['slots'])
        self.manager.install_library(self.libtask)

        self.task_graph = task_graph
        self.num_pending_parents_of_key = {k: len(self.task_graph.parents_of[k]) for k in self.task_graph.task_dict.keys()}
        self.num_pending_children_of_key = {k: len(self.task_graph.children_of[k]) for k in self.task_graph.task_dict.keys()}

        self.ready_keys = deque([k for k in self.task_graph.task_dict.keys() if self.num_pending_parents_of_key[k] == 0])

    def execute(self):
        key_of_task = {}
        with create_progress_bar() as pbar:
            pbar_task = pbar.add_task("Executing task graph", total=len(self.task_graph.task_dict))
            while not pbar.tasks[pbar_task].finished:
                while self.ready_keys:
                    # submit all ready keys
                    rk = self.ready_keys.popleft()
                    keys = [rk]

                    t = FunctionCall('library', 'compute_group_keys', keys)

                    t.set_cores(1)
                    key_of_task[t] = rk

                    # add input files
                    for k in self.task_graph.get_input_keys_of_group(keys):
                        t.add_input(self.task_graph.output_vine_file_of[k], self.task_graph.output_filename_of[k])

                    # add output files
                    for k in keys:
                        worker_transfers = True
                        if worker_transfers:
                            t.enable_temp_output()
                            f = self.manager.declare_temp()
                        else:
                            output_path = os.path.join(self.manager.staging_directory, "outputs", self.task_graph.output_filename_of[k])
                            f = self.manager.declare_file(output_path, unlink_when_done=False, cache="workflow")
                        t.add_output(f, self.task_graph.output_filename_of[k])
                        self.task_graph.output_vine_file_of[k] = f

                    self.manager.submit(t)

                t = self.manager.wait(5)
                if t:
                    if t.successful():
                        pbar.update(pbar_task, advance=1)
                        
                        task_key = key_of_task[t]

                        # enqueue new tasks
                        for child in self.task_graph.children_of[task_key]:
                            self.num_pending_parents_of_key[child] -= 1
                            if self.num_pending_parents_of_key[child] == 0:
                                self.ready_keys.append(child)

                        # prune stale files
                        for parent in self.task_graph.parents_of[task_key]:
                            self.num_pending_children_of_key[parent] -= 1
                            if self.num_pending_children_of_key[parent] == 0:
                                self.manager.prune_file(self.task_graph.output_vine_file_of[parent])
                    else:
                        print(f"failed")
                        time.sleep(1)

                    try:
                        with open("output.txt", "a") as f:
                            f.write(f"{t.output}\n")
                    except Exception as e:
                        pass


def main(checkpoint=False, load=False, expand=False, prune_depth=1):
    # tasks = load_from("tasks.pkl")
    tasks = load_from("large_tasks.pkl")
    task_list = list(tasks.values())
    hlg = collections_to_dsk(task_list)
    task_dict = flatten_hlg(hlg)

    if load:
        task_graph = load_task_graph()
    else:
        task_graph = TaskGraph(task_dict, expand_dsk=expand, enable_sexpr_conversion=False)

    if checkpoint:
        with open("task_graph.pkl", 'wb') as f:
            cloudpickle.dump(task_graph, f)

    task_graph_executor = TaskGraphExecutor(task_graph, lib_resources={"cores": 16, "slots": 16}, prune_depth=prune_depth)
    task_graph_executor.execute()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--expand", action="store_true")
    parser.add_argument("--prune-depth", type=int, default=1)
    args = parser.parse_args()

    assert not (args.checkpoint and args.load)

    main(args.checkpoint, args.load, args.expand, args.prune_depth)