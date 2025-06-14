from task_graph import TaskGraph
from function_group import FunctionGroup
import ndcctools.taskvine as vine
from utils import *
import dask
from dask.base import collections_to_dsk


class SimpleGroup:
    def __init__(self, keys, sexpr_of, input_paths, output_paths):
        self.keys = keys
        self.sexpr_of = sexpr_of

        # these keys include external parents
        self.input_paths = input_paths         # {key: input filename}
        self.output_paths = output_paths       # {key: output filename}
        
        # store results for keys in this group and its parents
        self.values = {}

class SimpleGroup:
    def __init__(self, keys, sexpr_of, input_paths, output_paths):
        self.keys = keys
        self.sexpr_of = sexpr_of

        # these keys include external parents
        self.input_paths = input_paths         # {key: input filename}
        self.output_paths = output_paths       # {key: output filename}
        
        # store results for keys in this group and its parents
        self.values = {}

def execute_converted_group(group: SimpleGroup):
    # load parent task outputs
    for k, path in group.input_paths.items():
        try:
            with open(path, 'rb') as f:
                group.values[k] = cloudpickle.load(f)
        except Exception as e:
            raise(f"Error loading input {k} from {path}: {e}")

    def rec_call(expr):
        try:
            if expr in group.values:
                return group.values[expr]
        except TypeError:
            pass
        if isinstance(expr, list):
            return [rec_call(e) for e in expr]
        if isinstance(expr, tuple) and isinstance(expr[0], str) and expr[0] in MAP_ID_TO_CALLABLE:
            fn = MAP_ID_TO_CALLABLE[expr[0]]
            return fn(*[rec_call(a) for a in expr[1:]])
        return expr

    for k in group.keys:
        result = rec_call(group.sexpr_of[k])
        group.values[k] = result

        if k in group.output_paths:
            with open(group.output_paths[k], 'wb') as f:
                cloudpickle.dump(result, f)

    return group.output_paths


class TaskGraphExecutor():
    def __init__(self, task_graph, lib_resources=None):
        self.manager = vine.Manager(
            9126,
            name="graph-optimization",
            # run_info_template="test2",
        )
        # self.manager.tune("watch-library-logfiles", 1)
        # self.manager.tune("temp-replica-count", 3)
        self.manager.tune("worker-source-max-transfers", 10000)
        self.manager.tune("max-retrievals", -1)
        MAP_ID_TO_CALLABLE = task_graph.map_id_to_callable
        self.libtask = self.manager.create_library_from_functions('library', execute_converted_group, add_env=False, hoisting_modules=[SimpleGroup, MAP_ID_TO_CALLABLE])
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
        self.ready_keys = deque([k for k in self.task_graph.task_dict.keys() if self.num_pending_parents_of_key[k] == 0])

    def create_group(self, key):
        group_keys = [key]
        group_sexpr_of = self.task_graph.get_sexpr_of_group(group_keys)

        group = SimpleGroup(
            keys=group_keys,
            sexpr_of=group_sexpr_of,
            input_paths={
                k: self.task_graph.output_filename_of[k]
                for k in self.task_graph.get_input_keys_of_group(group_keys)
            },
            output_paths={
                k: self.task_graph.output_filename_of[k]
                for k in self.task_graph.get_output_keys_of_group(group_keys)
            }
        )
        return group

    def execute(self):
        key_of_task = {}
        with create_progress_bar() as pbar:
            pbar_task = pbar.add_task("Executing task graph", total=len(self.task_graph.task_dict))
            while not pbar.tasks[pbar_task].finished:
                while self.ready_keys:
                    # submit all ready keys
                    rk = self.ready_keys.popleft()

                    # create a function call task
                    group = self.create_group(rk)
                    t = FunctionGroup('library', 'execute_converted_group', group)

                    # if rk not in self.task_graph.compute_keys:
                    # t.enable_temp_output()
                    # else:
                    #     pass

                    t.set_cores(1)
                    t.set_priority(time.time())
                    key_of_task[t] = rk

                    # add input files
                    for k in self.task_graph.get_input_keys_of_group(group.keys):
                        t.add_input(self.task_graph.output_vine_file_of[k], self.task_graph.output_filename_of[k])

                    # add output files
                    # print(f"output keys: {self.task_graph.get_output_keys_of_group(group.keys)}")
                    for k in self.task_graph.get_output_keys_of_group(group.keys):
                    # TEMP HACK: use itself as the outkey
                    # for k in group.keys:
                        f = self.manager.declare_temp()
                        t.add_output(f, self.task_graph.output_filename_of[k])
                        self.task_graph.output_vine_file_of[k] = f

                    self.manager.submit(t)

                t = self.manager.wait(5)
                if t:
                    if t.successful():
                        task_key = key_of_task[t]
                        for child in self.task_graph.children_of[task_key]:
                            self.num_pending_parents_of_key[child] -= 1
                            if self.num_pending_parents_of_key[child] == 0:
                                self.ready_keys.append(child)
                        pbar.update(pbar_task, advance=1)

                        print(f"{t.output}")
                    else:
                        print(f"failed")


tasks = load_from("tasks.pkl")
task_list = list(tasks.values())
hlg = collections_to_dsk(task_list)
assert isinstance(hlg, HighLevelGraph)

def flatten_hlg(hlg):
    assert isinstance(hlg, HighLevelGraph)
    task_dict = {}
    for k, sexpr in hlg.items():
        if isinstance(sexpr, SubgraphCallable):
            args = sexpr.args
            outkey = sexpr.outkey
            task_dict[k] = hash_name(k, outkey)
            for subkey, subexpr in sexpr.dsk.items():
                unique_key = hash_name(k, subkey)
                converted = TaskGraph.convert_expr_to_task_args(None, sexpr.dsk, subexpr, args)
                task_dict[unique_key] = converted
        elif isinstance(sexpr, tuple):
            task_dict[k] = sexpr
        elif isinstance(sexpr, (int, float, str)):
            task_dict[k] = sexpr
        else:
            raise ValueError(f"Unexpected type in HLG: {type(sexpr)}")
    return task_dict

task_dict = flatten_hlg(hlg)

task_graph = TaskGraph(task_dict, expand_dsk=False, enable_sexpr_conversion=False)

task_graph_executor = TaskGraphExecutor(task_graph, lib_resources={"cores": 20, "slots": 20})
task_graph_executor.execute()

