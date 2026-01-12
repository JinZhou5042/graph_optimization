from ndcctools.taskvine import DaskVine

from ndcctools.taskvine.dagvine import DAGVine
from ndcctools.taskvine.dagvine.blueprint_graph.adaptor import collections_from_blueprint_graph

import analysis_tools

import cloudpickle
import argparse
import time

from graph_generator import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manager-name", "-M", type=str, default=None)
    parser.add_argument("--task-file", type=str, default=None)
    parser.add_argument("--libcores", type=int, default=16)
    parser.add_argument("--prune-depth", type=int, default=0)
    parser.add_argument("--temp-replica-count", type=int, default=1)
    parser.add_argument("--immediate-recovery", type=int, default=0)
    parser.add_argument("--wait-for-workers", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=-1)
    parser.add_argument("--watch-library-logfiles", type=int, default=0)
    parser.add_argument("--enforce-worker-eviction-interval", type=int, default=-1)
    parser.add_argument("--task-priority-mode", type=str, default="random")
    parser.add_argument("--checkpoint-fraction", type=float, default=0)
    parser.add_argument("--scheduling-mode", type=str, default="files")
    parser.add_argument("--run-info-path", type=str, default="/users/jzhou24/afs/taskvine-report-tool/logs/project-dagvine")
    parser.add_argument("--run-info-template", type=str, default="test")
    parser.add_argument("--shift-disk-load", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="/project01/ndcms/jzhou24/outputs")
    parser.add_argument("--checkpoint-dir", type=str, default="/project01/ndcms/jzhou24/checkpoints")
    parser.add_argument("--prioritize-recovery-tasks", type=int, default=0)
    parser.add_argument("--clean-redundant-replicas", type=int, default=0)
    parser.add_argument("--extra-task-output-size-mb", type=str, nargs=2, default=["0", "0"])
    parser.add_argument("--extra-task-sleep-time",type=str, nargs=2, default=["0", "0"])
    parser.add_argument("--failure-injection-step-percent", type=float, default=-1)
    parser.add_argument("--graph-from-log", type=str, default=None)
    parser.add_argument("--executor", "-X", type=str, default="dagvine")
    parser.add_argument("--graph", "-G", type=str, default=None)
    parser.add_argument("--enable-debug-log", type=int, default=1)
    parser.add_argument("--adapt-dask", type=int, default=False)
    parser.add_argument("--auto-recovery", type=int, default=1)
    parser.add_argument("--max-retry-attempts", type=int, default=3)
    parser.add_argument("--retry-interval-sec", type=int, default=0)
    args = parser.parse_args()

    # first remove the existing run info template
    import os
    import shutil
    run_info_template_full = os.path.join(args.run_info_path, args.run_info_template)
    if os.path.exists(run_info_template_full):
        shutil.rmtree(run_info_template_full)
        print(f"Removed existing run info template: {run_info_template_full}")

    hoisting_modules = [analysis_tools]
    try:
        import coffea
        import coffea.processor.taskvine_executor
        import analysis_processor
        hoisting_modules.append(analysis_processor)
        hoisting_modules.append(coffea, coffea.processor.taskvine_executor)
    except:
        pass

    if args.task_file:
        with open(args.task_file, "rb") as f:
            graph = cloudpickle.load(f)
    elif args.graph:
        if args.graph == "chain":
            graph = make_chain_graph(branches=1, chain_len=999)
        elif args.graph == "binary-forest":
            graph = make_binary_forest(branches=50, level=15)
        elif args.graph == "individuals":
            graph = make_individuals(num_tasks=1000)
        elif args.graph == "simple":
            graph = make_simple_graph()
        elif args.graph == "dv5-small":
            with open("task-graph-pkls/dv5-small-converted.pkl", "rb") as f:
                graph = cloudpickle.load(f)
        elif args.graph == "dv5-large":
            with open("task-graph-pkls/dv5-large-converted.pkl", "rb") as f:
                graph = cloudpickle.load(f)
        elif args.graph == "rstriphoton":
            with open("task-graph-pkls/rstriphoton-converted.pkl", "rb") as f:
                graph = cloudpickle.load(f)
        elif args.graph == "topeft":
            with open("task-graph-pkls/topeft-converted.pkl", "rb") as f:
                graph = cloudpickle.load(f)

    if not args.manager_name:
        manager_name = f"jzhou24-{args.graph}"
    else:
        manager_name = args.manager_name

    results = None

    time_start = time.time()

    if args.executor == "dagvine":
        dagvine = DAGVine(
            [9100, 9199],
            name=manager_name,
            run_info_path=args.run_info_path,
            run_info_template=args.run_info_template,
        )
        params = {
            "temp-replica-count": args.temp_replica_count,
            "wait-for-workers": args.wait_for_workers,
            "watch-library-logfiles": args.watch_library_logfiles,
            "immediate-recovery": args.immediate_recovery,
            "max-workers": args.max_workers,
            "enforce-worker-eviction-interval": args.enforce_worker_eviction_interval,
            "shift-disk-load": args.shift_disk_load,
            "libcores": args.libcores,
            "failure-injection-step-percent": args.failure_injection_step_percent,
            "prune-depth": args.prune_depth,
            "checkpoint-dir": args.checkpoint_dir,
            "extra-task-output-size-mb": [float(x) for x in args.extra_task_output_size_mb],
            "extra-task-sleep-time": [float(x) for x in args.extra_task_sleep_time],
            "checkpoint-fraction": args.checkpoint_fraction,
            "output-dir": args.output_dir,
            "progress-bar-update-interval-sec": 0.5,
            "enable-debug-log": args.enable_debug_log,
            "auto-recovery": args.auto_recovery,
            "max-retry-attempts": args.max_retry_attempts,
            "retry-interval-sec": args.retry_interval_sec,
        }
        dagvine.update_params(params)
        results = dagvine.run(
            graph, 
            target_keys=["output"],
            hoisting_modules=hoisting_modules,
            env_files={
            "./analysis_tools/dataset_info.py" : "analysis_tools/dataset_info.py",
            "./analysis_tools/signal_info.py" : "analysis_tools/signal_info.py",
            "./analysis_tools/storage_config.py" : "analysis_tools/storage_config.py",
            "./analysis_tools/coffea_utils.py" : "analysis_tools/coffea_utils.py",
            "./analysis/variables.py" : "analysis/variables.py",
            "./analysis/selections.py" : "analysis/selections.py",
            "./analysis/calculations.py" : "analysis/calculations.py",
            "./analysis/plotting.py" : "analysis/plotting.py",
            "./analysis_processor.py" : "analysis_processor.py",
            },
            adapt_dask=args.adapt_dask,
        )

    else:
        if not isinstance(graph, BlueprintGraph):
            bg = BlueprintGraph()
            for k, v in graph.items():
                f, a, kwa = v
                bg.add_task(k, f, *a, **kwa)
            graph = bg
        graph = collections_from_blueprint_graph(graph)
        m = DaskVine(
            [9100, 9190],
            name=manager_name,
            run_info_path=args.run_info_path,
            run_info_template=args.run_info_template,
            staging_path="/groups/dthain/users/jzhou24",
        )
        m.tune("wait-for-workers", args.wait_for_workers)
        m.tune("enable-debug-log", args.enable_debug_log)
        m.tune("attempt-schedule-depth", 100000)
        m.tune("shift-disk-load", args.shift_disk_load)
        m.tune("clean-redundant-replicas", args.clean_redundant_replicas)
        m.tune("max-retrievals", 10000000)
        m.tune("prefer-dispatch", 1)
        m.tune("worker-source-max-transfers", 1000)
        m.tune("temp-replica-count", args.temp_replica_count)
        m.tune("watch-library-logfiles", args.watch_library_logfiles)
        if "output" in graph:
            keys = ["output"]
        else:
            keys = []
        if args.executor == "daskvine-functioncall":
            results = m.get(graph, resources={"cores": 1}, keys=keys, lib_resources={'cores': args.libcores, 'slots': args.libcores}, worker_transfers=True, task_mode="function-calls", scheduling_mode=args.scheduling_mode,)
        elif args.executor == "daskvine-pythontask":
            results = m.get(graph, resources={"cores": 1}, keys=keys, worker_transfers=True, task_mode="tasks", scheduling_mode=args.scheduling_mode,)
        
    if results:
        print(f"results = {results}")

    time_end = time.time()
    print(f"Time taken: {round(time_end - time_start, 2)} seconds")

if __name__ == "__main__":
    main()
