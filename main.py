from ndcctools.taskvine.graph_executor import GraphExecutor

import analysis_tools
import cloudpickle
import argparse

def make_binary_forest(branches=1, level=1):
    assert level >= 1 and branches >= 1
    task_dict = {}
    finals = []
    add = lambda *args: sum(args)
    const = lambda x: x
    for b in range(branches):
        leaves = [f"B{b}_L{i}" for i in range(2**(level-1))]
        for leaf in leaves:
            task_dict[leaf] = (const, 1)
        cur = leaves
        for d in range(level-1):
            nxt = []
            for i in range(0, len(cur), 2):
                l, r = cur[i], cur[i+1]
                p = f"B{b}_N{d}_{i//2}"
                task_dict[p] = (add, l, r)
                nxt.append(p)
            cur = nxt
        f = f"B{b}_final"
        task_dict[f] = (add, cur[0])
        finals.append(f)
    task_dict["merge"] = (add, *finals)
    task_dict["output"] = (add, "merge")
    return task_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manager-name", type=str, default="default-manager")
    parser.add_argument("--task-file", type=str, default=None)
    parser.add_argument("--libcores", type=int, default=16)
    parser.add_argument("--prune-depth", type=int, default=0)
    parser.add_argument("--temp-replica-count", type=int, default=1)
    parser.add_argument("--wait-for-workers", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=-1)
    parser.add_argument("--watch-library-logfiles", type=int, default=0)
    parser.add_argument("--enforce-worker-eviction-interval", type=int, default=-1)
    parser.add_argument("--priority-mode", type=str, default="random")
    parser.add_argument("--scheduling-mode", type=str, default="files")
    parser.add_argument("--run-info-path", type=str, default="/users/jzhou24/afs/taskvine-report-tool/logs")
    parser.add_argument("--run-info-template", type=str, default="test")
    parser.add_argument("--balance-worker-disk-load", type=int, default=0)
    parser.add_argument("--pfs-percentage", type=float, default=0.0)
    parser.add_argument("--staging-dir", type=str, default="/project01/ndcms/jzhou24/staging")
    parser.add_argument("--shared-file-system-dir", type=str, default="/project01/ndcms/jzhou24/shared_file_system")
    parser.add_argument("--extra-task-output-size-mb", type=int, nargs=2, default=[0, 0])
    parser.add_argument("--extra-task-sleep-time",type=int, nargs=2, default=[0, 0])
    parser.add_argument("--failure-injection-step-percent", type=float, default=-1)
    args = parser.parse_args()

    executor = GraphExecutor(
        [9100, 9199],
        name=args.manager_name,
        hoisting_modules=[analysis_tools],
        run_info_path=args.run_info_path,
        run_info_template=args.run_info_template,
        temp_replica_count=args.temp_replica_count,
        wait_for_workers=args.wait_for_workers,
        watch_library_logfiles=args.watch_library_logfiles,
        max_workers=args.max_workers,
        enforce_worker_eviction_interval=args.enforce_worker_eviction_interval,
        libcores=args.libcores,
        libtask_env_files={
            "./analysis_tools/dataset_info.py" : "analysis_tools/dataset_info.py",
            "./analysis_tools/signal_info.py" : "analysis_tools/signal_info.py",
            "./analysis_tools/storage_config.py" : "analysis_tools/storage_config.py",
            "./analysis_tools/coffea_utils.py" : "analysis_tools/coffea_utils.py",
            "./analysis/variables.py" : "analysis/variables.py",
            "./analysis/selections.py" : "analysis/selections.py",
            "./analysis/calculations.py" : "analysis/calculations.py",
            "./analysis/plotting.py" : "analysis/plotting.py",
        },
    )

    if args.task_file:
        with open(args.task_file, "rb") as f:
            collection_dict = cloudpickle.load(f)
    else:
        collection_dict = make_binary_forest(branches=100, level=12)

    results = executor.run(
                collection_dict,
                # target_keys=["output"],
                priority_mode=args.priority_mode,
                scheduling_mode=args.scheduling_mode,
                balance_worker_disk_load=args.balance_worker_disk_load,
                prune_depth=args.prune_depth,
                staging_dir=args.staging_dir,
                extra_task_output_size_mb=args.extra_task_output_size_mb,
                extra_task_sleep_time=args.extra_task_sleep_time,
                shared_file_system_dir=args.shared_file_system_dir,
                failure_injection_step_percent=args.failure_injection_step_percent,
                outfile_type={
                    "temp": 1 - args.pfs_percentage,
                    "shared-file-system": args.pfs_percentage,
                },
            )
    
    if results:
        print(f"results = {results}")

if __name__ == "__main__":
    main()
