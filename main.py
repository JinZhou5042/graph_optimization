#!/usr/bin/env python3

from ndcctools.taskvine.graph_manager import GraphManager

from dask.base import collections_to_dsk
import cloudpickle
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", type=str, default="tasks.pkl")
    parser.add_argument("--libcores", type=int, default=16)
    parser.add_argument("--expand-dsk", action="store_true", default=False)
    parser.add_argument("--nls-prune-depth", type=int, default=1)
    parser.add_argument("--nls-percentage", type=float, default=1.0)
    parser.add_argument("--checkpoint-percentage", type=float, default=0.0)
    parser.add_argument("--persistence-percentage", type=float, default=0.0)
    parser.add_argument("--temp-replica-count", type=int, default=3)
    parser.add_argument("--wait-for-workers", type=int, default=0)
    parser.add_argument("--watch-library-logfiles", type=int, default=0)
    parser.add_argument("--enforce-worker-eviction-interval", type=int, default=60)
    parser.add_argument("--priority-mode", type=str, default="largest-input-first")
    parser.add_argument("--run-info-template", type=str, default="test")
    parser.add_argument("--staging-dir", type=str, default="/project01/ndcms/jzhou24/staging")
    args = parser.parse_args()

    with open(args.task_file, "rb") as f:
        tasks = cloudpickle.load(f)
    task_list = list(tasks.values())
    task_dict = collections_to_dsk(task_list)

    gm = GraphManager(
        9124,
        name="graph-optimization",
        run_info_path="/users/jzhou24/afs/taskvine-report-tool/logs",
        run_info_template=args.run_info_template,
        nls_prune_depth=args.nls_prune_depth,
        priority_mode=args.priority_mode,
        nls_percentage=args.nls_percentage,
        checkpoint_percentage=args.checkpoint_percentage,
        persistence_percentage=args.persistence_percentage,
        temp_replica_count=args.temp_replica_count,
        wait_for_workers=args.wait_for_workers,
        watch_library_logfiles=args.watch_library_logfiles,
        staging_dir=args.staging_dir,
        enforce_worker_eviction_interval=args.enforce_worker_eviction_interval,
        libcores=args.libcores,
    )
    gm.execute(task_dict, expand_dsk=args.expand_dsk)

if __name__ == "__main__":
    main()
