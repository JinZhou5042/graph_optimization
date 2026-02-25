# graph_optimization Quick Guide

Run task-graph experiments with TaskVine / DAGVine / DaskVine.

## 1) Create the environment

From repo root:

```bash
bash create_conda_env.sh
```

What it does:
- Uses `"$HOME/miniconda3/bin/conda"` as the conda binary.
- Creates `dagvine-env` at `~/miniconda3/envs/dagvine-env`.
- Builds and installs `cctools-dagvine` (`task-graph` branch).
- Installs Python deps.
- Packs worker env to `factories/dagvine-env.tar.gz`.

Important:
- If `~/miniconda3/envs/dagvine-env` already exists, the script exits immediately.

Verify:

```bash
conda activate dagvine-env
vine_worker --version
```

## 2) Start workers (factory)

Use `factories/run_factory.sh`:

```bash
cd factories
bash run_factory.sh \
  -M dagvine-manager \
  --workers 40 \
  --cores 12 \
  --memory 10 \
  --disk 100 \
  --poncho-env dagvine-env.tar.gz
```

Key args:
- `-M, --manager-name`: must match `main.py`.
- `--workers`: fixed worker count.
- `--cores`, `--memory`, `--disk`: per-worker resources (`memory`/`disk` in GB).
- `--poncho-env`: packed env tarball.

## 3) Run `main.py`

```bash
conda activate dagvine-env
```

Built-in graph example:

```bash
python main.py \
  -X dagvine \
  -M dagvine-manager \
  -G dv5-small \
  --libcores 12 \
  --wait-for-workers 1 \
  --run-info-template test-dv5-small
```

Rules:
- Use `-G/--graph` as the workflow input.
- `--task-complexity` is only for `--graph`, range `[1, 5]`.
- `--manager-name` must match factory `-M`.
- **Required:** `main.py --libcores` must be set and must match factory `--cores`.
- For concurrent runs, use different `--run-info-template`.
- Use `--port` to avoid port collisions.

## 4) Main options (detailed)

Input and graph selection:
- `-G, --graph GRAPH [TOTAL_TASKS]`: graph name plus optional task count.
- `--task-complexity N`: per-task Python complexity for generated graphs (`1..5`).
- `--task-file FILE`: load a pickled graph file (available, but this guide uses `--graph`).
- `--graph-from-log PATH`: reserved for log-based graph input (currently not used in `main.py` flow).

Executor and manager:
- `-X, --executor`: execution backend (`dagvine`, `daskvine-functioncall`, `daskvine-pythontask`).
- `-M, --manager-name`: manager name used for worker/factory matching.
- `--port N`: force a fixed manager port (helps avoid port collisions).
- `--wait-for-workers N`: wait before scheduling (`1` waits for workers, `0` does not).
- `--max-workers N`: cap workers seen by manager (`-1` means default/no explicit cap).

Run directories and run metadata:
- `--run-info-path DIR`: base directory for TaskVine run info.
- `--run-info-template NAME`: run-info subdirectory name; use unique values for concurrent runs.
- `--staging-path DIR`: staging directory for wrapper/temp files; if unset, auto-generated per run.
- `--output-dir DIR`: DAGVine output directory.
- `--checkpoint-dir DIR`: checkpoint output directory.
- `--time-metrics-filename FILE`: write internal timing metrics to this file.

Resource and scheduling controls:
- `--libcores N`: cores for DAG library tasks / library resources.
- `--max-cores N`: upper bound on total cores used by DAGVine.
- `--scheduling-mode MODE`: scheduling mode for DaskVine paths (default `files`).
- `--task-priority-mode MODE`: task priority strategy (default `random`).
- `--prune-depth N`: graph/task pruning depth for workflow reduction experiments.
- `--temp-replica-count N`: temporary replica count for files/tasks.
- `--clean-redundant-replicas N`: enable cleanup of redundant replicas.
- `--enforce-worker-eviction-interval N`: worker eviction interval override.
- `--watch-library-logfiles N`: enable worker/library log-file watching.
- `--shift-disk-load N`: shift disk I/O load policy toggle.
- `--adapt-dask N`: enable adaptive Dask behavior in DAGVine run call.

Recovery and retry behavior:
- `--immediate-recovery N`: trigger immediate recovery behavior on failures.
- `--auto-recovery N`: enable automatic recovery.
- `--prioritize-recovery-tasks N`: prioritize recovery tasks in scheduling.
- `--max-retry-attempts N`: maximum retry attempts per failed task.
- `--retry-interval-sec N`: delay between retries (seconds).
- `--checkpoint-fraction F`: checkpoint fraction (0..1 style usage).

Fault injection and synthetic overhead:
- `--failure-injection-step-percent F`: inject failures at a workflow progress percentage.
- `--extra-task-output-size-mb A B`: add extra task output size (MB range/params).
- `--extra-task-sleep-time A B`: add extra sleep time to tasks.
- `--extra-serialize-time-sec F`: inject extra serialization time.
- `--short-timeout N`: use shorter timeout behavior for testing.

Debug and output verbosity:
- `--enable-debug-log N`: enable/disable debug logging.
- `--print-graph-details N`: print graph details before or during execution.
- `--print-results N`: print full result payload at the end.

Repeat runs:
- `--repeats N`: run the same submission multiple times in one call.

Full help:

```bash
python main.py -h
```

## 5) Supported `--graph` names

`chain`, `binary-forest`, `individuals`, `simple`, `dv5-small`, `dv5-large`, `rstriphoton`, `topeft`, `trivial`, `stencil`, `fft`, `sweep`, `binary-tree`, `random`, `blast`, `sipht`, `synthetic_scec`, `epigenomics`, `ligo`, `montage`.