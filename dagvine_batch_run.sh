#!/usr/bin/env bash

set -u -o pipefail

START_TIME_SEC=$(date +%s)

# Ignore SIGPIPE so that broken pipes (e.g. piped output consumers exiting early)
# don't terminate the script.
trap '' PIPE

# ===== CLI args =====
THREADS=12
REPEATS=3
# Single executor passed through to main.py -X
EXECUTOR="dagvine"
ENABLE_GDB=0
# If set to non-"none", forces all graph-threads to use the same manager name.
MANAGER_NAME_OVERRIDE="none"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --threads)
      shift
      THREADS="${1:-}"
      if [[ -z "${THREADS}" || ! "${THREADS}" =~ ^[0-9]+$ || "${THREADS}" -lt 1 ]]; then
        echo "[error] --threads expects a positive integer" >&2
        exit 2
      fi
      shift
      ;;
    --repeats)
      shift
      REPEATS="${1:-}"
      if [[ -z "${REPEATS}" || ! "${REPEATS}" =~ ^[0-9]+$ || "${REPEATS}" -lt 1 ]]; then
        echo "[error] --repeats expects a positive integer" >&2
        exit 2
      fi
      shift
      ;;
    --executor|-X)
      shift
      EXECUTOR="${1:-}"
      if [[ -z "${EXECUTOR}" || "${EXECUTOR}" == -* ]]; then
        echo "[error] --executor/-X expects a value (e.g. -X dagvine)" >&2
        exit 2
      fi
      shift
      ;;
    --gdb)
      ENABLE_GDB=1
      shift
      ;;
    --manager-name|-M)
      shift
      MANAGER_NAME_OVERRIDE="${1:-}"
      if [[ -z "${MANAGER_NAME_OVERRIDE}" ]]; then
        echo "[error] --manager-name/-M expects a value (or use 'none' to disable override)" >&2
        exit 2
      fi
      shift
      ;;
    --help|-h)
      cat <<'EOF'
Usage: bash dagvine_batch_run.sh [--threads N] [--repeats N] [--executor EXEC|-X EXEC] [--manager-name NAME|-M NAME] [--gdb]

Options:
  --threads N     Max concurrent runs per iteration (default: 12)
  --repeats N     Repeats per (executor, graph, cores) (default: 3); tp.out will have N lines
  --executor EXEC, -X EXEC
                 Executor passed to main.py -X (default: dagvine)
  --manager-name NAME, -M NAME
                 Force all threads to use the same manager name for main.py -M (default: none).
                 When set to 'none', each thread uses its own per-graph manager name.
  --gdb           On SIGSEGV(139), rerun under gdb and print full backtrace (default: off)
EOF
      exit 0
      ;;
    *)
      echo "[error] Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

# ===== Ctrl+C / SIGTERM handling: terminate all spawned procs =====
RUN_PIDS=()
cleanup_on_signal() {
  trap - INT TERM
  echo "" >&2
  echo "[interrupt] caught signal, terminating ${#RUN_PIDS[@]} running process(es)..." >&2

  # Prefer killing the whole process group (we start jobs with setsid so pid == pgid).
  for pid in "${RUN_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    fi
  done

  # Give processes a moment to exit gracefully, then SIGKILL leftovers.
  sleep 2
  for pid in "${RUN_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
  done

  # Reap children to avoid zombies.
  for pid in "${RUN_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done

  exit 130
}
trap cleanup_on_signal INT TERM

# ===== Helpers =====
count_tp_lines() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo 0
    return 0
  fi
  awk 'NF { n++ } END { print n+0 }' "$f"
}

extract_throughput_from_log() {
  local log_file="$1"
  sed -E 's/\x1B\[[0-9;]*[[:alpha:]]//g' "$log_file" | awk '
    # Preferred stable format:
    #   [graph] Throughput: <num> tasks/s
    /^\[[^]]+\][[:space:]]+Throughput:/ { tp=$3 }
    /^[[:space:]]*throughput[[:space:]]*=/ { tp=$3 }
    /Throughput:/ {
      for (i = 1; i <= NF; i++) {
        if ($i == "Throughput:") { tp = $(i + 1) }
      }
    }
    END { if (tp != "") print tp }
  '
}

signal_name_from_rc() {
  local rc="$1"
  if [[ -z "${rc}" || ! "${rc}" =~ ^[0-9]+$ ]]; then
    echo "?"
    return 0
  fi
  if (( rc >= 128 )); then
    local sig=$(( rc - 128 ))
    # `kill -l <num>` prints signal name on Linux.
    kill -l "$sig" 2>/dev/null || echo "$sig"
    return 0
  fi
  echo "-"
}

print_failure_details_if_needed() {
  local executor="$1"
  local graph="$2"
  local iter="$3"
  local max_cores="$4"
  local r="$5"
  local rc="$6"
  local log_file="$7"

  local sig_name
  sig_name="$(signal_name_from_rc "$rc")"

  echo "[fail] executor=${executor} graph=${graph} iter=${iter} max_cores=${max_cores} r=${r} exit=${rc} signal=${sig_name} (see ${log_file})" >&2

  # If segfault (SIGSEGV -> 11 -> rc 139), dump the full log to terminal.
  if [[ "${rc}" == "139" ]]; then
    echo "[fail] ===== BEGIN FULL LOG (SIGSEGV) ${log_file} =====" >&2
    if [[ -f "${log_file}" ]]; then
      cat "${log_file}" >&2 || true
    else
      echo "[fail] (log file missing)" >&2
    fi
    echo "[fail] ===== END FULL LOG (SIGSEGV) ${log_file} =====" >&2
  fi
}

run_graph_job() {
  local executor="$1"
  local graph="$2"
  local iter="$3"
  local max_cores="$4"
  local wait_for_workers="$5"
  local repeats="$6"
  local data_root="$7"
  local extra_sleep_tag="$8"
  local manager_name_override="$9"
  local manager_prefix="${10}"
  local libcores="${11}"
  local run_info_path="${12}"
  local total_tasks="${13}"
  local extra_task_sleep_time_str="${14}"
  local port="${15}"
  local enable_gdb="${16}"

  local out_dir tp_file completed r log_file manager_name run_info_template throughput rc
  out_dir="${data_root}/${executor}/${graph}/${extra_sleep_tag}/${max_cores}"
  tp_file="${out_dir}/tp.out"

  mkdir -p "$out_dir"

  completed="$(count_tp_lines "$tp_file")"
  if (( completed >= repeats )); then
    echo "[skip] executor=${executor} graph=${graph} iter=${iter} max_cores=${max_cores} -> ${tp_file} has ${completed} lines" >&2
    return 0
  fi

  for r in $(seq "$completed" $((repeats - 1))); do
    log_file="${out_dir}/run_r${r}.log"
    : > "$log_file"

    if [[ "${manager_name_override}" != "none" ]]; then
      manager_name="${manager_name_override}"
    else
      manager_name="${manager_prefix}-${graph}"
    fi
    run_info_template="${executor}-${graph}-${extra_sleep_tag}-mc${max_cores}-iter${iter}-r${r}"

    # main.py expects 2 values for --extra-task-sleep-time
    read -r -a _extra_task_sleep_time_arr <<< "$extra_task_sleep_time_str"
    if (( ${#_extra_task_sleep_time_arr[@]} != 2 )); then
      echo "[error] internal: --extra-task-sleep-time expects 2 values, got: ${extra_task_sleep_time_str}" >&2
      return 2
    fi

    run_cmd=(
      env NO_COLOR=1 TERM=dumb python3 main.py
      -X "$executor"
      -M "$manager_name"
      --port "$port"
      --libcores "$libcores"
      --run-info-path "$run_info_path"
      --run-info-template "$run_info_template"
      -G "$graph" "$total_tasks"
      --wait-for-workers "$wait_for_workers"
      --max-cores "$max_cores"
      --extra-task-sleep-time "${_extra_task_sleep_time_arr[@]}"
      --enable-debug-log 0
    )

    printf 'CMD:' >> "$log_file"
    printf ' %q' "${run_cmd[@]}" >> "$log_file"
    printf '\n' >> "$log_file"

    echo "[start] executor=${executor} graph=${graph} iter=${iter} max_cores=${max_cores} r=${r} manager=${manager_name} port=${port} log=${log_file}" >&2

    "${run_cmd[@]}" >>"$log_file" 2>&1
    rc=$?
    if (( rc != 0 )); then
      print_failure_details_if_needed "$executor" "$graph" "$iter" "$max_cores" "$r" "$rc" "$log_file"

      # If segfault, rerun once under gdb to capture a full native backtrace.
      # This rerun uses an isolated run-info-template and port to avoid collisions.
      if [[ "${rc}" == "139" && "${enable_gdb}" == "1" ]]; then
        if ! command -v gdb >/dev/null 2>&1; then
          echo "[gdb] gdb not found in PATH; cannot collect backtrace" >&2
          return "$rc"
        fi

        gdb_log="${log_file%.log}.gdb.txt"
        gdb_port=$((port + 1000))

        # Clone command and tweak --port and --run-info-template for the gdb rerun.
        gdb_run_cmd=("${run_cmd[@]}")
        for ((idx=0; idx<${#gdb_run_cmd[@]}; idx++)); do
          if [[ "${gdb_run_cmd[$idx]}" == "--port" ]]; then
            gdb_run_cmd[$((idx+1))]="$gdb_port"
          fi
          if [[ "${gdb_run_cmd[$idx]}" == "--run-info-template" ]]; then
            gdb_run_cmd[$((idx+1))]="${gdb_run_cmd[$((idx+1))]}-gdb"
          fi
        done

        echo "[gdb] segfault detected; rerunning under gdb for backtrace -> ${gdb_log}" >&2
        {
          echo "===== GDB CMD ====="
          printf '%q ' timeout 180s gdb -q -batch \
            -ex "set pagination off" \
            -ex "set confirm off" \
            -ex "set follow-fork-mode child" \
            -ex "set detach-on-fork off" \
            -ex "run" \
            -ex "thread apply all bt full" \
            -ex "quit" \
            --args "${gdb_run_cmd[@]}"
          echo ""
          echo "===== GDB OUTPUT ====="
          timeout 180s gdb -q -batch \
            -ex "set pagination off" \
            -ex "set confirm off" \
            -ex "set follow-fork-mode child" \
            -ex "set detach-on-fork off" \
            -ex "run" \
            -ex "thread apply all bt full" \
            -ex "quit" \
            --args "${gdb_run_cmd[@]}"
          gdb_rc=$?
          if [[ "$gdb_rc" == "124" ]]; then
            echo "[gdb] timed out (180s): segfault not reproduced under gdb; see ${gdb_log}" >&2
          fi
          echo "===== END GDB OUTPUT ====="
        } 2>&1 | tee "$gdb_log" >&2
      fi

      return "$rc"
    fi

    throughput="$(extract_throughput_from_log "$log_file")"
    if [[ -z "${throughput:-}" ]]; then
      echo "[fail] executor=${executor} graph=${graph} iter=${iter} max_cores=${max_cores} r=${r} throughput=NA (see ${log_file})" >&2
      return 3
    fi

    printf '%s\n' "$throughput" >> "$tp_file"
    echo "[ok] executor=${executor} graph=${graph} iter=${iter} max_cores=${max_cores} r=${r} throughput=${throughput} -> ${tp_file}" >&2
  done

  return 0
}

graph_thread() {
  local executor="$1"
  local graph="$2"
  local repeats="$3"
  local data_root="$4"
  local extra_sleep_tag="$5"
  local manager_name_override="$6"
  local manager_prefix="$7"
  local libcores="$8"
  local run_info_path="$9"
  local total_tasks="${10}"
  local extra_task_sleep_time_str="${11}"
  local port="${12}"
  local enable_gdb="${13}"

  local i max_cores wait_for_workers rc
  for i in $(seq 0 10); do
    max_cores=$(( 1 << i ))
    # wait_for_workers=$(( max_cores / libcores + 1 ))
    wait_for_workers=8

    echo "========== executor=${executor} graph=${graph} iter=${i} max_cores=${max_cores} extra_sleep=${extra_sleep_tag} repeats=${repeats} ==========" >&2

    run_graph_job \
      "$executor" "$graph" "$i" "$max_cores" "$wait_for_workers" \
      "$repeats" "$data_root" "$extra_sleep_tag" "$manager_name_override" "$manager_prefix" \
      "$libcores" "$run_info_path" "$total_tasks" "$extra_task_sleep_time_str" "$port" "$enable_gdb"
    rc=$?
    if (( rc != 0 )); then
      echo "[fail] executor=${executor} graph=${graph} iter=${i} max_cores=${max_cores} exit=${rc}" >&2
      return "$rc"
    fi
  done

  return 0
}

# Needed so `setsid bash -c '...'` can call these.
export -f count_tp_lines
export -f extract_throughput_from_log
export -f signal_name_from_rc
export -f print_failure_details_if_needed
export -f run_graph_job
export -f graph_thread

# ===== Global configuration =====
# 1) Define 12 workflows
GRAPHS=(trivial stencil sweep fft binary-tree random blast sipht synthetic_scec epigenomics montage ligo)

# 2) Total tasks (passed to main.py via -G GRAPH TOTAL_TASKS)
TOTAL_TASKS=5000
SLEEP_TIME=0

# 3) Wait for workers (computed dynamically per iteration)

# 4) --extra-task-sleep-time (main.py expects 2 values)
EXTRA_TASK_SLEEP_TIME=($SLEEP_TIME $SLEEP_TIME)

# Output layout (global)
DATA_ROOT="dagvine_exp/data"

# Each run uses the same run-info-path, but run-info-template must be unique
RUN_INFO_PATH="/users/jzhou24/afs/taskvine-report-tool/logs/project-dagvine"

# Fixed parameters
LIBCORES=12
# Manager name prefix (actual -M binds to graph name)
MANAGER_PREFIX="jzhou24"

extra_sleep_tag="$(printf '%s' "${EXTRA_TASK_SLEEP_TIME[*]}" | tr ' ' '_')"

# 5) Graph-threaded execution:
# Each background "thread" owns a single graph and independently advances iter=0..10.
# Repeats for the same (executor, graph, iter) are always serial (handled inside run_graph_job).
failures=0
executor="${EXECUTOR}"

# Track PIDs so Ctrl+C can terminate all running jobs.
RUN_PIDS=()
declare -A META_BY_PID=()

echo "========== executor=${executor} extra_sleep=${extra_sleep_tag} threads=${THREADS} repeats=${REPEATS} ==========" >&2

# Assign a unique port per graph-thread to avoid manager port collisions.
case "$executor" in
  dagvine) _port_base=20000 ;;
  daskvine-pythontask) _port_base=21000 ;;
  daskvine-functioncall) _port_base=22000 ;;
  *) _port_base=23000 ;;
esac

_graph_idx=0
for graph in "${GRAPHS[@]}"; do
  # Limit concurrency to THREADS graph-threads.
  while (( $(jobs -pr | wc -l) >= THREADS )); do
    sleep 0.2
  done

  port=$((_port_base + _graph_idx))
  setsid bash -c 'graph_thread "$@"' _ \
    "$executor" "$graph" \
    "$REPEATS" "$DATA_ROOT" "$extra_sleep_tag" "$MANAGER_NAME_OVERRIDE" "$MANAGER_PREFIX" \
    "$LIBCORES" "$RUN_INFO_PATH" "$TOTAL_TASKS" "${EXTRA_TASK_SLEEP_TIME[*]}" "$port" "$ENABLE_GDB" \
    &
  _graph_idx=$((_graph_idx + 1))

  pid="$!"
  RUN_PIDS+=("$pid")
  META_BY_PID["$pid"]="${executor} graph=${graph}"
done

# Wait for all graph threads in this executor.
for pid in "${RUN_PIDS[@]}"; do
  wait "$pid"
  rc=$?
  if (( rc != 0 )); then
    failures=$((failures + 1))
    echo "[fail] ${META_BY_PID[$pid]:-job} exit=${rc}" >&2
  fi
done

# End: clear RUN_PIDS so a later Ctrl+C won't target finished jobs.
RUN_PIDS=()

if [[ "$failures" -ne 0 ]]; then
  echo "[warn] total failures=${failures}" >&2
fi

END_TIME_SEC=$(date +%s)
TOTAL_TIME_USED_SEC=$((END_TIME_SEC - START_TIME_SEC))
echo "Total time used: ${TOTAL_TIME_USED_SEC} seconds" >&2
