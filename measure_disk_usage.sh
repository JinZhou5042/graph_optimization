#!/usr/bin/env bash
set -u -o pipefail

# Simple, single-threaded disk usage sweep for DagVine runs.
# For each (workflow, tasks) run:
# - run main.py with a unique run-info template under run-info path
# - after completion, measure total size of that run-info directory
# - record the size (GiB) into dagvine_exp/csv/process_disk_consumption.csv
# - delete the run-info directory after measurement

START_TIME_SEC=$(date +%s)

# Ignore SIGPIPE so broken pipes don't terminate the script.
trap '' PIPE

# ===== Ctrl+C / SIGTERM handling =====
CURRENT_RUN_INFO_DIR=""
CURRENT_STAGING_DIR=""
STAGING_BASE="/groups/dthain/users/jzhou24/staging"
cleanup_run_artifacts() {
  # Best-effort: remove per-run directories.
  if [[ -n "${CURRENT_RUN_INFO_DIR}" ]]; then
    rm -rf "${CURRENT_RUN_INFO_DIR}" 2>/dev/null || true
  fi
  if [[ -n "${CURRENT_STAGING_DIR}" ]]; then
    rm -rf "${CURRENT_STAGING_DIR}" 2>/dev/null || true
  fi
  CURRENT_RUN_INFO_DIR=""
  CURRENT_STAGING_DIR=""
}

kill_mainpy_processes() {
  # On interrupt, ensure no stray main.py processes remain.
  # Limit to current user for safety.
  local user="${USER:-}"
  local pattern='[p]ython.*main\.py'

  if command -v pgrep >/dev/null 2>&1; then
    # Try a graceful termination first.
    local pids
    pids="$(pgrep -u "$user" -f "$pattern" 2>/dev/null || true)"
    if [[ -n "${pids//[[:space:]]/}" ]]; then
      kill -TERM $pids 2>/dev/null || true
      sleep 1
      # If still alive, force kill.
      pids="$(pgrep -u "$user" -f "$pattern" 2>/dev/null || true)"
      if [[ -n "${pids//[[:space:]]/}" ]]; then
        kill -KILL $pids 2>/dev/null || true
      fi
    fi
    return 0
  fi

  # Fallback: best-effort via ps parsing.
  # shellcheck disable=SC2009
  local pids_ps
  pids_ps="$(ps -u "$user" -o pid=,args= 2>/dev/null | awk '/python/ && /main\\.py/ {print $1}')"
  if [[ -n "${pids_ps//[[:space:]]/}" ]]; then
    kill -TERM $pids_ps 2>/dev/null || true
    sleep 1
    kill -KILL $pids_ps 2>/dev/null || true
  fi
}
cleanup_on_signal() {
  trap - INT TERM
  echo "" >&2
  echo "[interrupt] caught signal, cleaning up and exiting..." >&2
  cleanup_run_artifacts
  kill_mainpy_processes
  exit 130
}
trap cleanup_on_signal INT TERM

# ===== Fixed executor / manager =====
DAGVINE_EXECUTOR_ARG="dagvine"
MANAGER_NAME="jzhou24-default"

# ===== Output =====
CSV_DIR="dagvine_exp/csv"
DISK_CSV_PATH="${CSV_DIR}/process_disk_consumption.csv"

# ===== Required run-info base (avoid collisions across runs) =====
# RUN_INFO_PATH="/users/jzhou24/afs/taskvine-report-tool/logs/project-dagvine"
RUN_INFO_PATH="/groups/dthain/users/jzhou24"
# ===== Workloads / tasks =====
WORKFLOWS=(trivial stencil sweep fft binary-tree random blast sipht synthetic_scec epigenomics montage ligo)
# Sweep tasks as powers of 2: 2^0 .. 2^24 (inclusive)
TASK_COUNTS=()
for (( _p=0; _p<=24; _p++ )); do
  TASK_COUNTS+=( "$((1 << _p))" )
done

cap_first() {
  local s="${1:-}"
  if [[ -z "$s" ]]; then
    echo ""
    return 0
  fi
  printf '%s' "${s^}"
}

fmt6() {
  local x="${1:-}"
  if [[ -z "${x//[[:space:]]/}" ]]; then
    echo ""
    return 0
  fi
  awk -v x="$x" 'BEGIN { printf "%.6f\n", (x+0.0) }'
}

bytes_to_gb() {
  # bytes -> GiB
  local bytes="${1:-0}"
  awk -v b="$bytes" 'BEGIN { printf "%.6f\n", (b+0.0)/1024.0/1024.0/1024.0 }'
}

random_run_info_template() {
  # Unique-ish token without external deps.
  printf 'disk-%s-%s-%s-%s' "$(date +%s%N)" "$$" "$BASHPID" "$RANDOM"
}

init_wide_csv_if_missing() {
  local path="$1"
  if [[ -e "$path" && -s "$path" ]]; then
    local header
    header="$(head -n 1 "$path" 2>/dev/null || true)"
    header="${header//$'\r'/}"
    if [[ -n "${header//[[:space:]]/}" ]]; then
      return 0
    fi
  fi

  local tmp="${path}.tmp"
  {
    printf 'tasks'
    local wf
    for wf in "${WORKFLOWS[@]}"; do
      printf ',%s' "$(cap_first "$wf")"
    done
    printf '\n'

    local t
    for t in "${TASK_COUNTS[@]}"; do
      printf '%s' "$t"
      for wf in "${WORKFLOWS[@]}"; do
        printf ','
      done
      printf '\n'
    done
  } > "$tmp"
  mv -f "$tmp" "$path"
}

ensure_wide_csv_format() {
  local path="$1"
  init_wide_csv_if_missing "$path"

  local expected_header="tasks"
  local wf
  for wf in "${WORKFLOWS[@]}"; do
    expected_header+=",$(cap_first "$wf")"
  done

  local header
  header="$(head -n 1 "$path" 2>/dev/null || true)"
  header="${header//$'\r'/}"
  if [[ "$header" != "$expected_header" ]]; then
    echo "[error] ${path} header mismatch. Expected:" >&2
    echo "        ${expected_header}" >&2
    echo "        Got: ${header}" >&2
    echo "[hint] delete ${path} to regenerate in latest format" >&2
    return 2
  fi

  # Ensure all tasks rows exist and are in canonical order.
  local tasks_csv
  tasks_csv="$(IFS=,; echo "${TASK_COUNTS[*]}")"

  local tmp="${path}.tmp"
  awk -F',' -v OFS=',' -v tasks_list="$tasks_csv" '
    BEGIN{ Tn = split(tasks_list, T, ","); N=0 }
    NR==1{ N=NF; print; next }
    {
      t=$1
      if (t=="") next
      if (!(t in ROW)) ROW[t]=$0
    }
    END{
      for (ti=1; ti<=Tn; ti++){
        t=T[ti]
        if (t in ROW) {
          print ROW[t]
        } else {
          # If missing, emit a tasks row with blanks for each workflow column.
          printf "%s", t
          for (i=2; i<=N; i++) printf ","
          printf "\n"
        }
      }
    }
  ' "$path" > "$tmp"
  mv -f "$tmp" "$path"
}

cell_is_empty() {
  local path="$1"
  local col="$2"   # header name (e.g. Binary-tree)
  local tasks="$3"
  awk -F',' -v c="$col" -v t="$tasks" '
    NR==1{
      for (i=1;i<=NF;i++) if ($i==c) COL=i
      next
    }
    $1==t{
      if (COL==0) { exit 3 }
      v=$COL
      gsub(/\r/, "", v)
      if (v=="") exit 0
      exit 1
    }
  ' "$path"
}

update_wide_csv_cell_if_empty() {
  local path="$1"
  local col="$2"
  local tasks="$3"
  local value="$4"

  local tmp="${path}.tmp"
  awk -v c="$col" -v t="$tasks" -v val="$value" -F',' -v OFS=',' '
    NR==1{
      for (i=1;i<=NF;i++) if ($i==c) COL=i
      print
      next
    }
    {
      if ($1 == t && COL > 0 && $COL == "") $COL = val
      print
    }
    END{
      if (COL==0) exit 3
    }
  ' "$path" > "$tmp"
  mv -f "$tmp" "$path"
}

measure_run_info_dir_bytes() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    echo "[error] run-info directory missing: $dir" >&2
    return 2
  fi
  # Apparent size in bytes for the directory tree.
  local bytes
  bytes="$(du -sb "$dir" 2>/dev/null | awk '{print $1}')"
  if [[ -z "${bytes:-}" || ! "${bytes}" =~ ^[0-9]+$ ]]; then
    echo "[error] failed to measure disk usage for: $dir" >&2
    return 3
  fi
  printf '%s' "$bytes"
}

run_one() {
  local workflow="$1"
  local tasks="$2"

  local run_info_template run_info_template_full
  run_info_template="$(random_run_info_template)"
  run_info_template_full="${RUN_INFO_PATH}/${run_info_template}"
  CURRENT_RUN_INFO_DIR="$run_info_template_full"
  # main.py default staging-path: /groups/dthain/users/jzhou24/staging/<run-info-template>
  CURRENT_STAGING_DIR="${STAGING_BASE}/${run_info_template}"

  echo "[start] workflow=${workflow} executor=dagvine tasks=${tasks} run_info=${run_info_template_full}" >&2

  # Run the workflow. Note: -G is graph name + optional tasks.
  if ! env NO_COLOR=1 TERM=dumb python3 main.py \
    -X "$DAGVINE_EXECUTOR_ARG" \
    -G "$workflow" "$tasks" \
    -M "$MANAGER_NAME" \
    --libcores 12 \
    --run-info-path "$RUN_INFO_PATH" \
    --run-info-template "$run_info_template" \
    --enable-debug-log 1 \
    --watch-library-logfiles 0 \
    >/dev/null 2>&1
  then
    echo "[fail] workflow=${workflow} tasks=${tasks}" >&2
    # Best-effort cleanup.
    cleanup_run_artifacts
    return 1
  fi

  local bytes gb
  bytes="$(measure_run_info_dir_bytes "$run_info_template_full")" || {
    cleanup_run_artifacts
    return 1
  }
  gb="$(bytes_to_gb "$bytes")"

  # Cleanup to avoid accumulating per-run logs/staging.
  cleanup_run_artifacts

  echo "[ok] workflow=${workflow} tasks=${tasks} disk_gb=${gb}" >&2
  printf '%s' "$gb"
}

mkdir -p "$CSV_DIR"
ensure_wide_csv_format "$DISK_CSV_PATH" || exit $?

echo "========== Disk usage sweep: workflows=${#WORKFLOWS[@]} tasks_points=${#TASK_COUNTS[@]} ==========" >&2

failures=0
for tasks in "${TASK_COUNTS[@]}"; do
  for wf in "${WORKFLOWS[@]}"; do
    col="$(cap_first "$wf")"
    if cell_is_empty "$DISK_CSV_PATH" "$col" "$tasks"; then
      disk_gb="$(run_one "$wf" "$tasks")" || { failures=1; continue; }
      update_wide_csv_cell_if_empty "$DISK_CSV_PATH" "$col" "$tasks" "$(fmt6 "$disk_gb")" || failures=1
    fi
  done
done

END_TIME_SEC=$(date +%s)
TOTAL_TIME_USED_SEC=$((END_TIME_SEC - START_TIME_SEC))
echo "Total time used: ${TOTAL_TIME_USED_SEC} seconds" >&2

if [[ "$failures" -ne 0 ]]; then
  echo "[warn] some runs failed" >&2
  exit 1
fi

exit 0

