#!/usr/bin/env bash
set -u -o pipefail

START_TIME_SEC=$(date +%s)

# Ignore SIGPIPE so that broken pipes (e.g. piped output consumers exiting early)
# don't terminate the script.
trap '' PIPE

# ===== Fixed executors =====
# As requested: only consider DaskVine function-calls mode.
DASKVINE_EXECUTOR_ARG="daskvine-functioncall"

# ===== Concurrency =====
THREADS=24
REPEATS=1

PARTS_DIR="dagvine_exp/csv/frontend-overhead-parts"

# ===== CLI args (only concurrency) =====
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
    --help|-h)
      cat <<'EOF'
Usage: bash frontend_overhead_batch_run.sh [--threads N] [--repeats N]

Options:
  --threads N   Max concurrent workflow threads (default: 24)
  --repeats N   Repeats per (workflow, tasks, executor) (default: 1)
EOF
      exit 0
      ;;
    *)
      echo "[error] Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

# ===== Required run-info settings (to avoid collisions) =====
RUN_INFO_PATH="/users/jzhou24/afs/taskvine-report-tool/logs/project-dagvine"

# ===== Ctrl+C / SIGTERM handling =====
cleanup_on_signal() {
  trap - INT TERM
  echo "" >&2
  echo "[interrupt] caught signal, exiting..." >&2
  # Best-effort: terminate background workflow threads.
  kill 0 2>/dev/null || true
  exit 130
}
trap cleanup_on_signal INT TERM

# ===== Workloads / tasks =====
WORKFLOWS=(trivial stencil sweep fft binary-tree random blast sipht synthetic_scec epigenomics montage ligo)
# Sweep tasks as powers of 2: 2^0 .. 2^24 (inclusive)
TASK_COUNTS=()
for (( _p=0; _p<=24; _p++ )); do
  TASK_COUNTS+=( "$((1 << _p))" )
done

key_for() {
  local workflow="$1"
  local tasks="$2"
  printf '%s|%s' "$workflow" "$tasks"
}

random_run_info_template() {
  # Generate a unique-ish token without external deps.
  # Example: feoh-1700000000-12345-6789-4242
  printf 'feoh-%s-%s-%s-%s' "$(date +%s%N)" "$$" "$BASHPID" "$RANDOM"
}

extract_frontend_overhead_seconds() {
  local text="$1"
  # Keep the last occurrence if multiple lines appear.
  awk '
    match($0, /=== Frontend Overhead:[[:space:]]*([0-9.]+)[[:space:]]*seconds/, m) { val=m[1] }
    END { if (val != "") print val }
  ' <<< "$text"
}

mean_stderr_from_values() {
  # Reads newline-separated numeric values on stdin and prints:
  #   <mean> <stderr>
  # stderr uses sample stddev (n-1) when n>1; otherwise stderr=0.
  awk '
    {
      x = $1 + 0.0
      n += 1
      sum += x
      sumsq += x * x
    }
    END {
      if (n <= 0) { exit 2 }
      mean = sum / n
      if (n == 1) {
        stderr = 0.0
      } else {
        var = (sumsq - (sum * sum) / n) / (n - 1)
        if (var < 0) var = 0
        stderr = sqrt(var) / sqrt(n)
      }
      printf "%.6f %.6f\n", mean, stderr
    }
  '
}

# ===== Repeat-values helpers =====
repeats_count() {
  local s="${1:-}"
  if [[ -z "$s" ]]; then
    echo 0
    return 0
  fi
  local -a arr
  IFS='|' read -r -a arr <<< "$s"
  echo "${#arr[@]}"
}

repeats_append() {
  local s="${1:-}"
  local v="${2:-}"
  if [[ -z "$s" ]]; then
    printf '%s' "$v"
  else
    printf '%s|%s' "$s" "$v"
  fi
}

repeats_nth() {
  # 1-based index. Prints empty if missing.
  local s="${1:-}"
  local n="${2:-}"
  if [[ -z "$s" ]]; then
    return 0
  fi
  awk -v n="$n" -F'|' '{
    if (n >= 1 && n <= NF) print $n
  }' <<< "$s"
}

mean_stderr_from_repeats_string() {
  # Prints "<mean> <stderr>" if >=1 value exists; otherwise prints empty line.
  local s="${1:-}"
  if [[ -z "$s" ]]; then
    echo ""
    return 1
  fi
  local values
  values="${s//|/$'\n'}"
  mean_stderr_from_values <<< "$values"
}

# ===== Resume support: load existing part files and skip completed configs =====
declare -A EXIST_DASK_REPEATS=()      # key=workflow|tasks -> "v1|v2|..."
declare -A EXIST_DAG_REPEATS=()       # key=workflow|tasks -> "v1|v2|..."
declare -A EXIST_DASK_COLS_BY_WF=()   # key=workflow -> max repeat columns seen in file header
declare -A EXIST_DAG_COLS_BY_WF=()    # key=workflow -> max repeat columns seen in file header

load_existing_parts() {
  if [[ ! -d "$PARTS_DIR" ]]; then
    return 0
  fi

  local line workflow part_file
  local loaded=0
  for part_file in "$PARTS_DIR"/*.csv; do
    [[ -e "$part_file" ]] || continue
    # Extract workflow name from filename (e.g., "blast.csv" -> "blast")
    workflow="$(basename "$part_file" .csv)"

    # Read header first to locate repeat columns.
    local header
    header="$(head -n 1 "$part_file" 2>/dev/null || true)"
    [[ -z "${header:-}" ]] && continue

    # Split header robustly (preserve trailing empties via sentinel).
    local header_s="${header},__END__"
    local -a H=()
    IFS=',' read -r -a H <<< "$header_s"
    unset 'H[${#H[@]}-1]'

    local idx_tasks=0
    local -A IDX_DASK=()
    local -A IDX_DAG=()
    local max_dask_cols=0
    local max_dag_cols=0

    local i colname n
    for ((i=0; i<${#H[@]}; i++)); do
      colname="${H[$i]}"
      if [[ "$colname" =~ ^daskvine-r([0-9]+)$ ]]; then
        n="${BASH_REMATCH[1]}"
        IDX_DASK["$n"]="$i"
        (( n > max_dask_cols )) && max_dask_cols="$n"
      elif [[ "$colname" =~ ^dagvine-r([0-9]+)$ ]]; then
        n="${BASH_REMATCH[1]}"
        IDX_DAG["$n"]="$i"
        (( n > max_dag_cols )) && max_dag_cols="$n"
      fi
    done

    # Track maximum repeat columns per workflow (so we don't delete extra repeats when --repeats shrinks).
    local prev
    prev="${EXIST_DASK_COLS_BY_WF[$workflow]:-0}"
    (( max_dask_cols > prev )) && EXIST_DASK_COLS_BY_WF["$workflow"]="$max_dask_cols"
    prev="${EXIST_DAG_COLS_BY_WF[$workflow]:-0}"
    (( max_dag_cols > prev )) && EXIST_DAG_COLS_BY_WF["$workflow"]="$max_dag_cols"

    # Read remaining lines (data rows). Use process-substitution to avoid subshell.
    local tasks k
    local -a F=()
    local dask_s dag_s v
    while IFS= read -r line || [[ -n "$line" ]]; do
      [[ -z "$line" ]] && continue
      # Split row robustly (preserve trailing empties via sentinel).
      local line_s="${line},__END__"
      IFS=',' read -r -a F <<< "$line_s"
      unset 'F[${#F[@]}-1]'

      tasks="${F[$idx_tasks]:-}"
      [[ -z "${tasks:-}" ]] && continue
      k="$(key_for "$workflow" "$tasks")"

      # Load consecutive non-empty repeats from r1..rN (stop at first empty).
      dask_s=""
      if (( max_dask_cols > 0 )); then
        local rr idx
        for ((rr=1; rr<=max_dask_cols; rr++)); do
          idx="${IDX_DASK[$rr]:-}"
          [[ -z "${idx:-}" ]] && break
          v="${F[$idx]:-}"
          [[ -z "${v:-}" ]] && break
          dask_s="$(repeats_append "$dask_s" "$v")"
        done
      fi

      dag_s=""
      if (( max_dag_cols > 0 )); then
        local rr idx
        for ((rr=1; rr<=max_dag_cols; rr++)); do
          idx="${IDX_DAG[$rr]:-}"
          [[ -z "${idx:-}" ]] && break
          v="${F[$idx]:-}"
          [[ -z "${v:-}" ]] && break
          dag_s="$(repeats_append "$dag_s" "$v")"
        done
      fi

      EXIST_DASK_REPEATS["$k"]="$dask_s"
      EXIST_DAG_REPEATS["$k"]="$dag_s"
      loaded=$((loaded + 1))
    done < <(tail -n +2 "$part_file" 2>/dev/null || true)
  done

  echo "[resume] loaded ${loaded} existing row(s) from ${PARTS_DIR}/" >&2
}

run_one() {
  local executor_label="$1"     # "dagvine" | "daskvine" (for stderr only)
  local executor_arg="$2"       # main.py -X value
  local workflow="$3"
  local tasks="$4"

  local run_info_template output overhead_s rc
  run_info_template="$(random_run_info_template)"
  local run_info_template_full="${RUN_INFO_PATH}/${run_info_template}"

  run_cmd=(
    env NO_COLOR=1 TERM=dumb python3 main.py
    -X "$executor_arg"
    -G "$workflow" "$tasks"
    --run-info-path "$RUN_INFO_PATH"
    --run-info-template "$run_info_template"
    --enable-debug-log 0
  )

  echo "[start] workflow=${workflow} executor=${executor_label} (arg=${executor_arg}) tasks=${tasks}" >&2
  output="$("${run_cmd[@]}" 2>&1)"
  rc=$?
  if (( rc != 0 )); then
    echo "[fail] workflow=${workflow} executor=${executor_label} tasks=${tasks} exit=${rc}" >&2
    echo "$output" >&2
    # Best-effort cleanup: remove this run-info template directory.
    if [[ -n "${run_info_template:-}" && "${run_info_template}" != */* ]]; then
      rm -rf "$run_info_template_full" 2>/dev/null || true
    fi
    return "$rc"
  fi

  overhead_s="$(extract_frontend_overhead_seconds "$output")"
  if [[ -z "${overhead_s:-}" ]]; then
    echo "[fail] workflow=${workflow} executor=${executor_label} tasks=${tasks} overhead=NA (missing '=== Frontend Overhead:' line)" >&2
    echo "$output" >&2
    # Best-effort cleanup: remove this run-info template directory.
    if [[ -n "${run_info_template:-}" && "${run_info_template}" != */* ]]; then
      rm -rf "$run_info_template_full" 2>/dev/null || true
    fi
    return 3
  fi

  echo "[ok] workflow=${workflow} executor=${executor_label} tasks=${tasks} overhead_s=${overhead_s}" >&2
  # Best-effort cleanup: remove this run-info template directory.
  if [[ -n "${run_info_template:-}" && "${run_info_template}" != */* ]]; then
    rm -rf "$run_info_template_full" 2>/dev/null || true
  fi
  # Return overhead on stdout so caller can store it.
  printf '%s\n' "$overhead_s"
  return 0
}

part_file_for_workflow() {
  local workflow="$1"
  printf '%s/%s.csv' "$PARTS_DIR" "$workflow"
}

part_lock_for_workflow() {
  local workflow="$1"
  printf '%s/%s.csv.lock' "$PARTS_DIR" "$workflow"
}

init_part_file_if_missing() {
  local workflow="$1"
  local part_file
  part_file="$(part_file_for_workflow "$workflow")"
  if [[ -e "$part_file" ]]; then
    return 0
  fi

  local tmp="${part_file}.tmp"
  {
    printf 'tasks'
    local rr
    for ((rr=1; rr<=REPEATS; rr++)); do
      printf ',daskvine-r%s' "$rr"
    done
    printf ',daskvine-mean,daskvine-stderr'
    for ((rr=1; rr<=REPEATS; rr++)); do
      printf ',dagvine-r%s' "$rr"
    done
    printf ',dagvine-mean,dagvine-stderr\n'

    local t
    for t in "${TASK_COUNTS[@]}"; do
      printf '%s' "$t"
      for ((rr=1; rr<=REPEATS; rr++)); do
        printf ','
      done
      printf ',,'
      for ((rr=1; rr<=REPEATS; rr++)); do
        printf ','
      done
      printf ',,\n'
    done
  } > "$tmp"
  mv -f "$tmp" "$part_file"
}

normalize_part_file() {
  # Ensure per-workflow part file is in repeat-columns format and has at least REPEATS cols.
  local workflow="$1"
  init_part_file_if_missing "$workflow"

  local part_file lock_file
  part_file="$(part_file_for_workflow "$workflow")"
  lock_file="$(part_lock_for_workflow "$workflow")"

  (
    flock -x 9

    local header
    header="$(head -n 1 "$part_file" 2>/dev/null || true)"
    [[ -z "${header:-}" ]] && return 0

    # If already in new format and columns >= REPEATS, nothing to do.
    if [[ "$header" == *"daskvine-r1"* || "$header" == *"dagvine-r1"* ]]; then
      local dask_cols dag_cols
      dask_cols="$(awk -F',' '{ for (i=1;i<=NF;i++) if ($i ~ /^daskvine-r[0-9]+$/) c++; print c+0 }' <<< "$header")"
      dag_cols="$(awk -F',' '{ for (i=1;i<=NF;i++) if ($i ~ /^dagvine-r[0-9]+$/) c++; print c+0 }' <<< "$header")"
      if (( dask_cols >= REPEATS && dag_cols >= REPEATS )); then
        return 0
      fi
    fi

    # Rewrite (migrate legacy or expand columns) in one pass.
    local tmp="${part_file}.tmp"
    awk -v want="$REPEATS" -F',' -v OFS=',' '
      function is_dask_repeat(x) { return (x ~ /^daskvine-r[0-9]+$/) }
      function is_dag_repeat(x)  { return (x ~ /^dagvine-r[0-9]+$/) }
      function max(a,b){ return (a>b)?a:b }
      NR==1{
        legacy = ($0 == "tasks,daskvine-mean,daskvine-stderr,dagvine-mean,dagvine-stderr")
        if (!legacy) {
          for (i=1;i<=NF;i++){
            if (is_dask_repeat($i)) dask_cols++
            if (is_dag_repeat($i))  dag_cols++
          }
        } else {
          dask_cols=0; dag_cols=0
        }
        new_dask=max(dask_cols, want)
        new_dag=max(dag_cols, want)
        next
      }
      {
        rows++
        for (i=1;i<=NF;i++) R[rows,i]=$i
        Rn[rows]=NF
      }
      END{
        # Header
        printf "tasks"
        for (r=1;r<=new_dask;r++) printf ",daskvine-r%d", r
        printf ",daskvine-mean,daskvine-stderr"
        for (r=1;r<=new_dag;r++) printf ",dagvine-r%d", r
        printf ",dagvine-mean,dagvine-stderr\n"

        for (row=1; row<=rows; row++){
          # Legacy row shape: tasks, dm, ds, gm, gs
          if (legacy) {
            tasks = R[row,1]
            dm = R[row,2]; ds = R[row,3]; gm = R[row,4]; gs = R[row,5]
            printf "%s", tasks
            for (r=1;r<=new_dask;r++) printf ","
            printf ",%s,%s", dm, ds
            for (r=1;r<=new_dag;r++) printf ","
            printf ",%s,%s\n", gm, gs
            continue
          }

          # New-ish row: copy available repeats; preserve existing mean/stderr columns if present.
          # We assume layout: tasks, dask-repeats..., dask-mean, dask-stderr, dag-repeats..., dag-mean, dag-stderr
          tasks = R[row,1]
          printf "%s", tasks

          # Copy dask repeats
          for (r=1; r<=new_dask; r++){
            v = (r<=dask_cols) ? R[row,1+r] : ""
            printf ",%s", v
          }
          dm = (dask_cols>0) ? R[row,2+dask_cols] : ""
          ds = (dask_cols>0) ? R[row,3+dask_cols] : ""
          printf ",%s,%s", dm, ds

          dag_start = 4 + dask_cols
          for (r=1; r<=new_dag; r++){
            v = (r<=dag_cols) ? R[row,dag_start + (r-1)] : ""
            printf ",%s", v
          }
          gm = (dag_cols>0) ? R[row,dag_start + dag_cols] : ""
          gs = (dag_cols>0) ? R[row,dag_start + dag_cols + 1] : ""
          printf ",%s,%s\n", gm, gs
        }
      }
    ' "$part_file" > "$tmp"
    mv -f "$tmp" "$part_file"
  ) 9>"$lock_file"
}

emit_missing_jobs_for_workflow() {
  # Prints one job per missing repeat on stdout:
  #   <workflow> <tasks> <executor_label> <executor_arg>
  local workflow="$1"
  local part_file
  part_file="$(part_file_for_workflow "$workflow")"

  normalize_part_file "$workflow"

  awk -v wf="$workflow" -v want="$REPEATS" -v dask_arg="$DASKVINE_EXECUTOR_ARG" -F',' '
    function max(a,b){ return (a>b)?a:b }
    NR==1{
      for (i=1;i<=NF;i++){
        if ($i ~ /^daskvine-r[0-9]+$/) dask_cols++
        if ($i ~ /^dagvine-r[0-9]+$/)  dag_cols++
      }
      dask_cols=max(dask_cols, want)
      dag_cols=max(dag_cols, want)
      next
    }
    {
      tasks=$1
      # Count consecutive filled repeats (r1.. until first empty).
      dfilled=0
      for (r=1;r<=dask_cols;r++){
        v=$(1+r)
        if (v == "") break
        dfilled++
      }
      dmiss = want - dfilled
      if (dmiss < 0) dmiss = 0
      for (k=0;k<dmiss;k++) print wf, tasks, "daskvine", dask_arg

      dag_start = 4 + dask_cols
      gfilled=0
      for (r=1;r<=dag_cols;r++){
        v=$(dag_start + (r-1))
        if (v == "") break
        gfilled++
      }
      gmiss = want - gfilled
      if (gmiss < 0) gmiss = 0
      for (k=0;k<gmiss;k++) print wf, tasks, "dagvine", "dagvine"
    }
  ' "$part_file"
}

update_part_file_with_value() {
  # Append VALUE into the first empty repeat col for (workflow,tasks,executor),
  # expanding columns if needed. Recompute mean/stderr from all non-empty repeats.
  local workflow="$1"
  local tasks="$2"
  local executor_label="$3"   # daskvine | dagvine
  local value="$4"

  local part_file lock_file
  part_file="$(part_file_for_workflow "$workflow")"
  lock_file="$(part_lock_for_workflow "$workflow")"

  (
    flock -x 9

    local tmp="${part_file}.tmp"
    awk -v want="$REPEATS" -v target_tasks="$tasks" -v exec="$executor_label" -v val="$value" -F',' -v OFS=',' '
      function max(a,b){ return (a>b)?a:b }
      function isnum(x){ return (x ~ /^-?[0-9]+(\.[0-9]+)?$/) }
      function mean_stderr(n, sum, sumsq,    mean,var,stderr){
        if (n <= 0) return ""
        mean = sum / n
        if (n == 1) stderr = 0.0
        else {
          var = (sumsq - (sum * sum) / n) / (n - 1)
          if (var < 0) var = 0
          stderr = sqrt(var) / sqrt(n)
        }
        return sprintf("%.6f %.6f", mean, stderr)
      }
      NR==1{
        legacy = ($0 == "tasks,daskvine-mean,daskvine-stderr,dagvine-mean,dagvine-stderr")
        if (!legacy) {
          for (i=1;i<=NF;i++){
            if ($i ~ /^daskvine-r[0-9]+$/) dask_cols++
            if ($i ~ /^dagvine-r[0-9]+$/)  dag_cols++
          }
        } else {
          dask_cols=0; dag_cols=0
        }
        next
      }
      {
        rows++
        for (i=1;i<=NF;i++) R[rows,i]=$i
        Rn[rows]=NF
      }
      END{
        dask_cols=max(dask_cols, want)
        dag_cols=max(dag_cols, want)

        # Locate target row
        trow=0
        for (row=1; row<=rows; row++){
          if (R[row,1] == target_tasks) { trow=row; break }
        }

        # Determine current repeats content for target executor and first empty slot.
        slot=0
        if (trow != 0) {
          if (legacy) {
            # No repeats in legacy; slot becomes 1.
            slot=1
          } else if (exec == "daskvine") {
            for (r=1; r<=dask_cols; r++){
              v = R[trow,1+r]
              if (v == "") { slot=r; break }
            }
            if (slot==0) slot=dask_cols+1
          } else {
            dag_start = 4 + dask_cols
            for (r=1; r<=dag_cols; r++){
              v = R[trow,dag_start + (r-1)]
              if (v == "") { slot=r; break }
            }
            if (slot==0) slot=dag_cols+1
          }
        } else {
          slot=1
        }

        new_dask=dask_cols
        new_dag=dag_cols
        if (exec == "daskvine" && slot > new_dask) new_dask = slot
        if (exec == "dagvine" && slot > new_dag) new_dag = slot

        # Header
        printf "tasks"
        for (r=1;r<=new_dask;r++) printf ",daskvine-r%d", r
        printf ",daskvine-mean,daskvine-stderr"
        for (r=1;r<=new_dag;r++) printf ",dagvine-r%d", r
        printf ",dagvine-mean,dagvine-stderr\n"

        # Emit rows
        for (row=1; row<=rows; row++){
          tasks = R[row,1]
          printf "%s", tasks

          # Gather repeats arrays and apply update if this is target row.
          # Dask repeats
          for (r=1; r<=new_dask; r++){
            v = ""
            if (!legacy && r<=dask_cols) v = R[row,1+r]
            if (row==trow && exec=="daskvine" && r==slot) v = val
            D[r]=v
            printf ",%s", v
          }
          # Dag repeats
          for (r=1; r<=new_dag; r++){
            v = ""
            if (!legacy) {
              dag_start = 4 + dask_cols
              if (r<=dag_cols) v = R[row,dag_start + (r-1)]
            }
            if (row==trow && exec=="dagvine" && r==slot) v = val
            G[r]=v
            # defer printing until after dask mean/stderr
          }

          # Compute dask mean/stderr from ALL non-empty repeats
          dn=0; dsum=0; dsumsq=0
          for (r=1; r<=new_dask; r++){
            if (D[r] != "" && isnum(D[r])) { x=D[r]+0.0; dn++; dsum+=x; dsumsq+=x*x }
          }
          if (dn > 0) {
            split(mean_stderr(dn, dsum, dsumsq), tmp, " ")
            dm=tmp[1]; ds=tmp[2]
          } else if (legacy) {
            dm=R[row,2]; ds=R[row,3]
          } else {
            dm=""; ds=""
          }
          printf ",%s,%s", dm, ds

          # Print dag repeats
          for (r=1; r<=new_dag; r++) printf ",%s", G[r]

          # Compute dag mean/stderr
          gn=0; gsum=0; gsumsq=0
          for (r=1; r<=new_dag; r++){
            if (G[r] != "" && isnum(G[r])) { x=G[r]+0.0; gn++; gsum+=x; gsumsq+=x*x }
          }
          if (gn > 0) {
            split(mean_stderr(gn, gsum, gsumsq), tmp2, " ")
            gm=tmp2[1]; gs=tmp2[2]
          } else if (legacy) {
            gm=R[row,4]; gs=R[row,5]
          } else {
            gm=""; gs=""
          }
          printf ",%s,%s\n", gm, gs
        }
      }
    ' "$part_file" > "$tmp"
    mv -f "$tmp" "$part_file"
  ) 9>"$lock_file"
}

run_job() {
  local workflow="$1"
  local tasks="$2"
  local executor_label="$3"
  local executor_arg="$4"

  local v
  v="$(run_one "$executor_label" "$executor_arg" "$workflow" "$tasks")" || return 1
  update_part_file_with_value "$workflow" "$tasks" "$executor_label" "$v"
}

# ===== Start: prepare output directory (main process) =====
mkdir -p "$PARTS_DIR"

echo "========== Frontend overhead sweep: workflows=${#WORKFLOWS[@]} tasks_points=${#TASK_COUNTS[@]} threads=${THREADS} repeats=${REPEATS} ==========" >&2

# ===== General parallel execution: schedule missing tasks over threads =====
JOBS_FILE="$(mktemp)"
JOBS_SHUF_FILE="$(mktemp)"
trap 'rm -f "$JOBS_FILE" "$JOBS_SHUF_FILE"' EXIT

for workflow in "${WORKFLOWS[@]}"; do
  emit_missing_jobs_for_workflow "$workflow" >> "$JOBS_FILE"
done

if [[ ! -s "$JOBS_FILE" ]]; then
  echo "[done] no missing configs for repeats=${REPEATS}" >&2
  END_TIME_SEC=$(date +%s)
  TOTAL_TIME_USED_SEC=$((END_TIME_SEC - START_TIME_SEC))
  echo "Total time used: ${TOTAL_TIME_USED_SEC} seconds" >&2
  exit 0
fi

shuf "$JOBS_FILE" > "$JOBS_SHUF_FILE"

export RUN_INFO_PATH DASKVINE_EXECUTOR_ARG PARTS_DIR REPEATS
export -f \
  part_file_for_workflow \
  part_lock_for_workflow \
  run_one \
  update_part_file_with_value \
  run_job \
  random_run_info_template \
  extract_frontend_overhead_seconds

set +e
xargs -P "$THREADS" -n 4 bash -c 'run_job "$@"' _ < "$JOBS_SHUF_FILE"
rc=$?
set -e

failures=0
if (( rc != 0 )); then
  failures=1
  echo "[warn] some jobs failed (xargs rc=${rc})" >&2
fi

END_TIME_SEC=$(date +%s)
TOTAL_TIME_USED_SEC=$((END_TIME_SEC - START_TIME_SEC))
echo "Total time used: ${TOTAL_TIME_USED_SEC} seconds" >&2
if [[ "$failures" -ne 0 ]]; then
  echo "[warn] total failures=${failures}" >&2
  exit 1
fi

exit 0
