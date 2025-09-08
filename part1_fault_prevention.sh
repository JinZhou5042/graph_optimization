#!/bin/bash
set -euo pipefail
source part_config.sh

trap "echo 'Interrupted by Ctrl+C'; kill 0" SIGINT

command -v envsubst >/dev/null 2>&1 || { echo "error: envsubst not found"; exit 3; }

parse_args "$@" || exit $?
setup_workflow "$PARSED_WORKFLOW" || { echo "error: setup_workflow failed"; exit 4; }

# general settings
export PART_NAME="fault_prevention"
export REPEAT_NAME="$REPEAT_NAME"

export EXPERIMENT_PATH="$(printf '%s' "$EXPERIMENT_PATH_TEMPLATE" | envsubst '$PART_NAME $REPEAT_NAME $WORKFLOW_NAME')"
export RUN_INFO_PATH="/users/jzhou24/afs/taskvine-report-tool/logs/$EXPERIMENT_PATH"

echo "RUN_INFO_PATH: $RUN_INFO_PATH"

GENERAL_ARGS=(
    --manager-name "$MANAGER_NAME"
    --task-file "$TASK_FILE"
    --run-info-path "$RUN_INFO_PATH"
    --wait-for-workers "$WAIT_FOR_WORKERS"
    --temp-replica-count 1
    --extra-task-sleep-time "$EXTRA_TASK_SLEEP_TIME_MIN" "$EXTRA_TASK_SLEEP_TIME_MAX"
    --extra-task-output-size-mb "$EXTRA_TASK_OUTPUT_SIZE_MB_MIN" "$EXTRA_TASK_OUTPUT_SIZE_MB_MAX"
    --shared-file-system-dir "$SHARED_FILE_SYSTEM_DIR"
    --libcores "$LIB_CORES"
)

# run baseline
python3 main.py "${GENERAL_ARGS[@]}" \
    --run-info-template "baseline"

# test disk load balancing
python3 main.py "${GENERAL_ARGS[@]}" \
    --balance-worker-disk-load 1 \
    --run-info-template "disk-load-balance"


# test different priority modes with pruning
for priority_mode in "random" "largest-input-first"; do
    python3 main.py "${GENERAL_ARGS[@]}" \
        --balance-worker-disk-load 1 \
        --prune-depth 1 \
        --priority-mode "$priority_mode" \
        --run-info-template "priority-mode-${priority_mode}"
done


# baseline vs. disk load balancing vs. pruning
