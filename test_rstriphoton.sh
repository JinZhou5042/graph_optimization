#!/bin/bash
set -euo pipefail
source part_config.sh

trap "echo 'Interrupted by Ctrl+C'; kill 0" SIGINT

command -v envsubst >/dev/null 2>&1 || { echo "error: envsubst not found"; exit 3; }

parse_args "$@" || exit $?
setup_workflow "$PARSED_WORKFLOW" || { echo "error: setup_workflow failed"; exit 4; }

# general settings
export PART_NAME="fault_tolerance"
export REPEAT_NAME="$REPEAT_NAME"
export EXPERIMENT_PATH="$(printf '%s' "$EXPERIMENT_PATH_TEMPLATE" | envsubst '$PART_NAME $REPEAT_NAME $WORKFLOW_NAME')"
echo "Experiment path: $EXPERIMENT_PATH"
export RUN_INFO_PATH="/users/jzhou24/afs/taskvine-report-tool/logs/$EXPERIMENT_PATH"

GENERAL_ARGS=(
    --manager-name "$MANAGER_NAME"
    --task-file "$TASK_FILE"
    --run-info-path "$RUN_INFO_PATH"
    --priority-mode largest-input-first
    --wait-for-workers "$WAIT_FOR_WORKERS"
    --extra-task-sleep-time "$EXTRA_TASK_SLEEP_TIME_MIN" "$EXTRA_TASK_SLEEP_TIME_MAX"
    --extra-task-output-size-mb "$EXTRA_TASK_OUTPUT_SIZE_MB_MIN" "$EXTRA_TASK_OUTPUT_SIZE_MB_MAX"
    --shared-file-system-dir "$SHARED_FILE_SYSTEM_DIR"
    --libcores "$LIB_CORES"
    --prune-depth 1
    --temp-replica-count 1
    --balance-worker-disk-load 1
    --watch-library-logfiles 0
    --failure-injection-step-percent 2.0
)

# test different checkpoint percentages
for checkpoint_percentage in  0.5 0.6 0.7 0.8 0.9 1.0; do
    python3 main.py "${GENERAL_ARGS[@]}" \
        --pfs-percentage "$checkpoint_percentage" \
        --run-info-template "checkpoint-percentage-${checkpoint_percentage}"
done
