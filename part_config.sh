setup_workflow() {
    local wf="$1"
    case "$wf" in
    DV5)
        export WORKFLOW_NAME=DV5
        export MANAGER_NAME="jzhou24-dv5"
        export TASK_FILE="task-graph-pkls/dv5.pkl"
        export WAIT_FOR_WORKERS=40
        export LIB_CORES=16
        export EXTRA_TASK_SLEEP_TIME_MIN=0
        export EXTRA_TASK_SLEEP_TIME_MAX=8
        export EXTRA_TASK_OUTPUT_SIZE_MB_MIN=0
        export EXTRA_TASK_OUTPUT_SIZE_MB_MAX=20
        export SHARED_FILE_SYSTEM_DIR="/project01/ndcms/jzhou24/sharedfs-dv5"
        ;;
    RSTriPhoton)
        export WORKFLOW_NAME=RSTriPhoton
        export MANAGER_NAME="jzhou24-rstriphoton"
        export TASK_FILE="task-graph-pkls/rstriphoton.pkl"
        export WAIT_FOR_WORKERS=80
        export LIB_CORES=8
        export EXTRA_TASK_SLEEP_TIME_MIN=0
        export EXTRA_TASK_SLEEP_TIME_MAX=5
        export EXTRA_TASK_OUTPUT_SIZE_MB_MIN=0
        export EXTRA_TASK_OUTPUT_SIZE_MB_MAX=50
        export SHARED_FILE_SYSTEM_DIR="/project01/ndcms/jzhou24/sharedfs-rstriphoton"
        ;;
    BinaryForest)
        export WORKFLOW_NAME=BinaryForest
        export MANAGER_NAME="jzhou24-binary-forest"
        export TASK_FILE="task-graph-pkls/binary_forest.pkl"
        export WAIT_FOR_WORKERS=20
        export LIB_CORES=32
        export EXTRA_TASK_SLEEP_TIME_MIN=0
        export EXTRA_TASK_SLEEP_TIME_MAX=3
        export EXTRA_TASK_OUTPUT_SIZE_MB_MIN=0
        export EXTRA_TASK_OUTPUT_SIZE_MB_MAX=20
        export SHARED_FILE_SYSTEM_DIR="/project01/ndcms/jzhou24/sharedfs-binary-forest"
        ;;
    *)
        echo "Error: unknown workflow: $wf" >&2
        return 1
        ;;
    esac
}

print_usage_common() {
    echo "usage: $0 --workflow {DV5|RSTriPhoton|BinaryForest} --repeat-id INT" >&2
}

parse_args() {
    local workflow=""
    local repeat_id=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
        --workflow)
            shift
            workflow="${1:-}"
            ;;
        --repeat-id)
            shift
            repeat_id="${1:-}"
            ;;
        -h|--help)
            print_usage_common
            return 2
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            print_usage_common
            return 2
            ;;
        esac
        shift
    done

    if [[ -z "$workflow" || -z "$repeat_id" ]]; then
        echo "error: missing required arguments" >&2
        print_usage_common
        return 2
    fi

    if [[ "$workflow" != "DV5" && "$workflow" != "RSTriPhoton" && "$workflow" != "BinaryForest" ]]; then
        echo "error: workflow must be DV5 or RSTriPhoton or BinaryForest" >&2
        print_usage_common
        return 2
    fi

    if ! [[ "$repeat_id" =~ ^[0-9]+$ ]]; then
        echo "error: --repeat-id must be an integer" >&2
        print_usage_common
        return 2
    fi

    export PARSED_WORKFLOW="$workflow"
    export REPEAT_NAME="repeat${repeat_id}"
}

export EXPERIMENT_PATH_TEMPLATE='$PART_NAME/$WORKFLOW_NAME/$REPEAT_NAME/'