MANAGER_NAME=jzhou24-default
N_WORKERS=40
CORES=12
MEMORY_GB=10
DISK_GB=100
PONCHO_ENV=dv5-new.tar.gz

# Ignore SIGPIPE so broken pipes don't terminate the script.
trap '' PIPE

while [ $# -gt 0 ]; do
    case "$1" in
        -M)
            [ $# -ge 2 ] || exit 2
            MANAGER_NAME="$2"
            shift 2
            ;;
        --manager-name)
            [ $# -ge 2 ] || exit 2
            MANAGER_NAME="$2"
            shift 2
            ;;
        --cores)
            [ $# -ge 2 ] || exit 2
            CORES="$2"
            shift 2
            ;;
        --memory)
            [ $# -ge 2 ] || exit 2
            MEMORY_GB="$2"
            shift 2
            ;;
        --disk)
            [ $# -ge 2 ] || exit 2
            DISK_GB="$2"
            shift 2
            ;;
        --workers)
            [ $# -ge 2 ] || exit 2
            N_WORKERS="$2"
            shift 2
            ;;
        --poncho-env)
            [ $# -ge 2 ] || exit 2
            PONCHO_ENV="$2"
            shift 2
            ;;
        -*)
            exit 2
            ;;
        *)
            break
            ;;
    esac
done

MEMORY=$((MEMORY_GB*1024))
DISK=$((DISK_GB*1024))

vine_cmd=(
    vine_factory
    -T condor
    --scratch-dir /scratch365/jzhou24/factory_dv5
    --poncho-env "$PONCHO_ENV"
    --wrapper /users/jzhou24/graph_optimization/factories/gdb_wrapper.sh
    --condor-requirements '((has_vast))'
    --manager-name "$MANAGER_NAME"
    --min-workers "$N_WORKERS"
    --max-workers "$N_WORKERS"
    --workers-per-cycle "$N_WORKERS"
    --cores "$CORES"
    --memory "$MEMORY"
    --disk "$DISK"
    --timeout 36000
)

"${vine_cmd[@]}"

