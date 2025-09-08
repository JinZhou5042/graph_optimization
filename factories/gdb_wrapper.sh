#!/bin/bash
trap '' SIGPIPE

unset PYTHONPATH

ip=$(hostname -I 2>/dev/null | awk '{print $1}' | tr '.' '_')
if [[ -z "$ip" ]]; then
    echo "error: unable to determine IP address" >&2
    exit 1
fi

logdir="/scratch365/jzhou24/worker_stdout/"
mkdir -p "$logdir"

logfile=$(mktemp "${logdir}/gdb-${ip}.XXXXXXXX.log")

echo "wrapper args: $@" > "$logfile"

if [[ "$1" == *poncho_package_run ]]; then
    shift
    shift
    shift
fi

exec gdb -batch \
    -ex "handle SIGPIPE nostop noprint ignore" \
    -ex "set logging file $logfile" \
    -ex "set logging on" \
    -ex run \
    -ex bt \
    --args "$@"
