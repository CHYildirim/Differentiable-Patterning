#!/bin/bash

set -euo pipefail

usage() {
    echo "Usage: $0 --cluster eidf --port PORT [--no-reindex]"
    echo "  --port is the port printed by 'kubectl pvcsync <user>-ws1 up' on EIDF."
    echo "  --no-reindex skips rebuilding models/registry.sqlite after the copy."
}

cluster=""
port=""
reindex=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--cluster)
            [[ $# -ge 2 ]] || { usage >&2; exit 1; }
            cluster="$2"
            shift 2
            ;;
        -p|--port)
            [[ $# -ge 2 ]] || { usage >&2; exit 1; }
            port="$2"
            shift 2
            ;;
        --no-reindex)
            reindex=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

case "$cluster" in
    eidf)
        if [[ ! "$port" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
            echo "A valid --port (1-65535) is required for EIDF." >&2
            usage >&2
            exit 1
        fi
        mkdir -p ./models/bundles
        rsync -a -e "ssh -J eidf -p $port" root@localhost:/data/Models/bundles/ ./models/bundles/
        ;;
    *)
        echo "--cluster must be 'eidf'." >&2
        usage >&2
        exit 1
        ;;
esac

if [[ "$reindex" == true ]]; then
    python -m Experiments.model_registry --root ./models reindex
fi
