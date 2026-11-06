#!/bin/sh
set -eu
data_dir="${SPACEFL_DATA_DIR:-/var/lib/spacefl}"
if [ "$data_dir" != "/var/lib/spacefl" ]; then
    echo "Training image requires SPACEFL_DATA_DIR=/var/lib/spacefl" >&2
    exit 1
fi
mkdir -p "$data_dir"
chown spacefl:spacefl "$data_dir"
# One process owns the shared queue. Do not add --workers or multiple instances.
exec gosu spacefl python -m uvicorn --factory web.server:create_app \
    --host 0.0.0.0 --port "${PORT:-8720}" --workers 1 \
    --proxy-headers --forwarded-allow-ips='*' --log-level warning
