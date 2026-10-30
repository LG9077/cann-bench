#!/usr/bin/env bash
set -e -o pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
submission=$(cd -- "${1:-$repo/submission}" && pwd)
image=${IMAGE:-cannbench:9.2.0-a3}

if [[ ! -f "$submission/build.sh" ]]; then
    printf 'Missing build.sh in %s\n' "$submission" >&2
    exit 1
fi

docker run --rm --init --pull=never \
    --user "$(id -u):$(id -g)" \
    --env HOME=/tmp --env PYTHONUNBUFFERED=1 \
    --mount "type=bind,src=$submission,dst=/workspace/submission" \
    --workdir /workspace/submission \
    --entrypoint /bin/bash "$image" -e -o pipefail -c '
        umask 022
        source "$ASCEND_HOME_PATH/set_env.sh"
        exec bash build.sh
    ' 2>&1 | tee "$submission/_compile.log"

printf '[PASS] Build completed. Log: %s\n' "$submission/_compile.log"
