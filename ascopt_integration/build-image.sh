#!/usr/bin/env bash
set -e -o pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
image=cannbench:9.2.0-a3
platform=$(docker info --format '{{.OSType}}/{{.Architecture}}')
case "$platform" in
    linux/aarch64|linux/arm64) ;;
    *)
        printf 'This image requires a Linux ARM64 Docker host; found %s. The msprof installer is aarch64.\n' "$platform" >&2
        exit 1
        ;;
esac

docker build --platform=linux/arm64 --progress=plain \
    -f "$repo/ascopt_integration/Dockerfile" -t "$image" "$repo"

docker run --rm -i --user 10000:10000 --cap-drop=ALL \
    --security-opt=no-new-privileges --network=none \
    --mount "type=bind,src=$repo,dst=/workspace,readonly" \
    --workdir /workspace \
    --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONNOUSERSITE=1 \
    --entrypoint /bin/bash "$image" -s <<'SMOKE'
set -e -o pipefail
HOME=$(mktemp -d /tmp/cannbench-smoke.XXXXXX)
export HOME
source "$ASCEND_HOME_PATH/set_env.sh"
source /workspace/ascopt_integration/env.sh

test "$(id -u)" -eq 10000
printf '[PASS] Non-root user: %s\n' "$(id)"

msprof --help >/dev/null
printf '[PASS] msprof --help\n'

python3 - <<'PY'
import importlib
import os
from pathlib import Path
import sysconfig

for name in ("torch", "torch_npu", "cann_bench_utils", "kernel_eval"):
    module = importlib.import_module(name)
    print(f"[PASS] Import {name}: {module.__file__}", flush=True)

for path in (Path(os.environ["ASCEND_HOME_PATH"]), Path(sysconfig.get_path("purelib"))):
    if path.stat().st_uid != 0 or os.access(path, os.W_OK):
        raise SystemExit(f"[FAIL] Expected a root-owned directory not writable by the Agent: {path}")
    print(f"[PASS] Root-owned directory not writable by the Agent: {path}", flush=True)
PY

python3 -m kernel_eval.cli --help >/dev/null
printf '[PASS] CannBench CLI\n'
python3 -m ascopt_integration.evaluate --help >/dev/null
printf '[PASS] Ascopt evaluation CLI\n'
SMOKE

printf '[PASS] %s built; non-root import/CLI checks passed. NPU execution was not tested.\n' "$image"
