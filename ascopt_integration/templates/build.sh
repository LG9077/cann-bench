#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
: "${ASCEND_HOME_PATH:?Initialize CANN before building}"
# Fixed target: build never probes or reserves an NPU.
bash scripts/build_run.sh
bash scripts/build_wheel.sh
