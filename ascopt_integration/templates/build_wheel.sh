#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# setup.py configures CMake again for the Python extension.
python -c 'import os; assert os.environ.get("ASCEND_HOME_PATH"), "CANN environment is missing"'
python setup.py bdist_wheel --dist-dir dist
shopt -s nullglob
wheels=(dist/cann_bench*.whl)
if (( ${#wheels[@]} != 1 )); then
    echo "Expected exactly one fresh cann_bench wheel" >&2
    exit 1
fi
