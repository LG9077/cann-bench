#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# Generated output directories only. Failures must not reuse an earlier package.
rm -rf -- build build_py dist
mkdir -p dist
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DASCEND_COMPUTE_UNIT=ascend910_93
cmake --build build --parallel "$(nproc)"
cmake --build build --target binary --parallel "$(nproc)"
cmake --build build --target package
mapfile -t packages < <(find build dist -type f -name '*.run' -not -path '*/_CPack_Packages/*')
if (( ${#packages[@]} != 1 )); then
    printf 'Expected exactly one fresh .run package, found %s\n' "${#packages[@]}" >&2
    exit 1
fi
if [[ "${packages[0]}" != "dist/cann_bench_custom_ops.run" ]]; then
    mv -- "${packages[0]}" dist/cann_bench_custom_ops.run
fi
