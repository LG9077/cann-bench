#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: bash install.sh [absolute CANN directory; default: /usr/local/Ascend/cann]"
    exit 0
fi
if (( $# > 1 )); then
    echo "Expected at most one argument: the CANN installation directory" >&2
    exit 2
fi
if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "aarch64" ]]; then
    echo "This package requires Linux aarch64; run it inside the target ARM64 image" >&2
    exit 1
fi

install_path="${1:-/usr/local/Ascend/cann}"
if [[ "$install_path" != /* || ! -f "$install_path/set_env.sh" ]]; then
    echo "Pass an absolute existing CANN directory containing set_env.sh" >&2
    exit 1
fi
if [[ ! -w "$install_path" ]]; then
    echo "CANN is not writable: $install_path; prepare a writable container image with msprof installed" >&2
    exit 1
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if ! sha256sum --check --status SHA256SUMS; then
    echo "Profiler package is missing or its SHA256 does not match" >&2
    exit 1
fi

# Startup runs as root; the non-root Agent must also be able to use the profiler.
exec bash ./mindstudio-profiler_26.1.0_aarch64.run --quiet --install --install-for-all "--install-path=$install_path"
