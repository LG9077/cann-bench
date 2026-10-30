#!/usr/bin/env bash
# Sourced by BASH_ENV for independent Agent and evaluation shells.
export PYTHONPATH="/workspace/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    export PATH="${ASCEND_HOME_PATH}/tools/profiler/bin:$PATH"
    # Installation is a separate process; load its environment in every new shell.
    ascopt_vendor="${ASCEND_OPP_PATH:-$ASCEND_HOME_PATH/opp}/vendors/custom_ops"
    if [[ -r "$ascopt_vendor/bin/set_env.bash" ]]; then
        source "$ascopt_vendor/bin/set_env.bash"
        for ascopt_lib in "$ascopt_vendor/op_api/lib" \
            "$ascopt_vendor/op_impl/ai_core/tbe/op_tiling" \
            "$ascopt_vendor/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64"; do
            if [[ -d "$ascopt_lib" ]]; then
                export LD_LIBRARY_PATH="$ascopt_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            fi
        done
    fi
    unset ascopt_vendor ascopt_lib
fi
