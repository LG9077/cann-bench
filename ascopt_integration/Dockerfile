FROM quay.io/ascend/cann:9.1.0-a3-ubuntu22.04-py3.12-devel

USER root
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]

RUN --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
    umask 022 && \
    python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# Build outputs stay on the temporary mount; only the installed wheel is retained.
RUN --mount=type=bind,source=src/cann_bench_utils,target=/tmp/cann_bench_utils,rw \
    umask 022 && \
    source "$ASCEND_HOME_PATH/set_env.sh" && \
    PYTHON=python3 bash /tmp/cann_bench_utils/build.sh --soc=ascend910_93 --clean && \
    python3 -m pip install --no-cache-dir --no-deps /tmp/cann_bench_utils/dist/cann_bench_utils*.whl && \
    python3 -c 'import torch, torch_npu, cann_bench_utils'

RUN --mount=type=bind,source=ascopt_integration/msprof,target=/tmp/msprof \
    umask 022 && \
    source "$ASCEND_HOME_PATH/set_env.sh" && \
    bash /tmp/msprof/install.sh "$ASCEND_HOME_PATH" && \
    "$ASCEND_HOME_PATH/tools/profiler/bin/msprof" --help >/dev/null && \
    chmod -R go-w "$ASCEND_HOME_PATH" \
      "$(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"

WORKDIR /workspace
