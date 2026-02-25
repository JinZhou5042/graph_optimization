#!/usr/bin/env bash

# NOTE: avoid `set -u` because conda activate/deactivate scripts may reference
# unset backup vars and fail with "unbound variable".
set -eo pipefail

unset PYTHONPATH
eval "$(conda shell.bash hook)"

ENV_NAME="dagvine-env"
CONDA_BIN="${HOME}/miniconda3/bin/conda"
if [[ ! -x "${CONDA_BIN}" ]]; then
    echo "ERROR: conda binary not found at ${CONDA_BIN}" >&2
    exit 1
fi
CONDA_BASE="$("${CONDA_BIN}" info --base)"

ENV_PATH="${CONDA_BASE}/envs/${ENV_NAME}"

# Exit immediately if target env path already exists.
if [[ -d "${ENV_PATH}" ]]; then
    echo "Conda env path already exists: ${ENV_PATH}. Exit without changes."
    exit 0
fi

# Warn if a nested env exists under the currently active env path.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    NESTED_ENV_PATH="${CONDA_PREFIX}/envs/${ENV_NAME}"
    if [[ "${NESTED_ENV_PATH}" != "${ENV_PATH}" && -f "${NESTED_ENV_PATH}/conda-meta/history" ]]; then
        echo "WARNING: found nested env at ${NESTED_ENV_PATH}. Target env path is ${ENV_PATH}."
    fi
fi

# If the environment directory exists but is not a real conda env, remove it.
if [[ -d "${ENV_PATH}" && ! -f "${ENV_PATH}/conda-meta/history" ]]; then
    echo "Detected broken conda env at ${ENV_PATH}. Recreating ${ENV_NAME}..."
    rm -rf "${ENV_PATH}"
fi

# Create conda environment by explicit prefix to avoid nested env paths.
conda create -y -p "${ENV_PATH}" -c conda-forge --strict-channel-priority python=3.11.12 conda-pack

# Ensure conda activation works
conda activate "${ENV_PATH}"
if [[ "${CONDA_PREFIX:-}" != "${ENV_PATH}" ]]; then
    echo "ERROR: current conda prefix is '${CONDA_PREFIX:-<none>}', expected '${ENV_PATH}'." >&2
    exit 1
fi

echo "Current conda env: ${CONDA_DEFAULT_ENV:-<none>} (${CONDA_PREFIX})"

# Install build requirements for cctools
conda install -y -c conda-forge --strict-channel-priority \
    gcc_linux-64 gxx_linux-64 gdb m4 perl swig make zlib \
    libopenssl-static openssl conda-pack packaging cloudpickle \
    flake8 clang-format threadpoolctl pympler pytz
python -c "import conda_pack"

echo "Cloning cctools..."
if [[ ! -d cctools-dagvine ]]; then
    git clone https://github.com/JinZhou5042/cctools.git cctools-dagvine
fi
cd cctools-dagvine
git fetch --all
git checkout task-graph
./configure --with-base-dir "${CONDA_PREFIX}" --prefix "${CONDA_PREFIX}"
make clean
make -j8
make install
vine_worker --version
echo "Done. cctools-dagvine installed in ${CONDA_PREFIX}"
cd ..

# Install environment for dagvine
pip install dask==2024.7.1 dask-awkward==2024.7.0 awkward==2.6.6 coffea==2024.4.0 fastjet==3.4.3.1 numpy==2.2.6

# Package conda environment for taskvine workers
poncho_package_create "${CONDA_PREFIX}" factories/dagvine-env.tar.gz

# complete
echo "Done. dagvine environment created in ${CONDA_PREFIX}, run 'conda activate ${ENV_NAME}' to activate the environment."