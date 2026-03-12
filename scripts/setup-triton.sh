#!/usr/bin/env bash

set -euo pipefail

SOURCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR_INPUT=${1:-"${SOURCE_DIR}/build-riscv"}

mkdir -p "${BUILD_DIR_INPUT}"
BUILD_DIR=$(cd "${BUILD_DIR_INPUT}" && pwd)

TRITON_SOURCE_DIR="${SOURCE_DIR}/third_party/triton"
TRITON_CPU_SOURCE_DIR="${SOURCE_DIR}/third_party/triton-cpu"
TRITON_SHARED_SOURCE_DIR="${SOURCE_DIR}/third_party/triton_shared"
TRITON_ROOT="${BUILD_DIR}/third_party/triton"
TRITON_SHARED_ROOT="${TRITON_ROOT}/shared"
TRITON_SHARED_VENV_DIR="${TRITON_SHARED_ROOT}/venv"
TRITON_SHARED_BUILD_DIR="${TRITON_SHARED_ROOT}/build"
TRITON_SHARED_HOME="${TRITON_SHARED_ROOT}/home"
TRITON_CPU_ROOT="${TRITON_ROOT}/cpu"
TRITON_CPU_VENV_DIR="${TRITON_CPU_ROOT}/venv"
TRITON_CPU_BUILD_DIR="${TRITON_CPU_ROOT}/build"
TRITON_CPU_HOME="${TRITON_CPU_ROOT}/home"
MAX_JOBS="${MAX_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

git -C "${SOURCE_DIR}" submodule update --init --recursive third_party/triton third_party/triton-cpu third_party/triton_shared

EXPECTED_TRITON_COMMIT=$(tr -d '\n' < "${TRITON_SHARED_SOURCE_DIR}/triton-hash.txt")
CURRENT_TRITON_COMMIT=$(git -C "${TRITON_SOURCE_DIR}" rev-parse HEAD)
if [[ "${CURRENT_TRITON_COMMIT}" != "${EXPECTED_TRITON_COMMIT}" ]]; then
  echo "third_party/triton is at ${CURRENT_TRITON_COMMIT}, expected ${EXPECTED_TRITON_COMMIT}" >&2
  exit 1
fi

resolve_llvm_cache() {
  local cache_root="$1"
  local link_target
  link_target=$(find "${cache_root}/.triton/llvm" -mindepth 1 -maxdepth 1 -type l -name 'llvm-*' -print -quit)
  if [[ -z "${link_target}" ]]; then
    link_target=$(find "${cache_root}/.triton/llvm" -mindepth 1 -maxdepth 1 -type d -name 'llvm-*' -print -quit)
  fi
  if [[ -z "${link_target}" ]]; then
    echo "failed to locate the Triton LLVM cache under ${cache_root}/.triton/llvm" >&2
    exit 1
  fi
  printf '%s\n' "$(readlink -f "${link_target}")"
}

python3 -m venv "${TRITON_SHARED_VENV_DIR}"
"${TRITON_SHARED_VENV_DIR}/bin/python" -m pip install --upgrade pip
"${TRITON_SHARED_VENV_DIR}/bin/python" -m pip install -r "${TRITON_SOURCE_DIR}/python/requirements.txt"

TRITON_PLUGIN_DIRS="${TRITON_SHARED_SOURCE_DIR}" \
TRITON_BUILD_DIR="${TRITON_SHARED_BUILD_DIR}" \
TRITON_HOME="${TRITON_SHARED_HOME}" \
CFLAGS="${CFLAGS:-} -Wno-error=deprecated-declarations" \
CXXFLAGS="${CXXFLAGS:-} -Wno-error=deprecated-declarations" \
MAX_JOBS="${MAX_JOBS}" \
"${TRITON_SHARED_VENV_DIR}/bin/python" -m pip install -e "${TRITON_SOURCE_DIR}" --no-build-isolation

mkdir -p "${TRITON_SHARED_ROOT}"
ln -sfn "$(resolve_llvm_cache "${TRITON_SHARED_HOME}")" "${TRITON_SHARED_ROOT}/llvm"

python3 -m venv "${TRITON_CPU_VENV_DIR}"
"${TRITON_CPU_VENV_DIR}/bin/python" -m pip install --upgrade pip
"${TRITON_CPU_VENV_DIR}/bin/python" -m pip install -r "${TRITON_CPU_SOURCE_DIR}/python/requirements.txt"

TRITON_BUILD_DIR="${TRITON_CPU_BUILD_DIR}" \
TRITON_HOME="${TRITON_CPU_HOME}" \
TRITON_BUILD_PROTON=OFF \
CFLAGS="${CFLAGS:-} -Wno-error=deprecated-declarations" \
CXXFLAGS="${CXXFLAGS:-} -Wno-error=deprecated-declarations" \
MAX_JOBS="${MAX_JOBS}" \
"${TRITON_CPU_VENV_DIR}/bin/python" -m pip install -e "${TRITON_CPU_SOURCE_DIR}" --no-build-isolation

mkdir -p "${TRITON_CPU_ROOT}"
ln -sfn "$(resolve_llvm_cache "${TRITON_CPU_HOME}")" "${TRITON_CPU_ROOT}/llvm"

test -x "${TRITON_SHARED_VENV_DIR}/bin/python"
test -x "${TRITON_SHARED_BUILD_DIR}/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt"
test -x "${TRITON_SHARED_ROOT}/llvm/bin/mlir-opt"
test -x "${TRITON_SHARED_ROOT}/llvm/bin/mlir-translate"
test -x "${TRITON_SHARED_ROOT}/llvm/bin/opt"
test -x "${TRITON_SHARED_ROOT}/llvm/bin/llc"
test -x "${TRITON_CPU_BUILD_DIR}/bin/triton-opt"
test -x "${TRITON_CPU_ROOT}/llvm/bin/mlir-translate"

cat <<EOF
Triton setup complete for ${BUILD_DIR}
  Shared Python: ${TRITON_SHARED_VENV_DIR}/bin/python
  triton-shared-opt: ${TRITON_SHARED_BUILD_DIR}/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt
  Triton Shared LLVM bin: ${TRITON_SHARED_ROOT}/llvm/bin
  triton-opt: ${TRITON_CPU_BUILD_DIR}/bin/triton-opt
  Triton CPU LLVM bin: ${TRITON_CPU_ROOT}/llvm/bin
EOF
