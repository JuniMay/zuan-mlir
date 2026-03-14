#!/usr/bin/env bash

set -euo pipefail

# This script exists because the repository went through a historical rename
# from Zuan to Dyno. It is a lightweight guard for that specific refactor, not
# a general-purpose repository lint.
#
# Optional line-based ignore list. Each entry is a repository-relative path
# that is allowed to keep legacy Zuan spellings because the file is being kept
# as historical reference material for the refactor.
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

IGNORE_FILE=${1:-}

ignore_globs=()
ignore_paths=()
if [[ -n "${IGNORE_FILE}" ]]; then
  if [[ ! -f "${IGNORE_FILE}" ]]; then
    echo "Ignore file not found: ${IGNORE_FILE}" >&2
    exit 1
  fi

  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" ]] && continue
    [[ "${line}" =~ ^[[:space:]]*# ]] && continue
    ignore_globs+=(--glob "!${line}")
    ignore_paths+=("${line}")
  done < "${IGNORE_FILE}"
fi

path_roots=(
  include
  lib
  tools
  test
  benchmarks
  scripts
  cmake
  runtime
  regression
)

content_roots=(
  CMakeLists.txt
  README.md
  include
  lib
  tools
  test
  benchmarks
  scripts
  cmake
  runtime
  regression
)

# Drop path matches that were explicitly allowed in the historical ignore list.
filter_ignored_paths() {
  local input=${1:-}

  if [[ -z "${input}" ]] || [[ ${#ignore_paths[@]} -eq 0 ]]; then
    printf '%s' "${input}"
    return
  fi

  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    local skip=0
    for ignored in "${ignore_paths[@]}"; do
      if [[ "${line}" == "${ignored}" ]]; then
        skip=1
        break
      fi
    done
    if [[ ${skip} -eq 0 ]]; then
      printf '%s\n' "${line}"
    fi
  done <<< "${input}"
}

# Canonical in-tree paths should not keep old spellings after the historical
# Zuan-to-Dyno refactor.
path_hits=$(
  find "${path_roots[@]}" -print 2>/dev/null \
    | sed "s#^${ROOT_DIR}/##" \
    | rg "(^|/).*([Zz]uan|zuan-).*" || true
)
path_hits=$(filter_ignored_paths "${path_hits}")

# Project-owned content should also be Dyno-only, except for files that are
# intentionally preserved as historical reference through the ignore list.
content_hits=$(
  rg -n \
    --glob '!scripts/check-workspace-rename.sh' \
    "${ignore_globs[@]}" \
    "(mlir::zuan|!zuan\\.tile|\\bZUAN_[A-Z0-9_]+\\b|\\bzuan-opt\\b|\\bzuan-translate\\b|\\bzuan-lsp\\b|\\bzuan\\.|\\bnamespace zuan\\b|\\bZuan\\b)" \
    "${content_roots[@]}" || true
)

if [[ -n "${path_hits}" ]]; then
  echo "Found paths that still use legacy spellings:"
  echo "${path_hits}"
  exit 1
fi

if [[ -n "${content_hits}" ]]; then
  echo "Found files that still use legacy spellings:"
  echo "${content_hits}"
  exit 1
fi

echo "Rename integrity check passed."
