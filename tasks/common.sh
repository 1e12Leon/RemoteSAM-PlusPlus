#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${TASK_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
cd "${TASK_ROOT}"

run_task() {
    local task="$1"
    shift
    exec "${PYTHON_BIN:-python}" "tasks/code/eval/${task}.py" "$@"
}
