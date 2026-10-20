#!/usr/bin/env bash
# Unified task entry point; select RGB or SAR with --modal.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
run_task "SEG" "$@"
