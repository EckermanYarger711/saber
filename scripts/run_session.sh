#!/usr/bin/env bash
# Run one acquisition configuration and print its comparison and frontier report.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m saber.cli.monitor --run "${RUN:-configs/run/primary.yaml}" \
  --out "${OUT:-runs}" --scale "${SCALE:-1}"
