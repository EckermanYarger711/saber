#!/usr/bin/env bash
# Write the release cohort and the resource register for one run selection.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m saber.cli.build_cohort --run "${RUN:-configs/run/primary.yaml}" \
  --out "${OUT:-cohort}" --scale "${SCALE:-1}"
