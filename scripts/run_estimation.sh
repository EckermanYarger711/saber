#!/usr/bin/env bash
# Fit the belief stack and write the state-estimation read-out.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m saber.cli.estimate --run "${RUN:-configs/run/primary.yaml}" \
  --out "${OUT:-artefacts}" --scale "${SCALE:-1}" --steps "${STEPS:-40}"
