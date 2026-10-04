#!/usr/bin/env bash
# Fit the response head and write both directed transfer tables.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m saber.cli.transfer --run "${RUN:-configs/run/primary.yaml}" \
  --out "${OUT:-artefacts}" --scale "${SCALE:-1}" --epochs "${EPOCHS:-40}"
