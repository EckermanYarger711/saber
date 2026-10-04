#!/usr/bin/env bash
# Run the coupled twin, its fidelity report and the off-policy evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m saber.cli.twin --run "${RUN:-configs/run/primary.yaml}" \
  --out "${OUT:-runs}"
