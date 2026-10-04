#!/usr/bin/env bash
# Run the verification driver twice: the first pass lays down the artefacts the suite's
# manifest test asserts against, and the second pass writes the shipped manifest.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m saber.cli.verify --run "${RUN:-configs/run/primary.yaml}" --scale "${SCALE:-1}"
python3 -m saber.cli.verify --run "${RUN:-configs/run/primary.yaml}" --scale "${SCALE:-1}"
