#!/usr/bin/env bash
set -euo pipefail

# Lightweight SpaceFL cloud experiment runner.
# Override any variable before execution, for example:
#   DEVICE=cuda ALGO=fedprox ROUNDS=120 bash scripts/run_cloud_experiment.sh

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_output/cloud_lightweight}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif [[ -x /root/miniconda3/bin/python ]]; then
    PYTHON_BIN="/root/miniconda3/bin/python"
  elif [[ -x /opt/conda/bin/python ]]; then
    PYTHON_BIN="/opt/conda/bin/python"
  else
    echo "No Python executable found. Set PYTHON_BIN=/path/to/python and rerun." >&2
    exit 1
  fi
fi

ALGO="${ALGO:-fedavg}"
DATASET="${DATASET:-mnist}"
DATA_DIR="${DATA_DIR:-./data}"
DEVICE="${DEVICE:-cpu}"
ROUNDS="${ROUNDS:-120}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.01}"
SEED="${SEED:-42}"

GS_LIST="${GS_LIST:-3 5}"
SAT_LIST="${SAT_LIST:-3 5}"
SIM_HOURS="${SIM_HOURS:-3}"
TIMESLOT_MIN="${TIMESLOT_MIN:-1}"

PARTITION_STRATEGY="${PARTITION_STRATEGY:-probability}"
CLASS_PROBABILITY="${CLASS_PROBABILITY:-0.8}"
PREFERENCE_MODE="${PREFERENCE_MODE:-class_balanced}"
PREFERRED_CLIENTS_PER_CLASS="${PREFERRED_CLIENTS_PER_CLASS:-1}"
SAMPLE_CAP_STRATEGY="${SAMPLE_CAP_STRATEGY:-preserve}"
CLASSES_PER_CLIENT="${CLASSES_PER_CLIENT:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"

cd "${PROJECT_DIR}"

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -e ".[full]"

"${PYTHON_BIN}" -m fl_space.cli tune reset
"${PYTHON_BIN}" -m fl_space.cli mount clear
"${PYTHON_BIN}" -m fl_space.cli mount algo "${ALGO}"
"${PYTHON_BIN}" -m fl_space.cli mount sim-hours "${SIM_HOURS}"
"${PYTHON_BIN}" -m fl_space.cli mount timeslot-min "${TIMESLOT_MIN}"
"${PYTHON_BIN}" -m fl_space.cli tune dataset "${DATASET}"
"${PYTHON_BIN}" -m fl_space.cli tune data-dir "${DATA_DIR}"
"${PYTHON_BIN}" -m fl_space.cli tune rounds "${ROUNDS}"
"${PYTHON_BIN}" -m fl_space.cli tune epochs "${EPOCHS}"
"${PYTHON_BIN}" -m fl_space.cli tune batch "${BATCH_SIZE}"
"${PYTHON_BIN}" -m fl_space.cli tune lr "${LR}"
"${PYTHON_BIN}" -m fl_space.cli tune seed "${SEED}"
"${PYTHON_BIN}" -m fl_space.cli tune device "${DEVICE}"
"${PYTHON_BIN}" -m fl_space.cli tune partition-strategy "${PARTITION_STRATEGY}"
"${PYTHON_BIN}" -m fl_space.cli tune class-probability "${CLASS_PROBABILITY}"
"${PYTHON_BIN}" -m fl_space.cli tune preference-mode "${PREFERENCE_MODE}"
"${PYTHON_BIN}" -m fl_space.cli tune preferred-clients-per-class "${PREFERRED_CLIENTS_PER_CLASS}"
"${PYTHON_BIN}" -m fl_space.cli tune sample-cap-strategy "${SAMPLE_CAP_STRATEGY}"
"${PYTHON_BIN}" -m fl_space.cli tune classes-per-client "${CLASSES_PER_CLIENT}"
"${PYTHON_BIN}" -m fl_space.cli tune max-samples "${MAX_SAMPLES}"

"${PYTHON_BIN}" -m fl_space.cli run experiment \
  --gs ${GS_LIST} \
  --sats-list ${SAT_LIST} \
  --output "${OUTPUT_DIR}"

echo "SpaceFL cloud experiment finished: ${OUTPUT_DIR}"
