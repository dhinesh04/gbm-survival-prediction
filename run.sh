#!/bin/bash
# run.sh

export PYTHONPATH="$(pwd)"
set -e

# ── Create timestamped run folder ─────────────────────────────────────────────
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="src/outputs/runs/${TIMESTAMP}"

mkdir -p "${RUN_DIR}/logs"
mkdir -p "${RUN_DIR}/data"
mkdir -p "${RUN_DIR}/plots"

LOG_FILE="${RUN_DIR}/logs/experiment.log"

exec > >(tee -a "$LOG_FILE") 2>&1

EXPERIMENT_START=$SECONDS

USE_MUTATIONS=true
MUTATION_FLAG="--use_mutations"

echo "============================================================"
echo "  GBM Experiment Run"
echo "  Started   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Run folder: ${RUN_DIR}/"
echo "  Log file  : ${LOG_FILE}"
echo "  Mutations : ${USE_MUTATIONS}"
echo "============================================================"

# ── Step 1: Data Processing (runs once) ──────────────────────────────────────
DATA_DIR="${RUN_DIR}/data"
PLOTS_DIR="${RUN_DIR}/plots"

echo ""
echo "--- Step 1/3: Data Processing ---"
python3 src/data/data_processing.py \
    --output_dir "$DATA_DIR" \
    $MUTATION_FLAG

# ── Step 2: Full pipeline + GCN (runs once) ───────────────────────────────────
echo ""
echo "--- Step 2/3: Main Pipeline + GCN ---"
python3 main.py \
    --data_dir  "$DATA_DIR" \
    --plots_dir "$PLOTS_DIR" \
    $MUTATION_FLAG

# ── Step 3: Imaging survival analysis (51-patient BraTS/TCGA overlap) ────────
echo ""
echo "--- Step 3/3: Imaging Survival Analysis ---"
python3 imaging_survival_analysis.py "$PLOTS_DIR"

# ── Summary ───────────────────────────────────────────────────────────────────
TOTAL_ELAPSED=$(( SECONDS - EXPERIMENT_START ))

echo ""
echo "============================================================"
echo "  All done!"
echo "  Finished  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Total time: $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo ""
echo "  Run folder: ${RUN_DIR}/"
echo "    data  ->  ${DATA_DIR}/"
echo "    plots ->  ${PLOTS_DIR}/"
echo "    log   ->  ${LOG_FILE}"
echo "============================================================"