#!/bin/bash
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --job-name=dnatest
#SBATCH --output=dnatest.out
#SBATCH --error=dnatest.err
#SBATCH --time=00:10:00

echo "========================================"
echo "DNA GENERATION TEST STARTED"
echo "START TIME: $(/bin/date)"
echo "========================================"

START_TIME=$(/bin/date +%s%3N)

source ~/.bashrc
conda activate evo2-portable

cd ~/evo2-portable-runtime

python3 test_dna_generation.py

END_TIME=$(/bin/date +%s%3N)

RUNTIME_MS=$((END_TIME - START_TIME))

RUNTIME_SEC=$(awk "BEGIN {printf \"%.3f\", ${RUNTIME_MS}/1000}")

echo ""
echo "========================================"
echo "DNA GENERATION TEST FINISHED"
echo "END TIME: $(/bin/date)"
echo "TOTAL RUNTIME: ${RUNTIME_SEC} seconds"
echo "========================================"
