#!/bin/bash
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --job-name=evo2bench
#SBATCH --output=evo2bench.out
#SBATCH --error=evo2bench.err
#SBATCH --time=24:00:00

echo "======================================"
echo "Evo2 Long Benchmark Started"
echo "Start Time: $(date)"
echo "======================================"

START_TIME=$(date +%s)

source ~/.bashrc
conda activate evo2-portable

cd ~/evo2-portable-runtime

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/home/rishabh.r/.conda/envs/evo2-portable/bin/python benchmark_runtime.py --batch-sizes 1 2 --windows 2048 4096 --num-variants 1000 --output-dir benchmark_large
--batch-sizes 1 2 
--windows 2048 4096 
--num-variants 1000 
--output-dir benchmark_large

END_TIME=$(date +%s)

ELAPSED=$((END_TIME - START_TIME))

echo "======================================"
echo "Benchmark Completed"
echo "End Time: $(date)"
echo "Total Runtime: ${ELAPSED} seconds"
echo "Total Runtime (hours): $(echo "scale=2; $ELAPSED/3600" | bc)"
echo "======================================"

