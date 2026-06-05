# Evo2 7B Portable Runtime & HPC Benchmarking Infrastructure

This repository contains custom engineering, validation, and benchmarking assets designed to establish a stable, production-grade, portable runtime environment specifically for **Evo2 7B** execution on single-GPU HPC systems. 

Unlike the core framework configurations, this project focuses strictly on validating execution on **NVIDIA Tesla V100-SXM2-32GB** GPUs using a clean **CUDA 12.4 + PyTorch 2.6.0** stack, removing all hardware-specific dependencies (such as FP8-based architectures or multi-GPU interconnect constraints).

---

## Technical Architecture & Scope

This project is tailored specifically for the **Evo2 7B** genomic foundation model. All tooling, benchmarking parameters, scripts, and runtime validation logic are designed for single-GPU execution paths. 

```
                                +---------------------------+
                                |      HPC Batch Job        |
                                |      (SLURM Scheduler)    |
                                +-------------+-------------+
                                              |
                                              v
                                +-------------+-------------+
                                |  Portable Runtime Host    |
                                | (PyTorch 2.6.0 + CUDA 12.4)|
                                +-------------+-------------+
                                              |
                                              v
                              +---------------+---------------+
                              |    NVIDIA Tesla V100 GPU      |
                              |  - FP32/FP16 Math Engines     |
                              |  - Strict Non-FP8 Execution   |
                              +---------------+---------------+
                                              |
                       +----------------------+----------------------+
                       |                                             |
                       v                                             v
        +--------------+--------------+               +--------------+--------------+
        |   Genomic Sequence Scoring  |               |  Deterministic DNA Gen      |
        | - Long-Context Performance  |               | - Standalone TXT Pipelines  |
        | - Sliding-Window Metrics    |               | - Fixed PRNG Seed Validation|
        +-----------------------------+               +-----------------------------+
```

### Core Architecture Goals
* **Environment Isolation:** Clean execution parameters for legacy and enterprise HPC systems lacking native FP8 support (e.g., V100/A100 hardware).
* **Determinism:** Standalone reproducible test pipelines to ensure that sequence generation matches baseline validation runs across heterogeneous cluster nodes.
* **SLURM Integration:** Out-of-the-box job configurations for batch queue execution, ensuring proper device affinity and memory management limits.

---

## Actual Implemented Workflows

This repository provides specific, operational workflows developed for runtime validation and execution.

### 1. HPC Benchmarking & Profiling
The primary profiling pipeline resides in `benchmark_runtime.py`. It tracks and records the execution characteristics of Evo2 7B over long context lengths:
* **Throughput Metrics:** Tracks tokens processed per second across varying batch sizes.
* **VRAM Allocation Analytics:** Monitors active and cached GPU memory thresholds to prevent Out-Of-Memory (OOM) faults.
* **Scaling Curves:** Measures latency scaling behavior across different context windows.
* **SLURM Integration:** Wraps execution under structured resource management (using `run_evo2.sh`).

### 2. Deterministic DNA Generation Testing
For verification of execution correctness on isolated cluster nodes, `test_dna_generation.py` provides a reproducible testing script that uses fixed seeds to output sequence generations and target mutations.
* Verifies PRNG state consistency.
* Generates clear runtime logs to identify server-to-server compute drift.
* Outputs standalone metrics for post-run analysis.

---

## Runtime Validation & Environment Specs

All components of this repository have been engineered and validated against the following specific execution baseline:

* **Model Variant:** `evo2_7b`
* **GPU Hardware:** NVIDIA Tesla V100-SXM2-32GB
* **CUDA Runtime:** 12.4
* **PyTorch Version:** `2.6.0+cu124`
* **Python Version:** 3.11
* **Platform:** Enterprise Linux HPC Cluster
* **Inference Mode:** Single-GPU FP16 Execution (No FP8/TransformerEngine dependencies required)

---

## Job Execution & HPC Deployment

### Running the Profiling Benchmark (SLURM)
To submit the benchmarking suite to a SLURM queue, configure your partition constraints inside [run_evo2.sh](file:///d:/evo2-portable-runtime/run_evo2.sh) and execute:

```bash
sbatch run_evo2.sh
```

The script configures `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to optimize VRAM allocations on the V100, and triggers:

```bash
python benchmark_runtime.py \
    --batch-sizes 1 2 \
    --windows 2048 4096 \
    --num-variants 1000 \
    --output-dir benchmark_large
```

### Running the DNA Generation Test (SLURM)
To run a rapid sanity check to verify system determinism and compute scaling on an allocated node:

```bash
sbatch run_dna_test.sh
```

This script executes [test_dna_generation.py](file:///d:/evo2-portable-runtime/test_dna_generation.py), which performs fixed-seed random generation and mutation calculations, tracking execution latency down to millisecond precision.

---

## Infrastructure Roadmap & Future Containerization

### Planned Singularity / Apptainer Containerization
Containerized Singularity-based deployment was planned as part of the future infrastructure optimization roadmap; however, benchmarking and validation for this environment were not completed during the current evaluation cycle. Current benchmarking results are limited to the actively validated bare-metal runtime environments. Singularity-based deployment and HPC container benchmarking remain outside the scope of the present assessment.

Future plans to build and validate the runtime environment within Singularity include:
* **Target Build Workflow (Planned):**
  ```bash
  # Future work: Convert the Dockerfile build layers to an Apptainer/Singularity container
  singularity build evo2_7b_portable.sif Dockerfile
  ```
* **Target HPC Execution (Planned):**
  ```bash
  # Future work: Execute within a containerized SLURM reservation
  singularity exec --nv \
      -B /scratch/user/.cache/huggingface:/root/.cache/huggingface \
      evo2_7b_portable.sif \
      python benchmark_runtime.py --batch-sizes 1 --windows 2048 --num-variants 100
  ```

---

## Repository Structure

```
├── .gitignore                   # Excludes large benchmark outputs, pycache, and logs
├── pyproject.toml               # Package specifications and dependency settings
├── benchmark_runtime.py         # Main profiling script tracking latency, throughput, and VRAM
├── test_dna_generation.py       # Deterministic DNA sequence generation validator
├── run_evo2.sh                  # SLURM script for full-scale long-context benchmarking
├── run_dna_test.sh              # SLURM script for deterministic sanity test execution
├── Dockerfile                   # Deployment container blueprint
├── evo/                         # Evo2 model configuration wrappers and hooks
│   ├── models.py                # Checkpoint loading and model orchestration
│   └── scoring.py               # Inference pipelines and scoring metrics
└── vortex/                      # Nested compilation hooks and adaptive fallbacks
```

---

## Attribution & Project Separation

This repository is an independent runtime engineering and benchmarking project. It is **not** an official distribution of the ARC Institute. 

* **Evo2 Core Architecture:** All intellectual property relating to the model architecture, pretrained parameters, and foundational weights belong to the original authors at the ARC Institute (see the paper *Genome modelling and design across all domains of life with Evo 2*).
* **Portable Runtime Modifications:** Custom benchmarking scripts (`benchmark_runtime.py`, `test_dna_generation.py`), batch execution workflows (`run_evo2.sh`, `run_dna_test.sh`), and environment-specific dependency alignments are independent contributions engineered to enable stable, long-context evaluation on enterprise V100 GPU nodes.
