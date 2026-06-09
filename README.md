# Evo2 7B — Portable Runtime & HPC Benchmarking Infrastructure

This repository contains the engineering, validation, and benchmarking assets required to run **Evo2 7B** reliably on single-GPU HPC systems beyond the model's original Ampere-native target. The primary goal is a stable, production-grade portable runtime validated on **NVIDIA Tesla V100-SXM2-32GB** hardware using a **CUDA 12.4 + PyTorch 2.6.0** stack, with all FP8 and multi-GPU interconnect dependencies removed.

> **This is not an official ARC Institute distribution.** See [Attribution](#attribution--project-separation) for details.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Cross-GPU Portability: What Changed](#cross-gpu-portability-what-changed)
3. [Validated Performance Benchmarks](#validated-performance-benchmarks-evo2-7b-on-v100)
4. [Deterministic DNA Mutation Validation](#deterministic-dna-mutation-validation)
5. [Implemented Workflows](#implemented-workflows)
6. [Validated Runtime Environment](#validated-runtime-environment)
7. [Job Execution & HPC Deployment](#job-execution--hpc-deployment)
8. [Infrastructure Roadmap](#infrastructure-roadmap)
9. [Repository Structure](#repository-structure)
10. [Attribution & Project Separation](#attribution--project-separation)

---

## Architecture Overview

```
                        ┌──────────────────────────────┐
                        │        HPC Batch Job          │
                        │      (SLURM Scheduler)        │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │     Portable Runtime Host     │
                        │  PyTorch 2.6.0 + CUDA 12.4   │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │    NVIDIA Tesla V100 GPU      │
                        │  FP32/FP16 — No FP8/TE dep.  │
                        └───────────┬──────────────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                             ▼
┌─────────────────────────────┐         ┌──────────────────────────────┐
│  Genomic Sequence Scoring   │         │  Deterministic DNA Gen       │
│  · Long-context performance │         │  · Fixed-seed PRNG pipelines │
│  · Sliding-window metrics   │         │  · Cross-node drift detection│
└─────────────────────────────┘         └──────────────────────────────┘
```

### Design Goals

| Goal | Implementation |
|---|---|
| **Environment isolation** | Removes FlashAttention, FP8, and TransformerEngine; targets legacy/enterprise V100 nodes |
| **Determinism** | Fixed-seed generation pipeline validates output consistency across heterogeneous cluster nodes |
| **SLURM integration** | Ready-to-submit job scripts with correct device affinity and memory management configuration |

---

## Cross-GPU Portability: What Changed

The original Evo2 7B model targets Ampere-class GPUs (A100/H100) and depends on BF16 precision and FlashAttention kernels unavailable on V100. Two source-level changes make the model portable:

| Component | Original | Edited (Portable) |
|---|---|---|
| **Precision** | BF16 | FP16 |
| **Attention backend** | FlashAttention (Ampere-native) | PyTorch SDPA fallback |
| **FP8 / TransformerEngine** | Required | Removed |

These changes enable execution on any CUDA 12.x–capable GPU without hardware-specific kernel dependencies, at the cost of throughput (see benchmarks below).

---

## Validated Performance Benchmarks (Evo2 7B on V100)

A full comparative benchmark was conducted between the **Original Evo2 7B** (BF16 + FlashAttention) and the **Edited Cross-GPU** model (FP16 + SDPA) on V100 hardware.

### Throughput & Latency

| Metric | Original (BF16 + FlashAttention) | Edited (FP16 + SDPA) |
|---|---|---|
| Throughput | ~9–11 variants/sec (~10–11k tokens/sec) | ~0.69–0.77 variants/sec (~700–800 tokens/sec) |
| Average latency | ~100 ms (very low variance) | ~1.40 sec (stable, no outlier spikes) |
| **Performance delta** | — | **13–15× throughput reduction** |

The slowdown is attributable to SDPA's un-fused memory operations on V100 architectures. Latency remains operationally stable with no spikes or OOM events.

### VRAM & Resource Utilisation

- **VRAM footprint:** Identical across both models — ~12.56 GB allocated, ~13.04 GB reserved. The FP16 conversion does **not** reduce VRAM usage relative to BF16.
- **Memory fragmentation:** ~480 MB allocator overhead (normal PyTorch behaviour). No memory leaks or GPU starvation observed.
- **Pipeline cost distribution:** Inference dominates total runtime (>99%). FASTA fetch, tokenisation, and CSV I/O are operationally negligible.

---

## Deterministic DNA Mutation Validation

Validation tests confirm zero functional divergence between the original and edited models under a fixed RNG seed.

- **Reference sequence:** Both models produced an identical 1,000 bp reference (GC content: **49.30%**).
- **Mutagenesis:** Using `random.seed(42)`, both applied the exact same 5 point substitutions.
- **Mutation profile:** All 5 substitutions are transversions (purine ↔ pyrimidine), confirming no stochastic drift between deployments.

| Position | Mutation | Type |
|---|---|---|
| 52 | G → C | Transversion |
| 94 | A → T | Transversion |
| 218 | G → T | Transversion |
| 655 | A → T | Transversion |
| 995 | A → C | Transversion |

---

## Implemented Workflows

### 1. HPC Benchmarking & Profiling (`benchmark_runtime.py`)

The primary profiling pipeline tracks Evo2 7B execution across long context lengths:

- **Throughput metrics:** Tokens per second across varying batch sizes
- **VRAM analytics:** Active and cached GPU memory monitoring (OOM prevention)
- **Scaling curves:** Latency as a function of context window length
- **SLURM integration:** Wrapped via `run_evo2.sh` for queue submission

### 2. Deterministic DNA Generation Testing (`test_dna_generation.py`)

A reproducible correctness harness for verifying execution on isolated cluster nodes:

- Validates PRNG state consistency across machines
- Generates timestamped runtime logs to detect server-to-server compute drift
- Outputs standalone metrics for post-run analysis

---

## Validated Runtime Environment

| Component | Specification |
|---|---|
| Model | `evo2_7b` |
| GPU | NVIDIA Tesla V100-SXM2-32GB |
| CUDA runtime | 12.4 |
| PyTorch | `2.6.0+cu124` |
| Python | 3.11 |
| Platform | Enterprise Linux HPC Cluster (SLURM) |
| Inference mode | Single-GPU FP16 (no FP8 / TransformerEngine required) |

---

## Job Execution & HPC Deployment

### Profiling Benchmark (SLURM)

Configure partition constraints in `run_evo2.sh`, then submit:

```bash
sbatch run_evo2.sh
```

The script sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to optimise VRAM allocation on V100, and runs:

```bash
python benchmark_runtime.py \
    --batch-sizes 1 2 \
    --windows 2048 4096 \
    --num-variants 1000 \
    --output-dir benchmark_large
```

### Deterministic Sanity Test (SLURM)

```bash
sbatch run_dna_test.sh
```

Executes `test_dna_generation.py` with fixed-seed random generation and mutation calculations, tracking latency to millisecond precision.

---

## Infrastructure Roadmap

### Singularity / Apptainer Containerisation (Planned)

Singularity-based deployment is on the roadmap but was not completed during the current evaluation cycle. All benchmarks reflect bare-metal runtime results only.

Planned build and execution workflows:

```bash
# Convert Dockerfile to Apptainer/Singularity container
singularity build evo2_7b_portable.sif Dockerfile

# Execute within a containerised SLURM reservation
singularity exec --nv \
    -B /scratch/user/.cache/huggingface:/root/.cache/huggingface \
    evo2_7b_portable.sif \
    python benchmark_runtime.py --batch-sizes 1 --windows 2048 --num-variants 100
```

### Performance Optimisation (Planned)

| Enhancement | Description |
|---|---|
| `torch.compile()` | Compile SDPA attention blocks via PyTorch's Inductor backend to recover throughput |
| INT8 quantisation | Reduce memory overhead during long-context sweeps |
| Triton attention backend | Adaptive fallback attention kernels to bridge the V100 performance gap |

---

## Repository Structure

```
evo2-portable-runtime/
├── .gitignore                   # Excludes benchmark outputs, pycache, logs
├── pyproject.toml               # Package specs and dependency configuration
├── Dockerfile                   # Container deployment blueprint
├── benchmark_runtime.py         # Profiling script: latency, throughput, VRAM
├── test_dna_generation.py       # Deterministic DNA generation & mutation validator
├── run_evo2.sh                  # SLURM job: full-scale long-context benchmarking
├── run_dna_test.sh              # SLURM job: deterministic sanity check
├── evo/
│   ├── models.py                # Checkpoint loading and model orchestration
│   └── scoring.py               # Inference pipelines and scoring metrics
└── vortex/                      # Compilation hooks and adaptive attention fallbacks
```

---

## Attribution & Project Separation

This repository is an **independent runtime engineering project** and is not affiliated with or endorsed by the ARC Institute.

**Evo2 Core Architecture:** Model architecture, pretrained weights, and foundational parameters are the intellectual property of the original authors at the ARC Institute. Reference: *Genome modelling and design across all domains of life with Evo 2*.

**Portable Runtime Contributions:** All benchmarking scripts (`benchmark_runtime.py`, `test_dna_generation.py`), batch execution workflows (`run_evo2.sh`, `run_dna_test.sh`), and environment-specific dependency modifications are independent engineering contributions developed to enable stable, long-context evaluation on enterprise V100 hardware.
