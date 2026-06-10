# Evo2 7B — Portable Runtime & HPC Benchmarking

A runtime engineering and benchmarking project to get **Evo2 7B** running on NVIDIA V100 GPUs — hardware the model was never designed for. The original codebase hard-requires BF16, FlashAttention, and FP8/TransformerEngine, none of which exist on V100 (sm_70). This repo patches all three and validates the result end-to-end.

> Not an official ARC Institute distribution.

---

## The Problem

Out of the box, loading Evo2 7B on a V100 throws three separate errors before the model even initialises:

- `Feature '.bf16' requires sm_80+` — V100 has no BF16 hardware support
- `No module named 'flash_attn_2_cuda'` — FlashAttention only builds on Ampere+
- `This model requires Transformer Engine FP8` — FP8 is H100-only

All three need to be patched before a single forward pass is possible.

---

## What Was Changed

| Component | Original | Patched |
|---|---|---|
| Precision | BF16 | FP16 |
| Attention | FlashAttention | PyTorch SDPA |
| FP8 / TransformerEngine | Required | Removed |

No weights were modified. The changes are applied to the Vortex runtime files before model load.

---

## What's in This Repo

- **BF16 → FP16 patch** — sed one-liner across `model.py`, `engine.py`, `utils.py` in the Vortex package
- **SDPA attention fallback** — drops FlashAttention, routes through `torch.nn.functional.scaled_dot_product_attention`
- **`benchmark_runtime.py`** — measures throughput, latency, and VRAM usage across batch sizes and context windows
- **`test_dna_generation.py`** — fixed-seed DNA generation test to verify deterministic output across different nodes
- **`run_evo2.sh` / `run_dna_test.sh`** — SLURM job scripts with correct GPU affinity, memory limits, and offline model loading
- **`evo2-runtime-final.sif`** — prebuilt Singularity container; self-contained, no host-side installs needed
- **10-phase numerical fix** — tracked down and resolved causal masking gaps, KV cache bleed between sequences, rotary embedding batch bugs, cross-OS tokenisation drift, and padding tokens leaking into ΔLL scores
- **Variant scoring runs** — 1,990 BRCA1 variants scored in ~1h 10min; 10,000 ClinVar variants in ~5h 20min
- **COSMIC × ClinVar pipeline** — exact-match overlap across 4.4M ClinVar and 5.5M COSMIC records; 561,107 matched, filtered to 5,330 tissue-specific variants

---

## Benchmark Numbers

Tested against the original Ampere build to quantify the cost of the fallback stack:

| Metric | Original (BF16 + FlashAttention) | This Repo (FP16 + SDPA) |
|---|---|---|
| Throughput | ~9–11 variants/sec | ~0.69–0.77 variants/sec |
| Latency | ~100 ms | ~1.40 sec |
| VRAM | ~12.56 GB | ~12.56 GB (unchanged) |
| Slowdown | — | ~13–15× |

The slowdown is real and expected — SDPA on V100 doesn't fuse ops the way FlashAttention does on Ampere. VRAM is unchanged because FP16 and BF16 have the same memory footprint at this model size. No OOM events across any run.

---

## Portable Container

Container on Hugging Face: **[rajsiri/evo2-7b-portable-runtime](https://huggingface.co/rajsiri/evo2-7b-portable-runtime)**
Artifact: `evo2-runtime-final.sif`

```bash
# Basic launch
singularity shell --nv evo2-runtime-final.sif

# Quick sanity check
python -c "import evo2; import vortex; print('Runtime OK')"

# Bind-mount local repo
singularity shell --nv \
    -B /path/to/evo2-portable-runtime:/workspace/evo2 \
    evo2-runtime-final.sif
```

Inside the container after bind-mount:

```bash
cd /workspace/evo2
export PYTHONPATH=$PWD:$PWD/vortex:$PYTHONPATH
```

---

## Running on HPC

```bash
sbatch run_evo2.sh        # full benchmark suite
sbatch run_dna_test.sh    # deterministic generation check
```

`run_evo2.sh` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and runs:

```bash
python benchmark_runtime.py \
    --batch-sizes 1 2 \
    --windows 2048 4096 \
    --num-variants 1000 \
    --output-dir benchmark_large
```

For long scoring jobs, run in the background:

```bash
nohup python score_variants.py > logs/run.log 2>&1 &
echo $! > logs/pid.txt
tail -f logs/run.log
```

---

## Errors You'll Hit and How to Fix Them

| Error | Fix |
|---|---|
| `Feature '.bf16' requires sm_80` | Run the BF16→FP16 sed patch before loading the model |
| `No module named 'flash_attn_2_cuda'` | `global_config.use_flash_attn = False` |
| `This model requires Transformer Engine FP8` | `global_config.use_fp8_input_projections = False` |
| `CUDA out of memory` | Drop `WINDOW_SIZE` to 512; set `batch_size=1`; call `torch.cuda.empty_cache()` |
| `LocalEntryNotFoundError` | Pre-download weights on the login node; set `HF_HOME` + `TRANSFORMERS_OFFLINE=1` in the job |
| `libcuda.so not found` inside Singularity | `export TRITON_LIBCUDA_PATH=/.singularity.d/libs` |
| Scores change between runs | Call `reset_inference_state()` before each sequence; fix random seed |
| Scores differ between machines | Make sure `WINDOW_SIZE` is the same on both — defaults vary |

---

## Repo Layout

```
evo2-portable-runtime/
├── benchmark_runtime.py      # Latency, throughput, VRAM profiling
├── test_dna_generation.py    # Deterministic fixed-seed generation test
├── run_evo2.sh               # SLURM: benchmarking job
├── run_dna_test.sh           # SLURM: sanity check job
├── Dockerfile
├── evo/
│   ├── models.py             # Model loading and checkpoint handling
│   └── scoring.py            # Forward pass and ΔLL scoring logic
└── vortex/                   # Patched attention and precision fallbacks
```

---

## Links

| | |
|---|---|
| This repo | https://github.com/SinGhRishaBBh/evo2-portable-runtime |
| Portable container | https://huggingface.co/rajsiri/evo2-7b-portable-runtime |
| Benchmark outputs | https://drive.google.com/drive/folders/1aV5g0mA8Ekvt-7SXlJ-fqKoAdocO43Fs |
| Scripts & code | https://drive.google.com/drive/folders/1i5XSa1Anmch156QVfxLi7DrX_30fX8IP |
| Evo2 upstream | https://github.com/arcinstitute/evo2 |

---

## Attribution

Evo2 model architecture and pretrained weights are the work of the ARC Institute (Nguyen et al., 2024 — *Genome modelling and design across all domains of life with Evo 2*). This repo covers only the runtime engineering: precision patch, attention fallback, benchmarking scripts, SLURM jobs, and the Singularity container.
