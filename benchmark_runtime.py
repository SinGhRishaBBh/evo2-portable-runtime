#!/usr/bin/env python3
"""
Evo2 HPC Inference Benchmark & Profiling Framework
==================================================
A production-grade, highly rigorous framework designed to benchmark
throughput, latency, CPU/GPU utilization, and memory behavior of the
Evo2 genomic inference runtime.

Features:
- Nested high-precision timing (PipelineTimer) with optional CUDA synchronization.
- Real-time system monitoring (CPU, RAM, GPU, VRAM) via background threading.
- Multi-dimensional sweeps (batch sizes, context windows, variant scaling).
- Publication-quality plotting engine generating 10 diagnostic figures (300 DPI).
- HPC-compatible CSV exports for downstream statistical profiling.
- Portable CPU/GPU fallback and synthetic/mock execution capability.
"""

import os
import sys
import csv
import time
import logging
import argparse
import traceback
import gc
import threading
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any

import numpy as np
import pandas as pd
import torch

# Ensure local imports work correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Optional third-party imports with robust fallbacks
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for HPC/headless environments
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

# Setup professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("evo2_benchmark")


# ---------------------------------------------------------------------------
# INDEXED FASTA READER
# ---------------------------------------------------------------------------
class IndexedFasta:
    def __init__(self, fasta_path: str):
        self.path = fasta_path
        self.index: Dict[str, dict] = {}
        fai_path = fasta_path + ".fai"
        if not os.path.exists(fai_path):
            raise FileNotFoundError(f"FASTA index (.fai) not found at {fai_path}")
            
        with open(fai_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5: continue
                name, length, offset, linebases, linewidth = parts
                self.index[name] = dict(
                    length=int(length), offset=int(offset),
                    linebases=int(linebases), linewidth=int(linewidth),
                )
        self._fh = open(fasta_path, "r")

    def fetch(self, chrom: str, start: int, end: int) -> Optional[str]:
        if chrom not in self.index:
            if f"chr{chrom}" in self.index: chrom = f"chr{chrom}"
            elif chrom.startswith("chr") and chrom[3:] in self.index: chrom = chrom[3:]
            else: return None
            
        idx = self.index[chrom]
        start, end = max(0, start), min(end, idx["length"])
        if start >= end: return None
        n = end - start
        lb, lw, off = idx["linebases"], idx["linewidth"], idx["offset"]
        byte0 = off + (start // lb) * lw + (start % lb)
        max_newlines = n // lb + 2
        self._fh.seek(byte0)
        raw = self._fh.read(n + max_newlines * (lw - lb))
        return raw.replace("\n", "").replace("\r", "")[:n].upper()

    def close(self):
        self._fh.close()


# ---------------------------------------------------------------------------
# MOCK / SYNTHETIC RUNTIME ENGINE
# ---------------------------------------------------------------------------
class MockModel:
    """Mock Evo2 Model representing the computational load of inference."""
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        # Simulate a base weight parameter size
        class MockInnerModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.zeros(1000, 1000))
            def reset_inference_state(self):
                pass
        self.model = MockInnerModel().to(device)

    def score_sequences(self, seqs: List[str], batch_size: int = 1, reduce_method: str = "sum") -> List[float]:
        # Simulate sequence scoring overhead
        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.synchronize()
        
        # Computational work simulation relative to sequence length
        total_len = sum(len(s) for s in seqs)
        # Allocate temp tensor on target device to simulate VRAM usage
        temp_size = min(max(1024, total_len * 10), 1024 * 1024 * 128)  # up to ~500MB VRAM allocation
        temp_tensor = torch.zeros(temp_size, dtype=torch.float32, device=self.device)
        # Compute a dummy gemm operation to simulate compute load
        n_dim = min(1024, int(math.sqrt(temp_size)))
        if n_dim > 64:
            mat = torch.randn(n_dim, n_dim, device=self.device)
            res = torch.matmul(mat, mat)
            del mat, res
        
        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.synchronize()
            
        del temp_tensor
        return [-0.5 * len(s) for s in seqs]


class MockTokenizer:
    def __init__(self):
        self.vocab_size = 512
    def tokenize(self, seq: str) -> List[int]:
        return list(seq.encode("ascii", errors="ignore"))


# ---------------------------------------------------------------------------
# PIPELINE TIMER (High Precision & CUDA synchronized)
# ---------------------------------------------------------------------------
class PipelineTimer:
    def __init__(self, device: torch.device):
        self.device = device
        self.records: Dict[str, List[float]] = {}
        self.active_stages: List[str] = []
        self._start_times: Dict[str, float] = {}

    def start(self, stage_name: str):
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize()
        self._start_times[stage_name] = time.perf_counter()
        self.active_stages.append(stage_name)

    def stop(self, stage_name: str):
        if stage_name not in self._start_times:
            return
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._start_times[stage_name]
        self.records.setdefault(stage_name, []).append(elapsed)
        self.active_stages.remove(stage_name)
        del self._start_times[stage_name]

    def __call__(self, stage_name: str):
        class TimerContext:
            def __init__(self, timer_obj: PipelineTimer, name: str):
                self.timer_obj = timer_obj
                self.name = name
            def __enter__(self):
                self.timer_obj.start(self.name)
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.timer_obj.stop(self.name)
        return TimerContext(self, stage_name)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for stage, times in self.records.items():
            if not times: continue
            summary[stage] = {
                "total": sum(times),
                "avg": np.mean(times),
                "std": np.std(times),
                "min": min(times),
                "max": max(times),
                "count": len(times)
            }
        return summary


# ---------------------------------------------------------------------------
# REAL-TIME SYSTEM MONITOR
# ---------------------------------------------------------------------------
class SystemMonitor(threading.Thread):
    def __init__(self, interval_sec: float = 0.05):
        super().__init__()
        self.daemon = True
        self.interval_sec = interval_sec
        self.stop_event = threading.Event()
        self.records: List[Dict[str, Any]] = []

    def get_system_metrics(self) -> Dict[str, float]:
        metrics = {"timestamp": time.time(), "cpu_util": 0.0, "ram_used_mb": 0.0}
        
        # CPU & RAM Metrics
        if HAS_PSUTIL:
            metrics["cpu_util"] = psutil.cpu_percent()
            metrics["ram_used_mb"] = psutil.virtual_memory().used / (1024 * 1024)
            
        # GPU Metrics
        if torch.cuda.is_available():
            metrics["gpu_vram_alloc_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            metrics["gpu_vram_max_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
            
            # Query nvidia-smi for precise utilization if possible
            try:
                import subprocess
                cmd = "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits"
                out = subprocess.check_output(cmd, shell=True, text=True).strip()
                gpu_util, gpu_mem_util, gpu_vram_used = map(float, out.split(","))
                metrics["gpu_util_pct"] = gpu_util
                metrics["gpu_mem_util_pct"] = gpu_mem_util
                metrics["gpu_vram_smi_mb"] = gpu_vram_used
            except Exception:
                metrics["gpu_util_pct"] = 0.0
                metrics["gpu_mem_util_pct"] = 0.0
                metrics["gpu_vram_smi_mb"] = metrics["gpu_vram_alloc_mb"]
        else:
            metrics["gpu_vram_alloc_mb"] = 0.0
            metrics["gpu_vram_max_mb"] = 0.0
            metrics["gpu_util_pct"] = 0.0
            metrics["gpu_mem_util_pct"] = 0.0
            metrics["gpu_vram_smi_mb"] = 0.0
            
        return metrics

    def run(self):
        while not self.stop_event.is_set():
            try:
                metrics = self.get_system_metrics()
                self.records.append(metrics)
            except Exception as e:
                pass
            time.sleep(self.interval_sec)

    def stop(self):
        self.stop_event.set()
        self.join()


# ---------------------------------------------------------------------------
# SYNTHETIC DATA GENERATOR
# ---------------------------------------------------------------------------
def generate_synthetic_inputs(num_variants: int, window: int) -> Tuple[pd.DataFrame, IndexedFasta]:
    """Generates synthetic Variant CSV and FASTA payload mapping hg38 structures."""
    import tempfile
    temp_dir = Path(tempfile.gettempdir()) / "evo2_benchmark_payload"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    fasta_path = temp_dir / "synthetic.fa"
    fai_path = temp_dir / "synthetic.fa.fai"
    csv_path = temp_dir / "variants.csv"
    
    # Write a highly repetitive base sequence
    base_chars = "ACGT"
    seq_len = num_variants * 100 + window * 2 + 5000
    np.random.seed(42)
    random_bases = "".join(np.random.choice(list(base_chars), size=seq_len))
    
    fasta_content = f">chr1\n{random_bases}\n"
    fasta_path.write_text(fasta_content)
    
    # Create corresponding index .fai
    fai_content = f"chr1\t{seq_len}\t6\t{seq_len}\t{seq_len + 1}\n"
    fai_path.write_text(fai_content)
    
    # Generate CSV rows
    rows = []
    for i in range(num_variants):
        pos = window + i * 50 + 10
        ref_base = random_bases[pos - 1]
        alt_base = "T" if ref_base != "T" else "A"
        rows.append({
            "CHROM": "chr1",
            "POS": pos,
            "REF": ref_base,
            "ALT": alt_base
        })
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    
    return df, IndexedFasta(str(fasta_path))


# ---------------------------------------------------------------------------
# CORE BENCHMARKING ORCHESTRATOR
# ---------------------------------------------------------------------------
import math

class Evo2BenchmarkSuite:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.device)
        
        # Resolve safety fallback
        if self.device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable. Falling back to CPU.")
            self.device = torch.device("cpu")
            self.args.device = "cpu"

        # Output Directories
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.output_dir / "plots_benchmark"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Pipeline timer
        self.timer = PipelineTimer(self.device)

        # Metrics aggregates
        self.aggregate_metrics: List[Dict[str, Any]] = []

    def load_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Loads either the authentic model or mock model."""
        if self.args.mock:
            logger.info("Initializing mock Evo2 runtime...")
            return MockModel(self.args.model, device=self.args.device), MockTokenizer()
        
        # authentic model loading
        logger.info(f"Loading real Evo2 model '{self.args.model}'...")
        from evo2.models import Evo2
        model = Evo2(self.args.model)
        if self.device.type == "cuda":
            model.model = model.model.to(self.args.device)
        model.model.eval()
        return model, model.tokenizer

    def run_benchmark_sweep(self):
        logger.info("Starting Benchmark Profiling Suite...")
        
        # Load Model & Tokenizer
        with self.timer("model_loading"):
            model, tokenizer = self.load_model_and_tokenizer()

        # Load/Generate input data
        df = None
        fasta = None
        if self.args.input and self.args.reference:
            # Custom real data
            logger.info(f"Loading user dataset: {self.args.input}")
            df = pd.read_csv(self.args.input)
            fasta = IndexedFasta(self.args.reference)
        else:
            # Synthetic standard data
            logger.info(f"Generating synthetic genomics payload (Count: {self.args.num_variants})...")
            df, fasta = generate_synthetic_inputs(self.args.num_variants, max(self.args.windows))

        # Start CPU/GPU continuous monitoring thread
        sys_monitor = SystemMonitor(interval_sec=0.05)
        sys_monitor.start()

        # Sweeping configurations
        try:
            for window in self.args.windows:
                for batch_size in self.args.batch_sizes:
                    logger.info(f"Sweeping Batch Size: {batch_size} | Half-Window: {window}bp")
                    
                    # Warmup run to isolate compile/caching latency
                    self._execute_inference_pipeline(
                        df.head(min(len(df), batch_size)), 
                        fasta, 
                        model, 
                        batch_size, 
                        window, 
                        warmup=True
                    )

                    # Official run
                    metrics = self._execute_inference_pipeline(
                        df, 
                        fasta, 
                        model, 
                        batch_size, 
                        window, 
                        warmup=False
                    )
                    self.aggregate_metrics.append(metrics)
                    
        finally:
            sys_monitor.stop()
            fasta.close()

        # Write Metrics & Plots
        self._write_reports(sys_monitor.records)
        self._generate_diagnostic_plots(sys_monitor.records)
        logger.info(f"Benchmarking completed successfully. Outputs saved to {self.output_dir.absolute()}")

    def _execute_inference_pipeline(
        self, 
        df: pd.DataFrame, 
        fasta: IndexedFasta, 
        model: Any, 
        batch_size: int, 
        half_window: int, 
        warmup: bool = False
    ) -> Dict[str, Any]:
        """Runs the whole preprocessing -> tokenization -> inference -> writing pipeline."""
        t_start = time.perf_counter()
        
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        prefix = "warmup_" if warmup else ""
        
        # 1. FASTA Extraction
        ref_seqs = []
        mut_seqs = []
        v_ids = []
        chroms, positions, refs, alts = [], [], [], []
        
        fasta_extract_tag = f"{prefix}fasta_extraction_w{half_window}_b{batch_size}"
        with self.timer(fasta_extract_tag):
            for idx, row in df.iterrows():
                chrom = str(row["CHROM"])
                pos = int(row["POS"])
                ref = str(row["REF"])
                alt = str(row["ALT"])
                
                # Slicing
                p0 = pos - 1
                start_0 = max(0, p0 - half_window)
                end_0 = p0 + len(ref) + half_window
                
                seq = fasta.fetch(chrom, start_0, end_0)
                if seq:
                    rel = p0 - start_0
                    mut = seq[:rel] + alt + seq[rel + len(ref):]
                    ref_seqs.append(seq)
                    mut_seqs.append(mut)
                    v_ids.append(f"{chrom}_{pos}_{ref}_{alt}")
                    chroms.append(chrom)
                    positions.append(pos)
                    refs.append(ref)
                    alts.append(alt)

        # 2. Tokenization & Inference batch loop
        num_batches = math.ceil(len(ref_seqs) / batch_size)
        out_csv_path = self.output_dir / f"benchmark_temp_results_w{half_window}_b{batch_size}.csv"
        
        inference_tag = f"{prefix}inference_w{half_window}_b{batch_size}"
        tokenization_tag = f"{prefix}tokenization_w{half_window}_b{batch_size}"
        csv_write_tag = f"{prefix}csv_writing_w{half_window}_b{batch_size}"
        
        total_tokens_processed = 0

        # Open temp CSV to record timings of CSV writing
        with open(out_csv_path, "w", newline="") as f_out:
            writer = csv.writer(f_out)
            writer.writerow(["variant_id", "ref_llh", "mut_llh"])

            for b in range(num_batches):
                b_start = b * batch_size
                b_end = min(b_start + batch_size, len(ref_seqs))
                
                # Fetch batch slices
                batch_ref = ref_seqs[b_start:b_end]
                batch_mut = mut_seqs[b_start:b_end]
                batch_vids = v_ids[b_start:b_end]
                
                # Flatten strings for combined scoring input
                all_seqs = []
                for r, m in zip(batch_ref, batch_mut):
                    all_seqs.extend([r, m])
                    total_tokens_processed += len(r) + len(m)
                
                # Measure Tokenization overhead
                with self.timer(tokenization_tag):
                    # Mock tokenization or standard model tokenizer trace
                    if self.args.mock:
                        for s in all_seqs:
                            _ = list(s.encode("ascii"))
                    else:
                        for s in all_seqs:
                            _ = model.tokenizer.tokenize(s)
                
                # Measure GPU inference forward pass
                with self.timer(inference_tag):
                    if hasattr(model.model, "reset_inference_state"):
                        model.model.reset_inference_state()
                        
                    scores = model.score_sequences(
                        all_seqs,
                        batch_size=len(all_seqs),
                        reduce_method="sum"
                    )

                # Measure CSV Writing block
                with self.timer(csv_write_tag):
                    for i in range(len(batch_vids)):
                        r_score = float(scores[2*i])
                        m_score = float(scores[2*i+1])
                        writer.writerow([batch_vids[i], r_score, m_score])
                    f_out.flush()

        t_end = time.perf_counter()
        elapsed_sec = t_end - t_start

        # Calculate metrics
        num_variants = len(ref_seqs)
        throughput_variants_sec = num_variants / elapsed_sec if elapsed_sec > 0 else 0
        throughput_batches_sec = num_batches / elapsed_sec if elapsed_sec > 0 else 0
        throughput_tokens_sec = total_tokens_processed / elapsed_sec if elapsed_sec > 0 else 0

        # Remove temp results to keep output directory clean
        try:
            os.remove(out_csv_path)
        except OSError:
            pass

        return {
            "batch_size": batch_size,
            "half_window": half_window,
            "num_variants": num_variants,
            "num_batches": num_batches,
            "total_tokens": total_tokens_processed,
            "total_runtime_sec": elapsed_sec,
            "throughput_variants_sec": throughput_variants_sec,
            "throughput_batches_sec": throughput_batches_sec,
            "throughput_tokens_sec": throughput_tokens_sec,
        }

    # ---------------------------------------------------------------------------
    # WRITE EXPORT METRICS
    # ---------------------------------------------------------------------------
    def _write_reports(self, system_records: List[Dict[str, Any]]):
        # 1. Write benchmark_summary.txt
        summary_path = self.output_dir / "benchmark_summary.txt"
        with open(summary_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("      EVO2 PORTABLE GENOMIC INFERENCE RUNTIME PERFORMANCE PROFILING REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Device: {self.args.device.upper()}\n")
            f.write(f"Mock Execution Mode: {self.args.mock}\n")
            f.write(f"Evo2 Model Version: {self.args.model}\n\n")

            f.write("STAGE RUNTIME BREAKDOWN & SYSTEM LATENCIES:\n")
            f.write("-" * 40 + "\n")
            timer_summary = self.timer.get_summary()
            for stage, stats in timer_summary.items():
                f.write(f"Stage: {stage}\n")
                f.write(f"  - Count:   {stats['count']}\n")
                f.write(f"  - Total:   {stats['total']:.4f} sec\n")
                f.write(f"  - Average: {stats['avg']:.4f} sec (± {stats['std']:.4f})\n")
                f.write(f"  - Min:     {stats['min']:.4f} sec | Max: {stats['max']:.4f} sec\n\n")

            f.write("\nAGGREGATE CONFIGURATION SWEEP THROUGHPUTS:\n")
            f.write("-" * 40 + "\n")
            for met in self.aggregate_metrics:
                f.write(f"Config: BatchSize={met['batch_size']}, HalfWindow={met['half_window']}bp\n")
                f.write(f"  - Variants processed: {met['num_variants']}\n")
                f.write(f"  - Runtime elapsed:    {met['total_runtime_sec']:.4f} sec\n")
                f.write(f"  - Throughput (Var/s): {met['throughput_variants_sec']:.2f}\n")
                f.write(f"  - Throughput (Tok/s): {met['throughput_tokens_sec']:.2f}\n\n")

        # 2. Write benchmark_metrics.csv
        metrics_df = pd.DataFrame(self.aggregate_metrics)
        metrics_df.to_csv(self.output_dir / "benchmark_metrics.csv", index=False)

        # 3. Write timing_breakdown.csv
        timing_records = []
        for stage, stats in timer_summary.items():
            timing_records.append({
                "stage": stage,
                "count": stats["count"],
                "total_time_sec": stats["total"],
                "avg_time_sec": stats["avg"],
                "std_time_sec": stats["std"],
                "min_time_sec": stats["min"],
                "max_time_sec": stats["max"],
            })
        pd.DataFrame(timing_records).to_csv(self.output_dir / "timing_breakdown.csv", index=False)

        # 4. Write system tracking csv files
        sys_df = pd.DataFrame(system_records)
        if not sys_df.empty:
            gpu_cols = ["timestamp", "gpu_util_pct", "gpu_mem_util_pct", "gpu_vram_alloc_mb", "gpu_vram_max_mb"]
            gpu_df = sys_df[[c for c in gpu_cols if c in sys_df.columns]]
            gpu_df.to_csv(self.output_dir / "gpu_metrics.csv", index=False)

            mem_cols = ["timestamp", "ram_used_mb", "gpu_vram_alloc_mb"]
            mem_df = sys_df[[c for c in mem_cols if c in sys_df.columns]]
            mem_df.to_csv(self.output_dir / "memory_metrics.csv", index=False)

    # ---------------------------------------------------------------------------
    # GENERATE DIAGNOSTIC PLOTS
    # ---------------------------------------------------------------------------
    def _generate_diagnostic_plots(self, system_records: List[Dict[str, Any]]):
        if not HAS_PLOTTING:
            logger.warning("Matplotlib or Seaborn not found. Skipping plot generation.")
            return

        logger.info(f"Generating 10 scientific plots inside: {self.plots_dir.absolute()}...")
        sys_df = pd.DataFrame(system_records)
        metrics_df = pd.DataFrame(self.aggregate_metrics)
        
        # Apply publication quality formatting
        sns.set_theme(style="whitegrid", context="paper", palette="muted")
        plt.rcParams.update({
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.titlesize": 14
        })

        # 1. Throughput vs Batch Size
        plt.figure(figsize=(6, 4), dpi=300)
        sns.lineplot(data=metrics_df, x="batch_size", y="throughput_variants_sec", hue="half_window", marker="o")
        plt.title("Genomic Inference Throughput vs Batch Size")
        plt.xlabel("Batch Size")
        plt.ylabel("Throughput (Variants/sec)")
        plt.tight_layout()
        plt.savefig(self.plots_dir / "throughput_vs_batch_size.png")
        plt.close()

        # 2. GPU Utilization Over Time
        if not sys_df.empty and "gpu_util_pct" in sys_df.columns:
            plt.figure(figsize=(6, 4), dpi=300)
            plt.plot(sys_df["timestamp"] - sys_df["timestamp"].iloc[0], sys_df["gpu_util_pct"], color="#9b59b6")
            plt.title("GPU Compute Utilization Over Time")
            plt.xlabel("Time (sec)")
            plt.ylabel("GPU Utilization (%)")
            plt.tight_layout()
            plt.savefig(self.plots_dir / "gpu_utilization_over_time.png")
            plt.close()

        # 3. VRAM Usage Over Time
        if not sys_df.empty and "gpu_vram_alloc_mb" in sys_df.columns:
            plt.figure(figsize=(6, 4), dpi=300)
            plt.plot(sys_df["timestamp"] - sys_df["timestamp"].iloc[0], sys_df["gpu_vram_alloc_mb"], color="#e74c3c")
            plt.title("Active GPU VRAM Allocated Over Time")
            plt.xlabel("Time (sec)")
            plt.ylabel("VRAM Usage (MB)")
            plt.tight_layout()
            plt.savefig(self.plots_dir / "vram_usage_over_time.png")
            plt.close()

        # 4. FASTA Extraction Latency
        fasta_latencies = []
        for key, vals in self.timer.records.items():
            if "fasta_extraction" in key:
                fasta_latencies.extend(vals)
        if fasta_latencies:
            plt.figure(figsize=(6, 4), dpi=300)
            sns.histplot(fasta_latencies, kde=True, color="#3498db")
            plt.title("Genomic FASTA Fetch Latency Distribution")
            plt.xlabel("Latency per query batch (sec)")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(self.plots_dir / "fasta_extraction_latency.png")
            plt.close()

        # 5. Inference Latency Distribution
        infer_latencies = []
        for key, vals in self.timer.records.items():
            if "inference" in key:
                infer_latencies.extend(vals)
        if infer_latencies:
            plt.figure(figsize=(6, 4), dpi=300)
            sns.histplot(infer_latencies, kde=True, color="#2ecc71")
            plt.title("Evo2 Forward Model Inference Latency")
            plt.xlabel("Latency (sec)")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(self.plots_dir / "inference_latency_distribution.png")
            plt.close()

        # 6. CPU Utilization
        if not sys_df.empty:
            plt.figure(figsize=(6, 4), dpi=300)
            plt.plot(sys_df["timestamp"] - sys_df["timestamp"].iloc[0], sys_df["cpu_util"], color="#f1c40f")
            plt.title("Host CPU Thread Utilization Over Time")
            plt.xlabel("Time (sec)")
            plt.ylabel("CPU Utilization (%)")
            plt.tight_layout()
            plt.savefig(self.plots_dir / "cpu_utilization.png")
            plt.close()

        # 7. Runtime Stage Breakdown
        timer_summary = self.timer.get_summary()
        stages, values = [], []
        for stage, stats in timer_summary.items():
            if "warmup" not in stage and ("fasta" in stage or "tokenization" in stage or "inference" in stage or "csv" in stage):
                stages.append(stage.split("_")[0].capitalize())
                values.append(stats["total"])
        if values:
            plt.figure(figsize=(6, 4), dpi=300)
            plt.pie(values, labels=stages, autopct="%1.1f%%", colors=["#3498db", "#2ecc71", "#e74c3c", "#f1c40f"])
            plt.title("Evo2 End-to-End Runtime Phase Breakdown")
            plt.tight_layout()
            plt.savefig(self.plots_dir / "runtime_stage_breakdown.png")
            plt.close()

        # 8. Scaling Curves (Throughput vs Window Size)
        plt.figure(figsize=(6, 4), dpi=300)
        sns.lineplot(data=metrics_df, x="half_window", y="throughput_variants_sec", hue="batch_size", marker="s")
        plt.title("Scaling Curves: Throughput vs Sequence Context Length")
        plt.xlabel("Half Context-Window (bp)")
        plt.ylabel("Throughput (Variants/sec)")
        plt.tight_layout()
        plt.savefig(self.plots_dir / "scaling_curves.png")
        plt.close()

        # 9. Batch Efficiency Curves (Throughput / VRAM MB)
        if not metrics_df.empty:
            metrics_df["efficiency_ratio"] = metrics_df["throughput_variants_sec"] / (metrics_df["batch_size"] + 1e-5)
            plt.figure(figsize=(6, 4), dpi=300)
            sns.lineplot(data=metrics_df, x="batch_size", y="efficiency_ratio", hue="half_window", marker="d")
            plt.title("Scaling Batch Workload Efficiency")
            plt.xlabel("Batch Size")
            plt.ylabel("Efficiency (Variants per sec / Batch Item)")
            plt.tight_layout()
            plt.savefig(self.plots_dir / "batch_efficiency_curves.png")
            plt.close()

        # 10. End-to-End Pipeline Breakdown Timeline
        plt.figure(figsize=(7, 4), dpi=300)
        non_warmup_records = {k: sum(v) for k, v in self.timer.records.items() if "warmup" not in k}
        sorted_records = sorted(non_warmup_records.items(), key=lambda x: x[1], reverse=True)
        if sorted_records:
            stages_list, total_times = zip(*sorted_records)
            y_pos = np.arange(len(stages_list))
            plt.barh(y_pos, total_times, align="center", color="#34495e")
            plt.yticks(y_pos, [s.replace("_", " ") for s in stages_list])
            plt.gca().invert_yaxis()
            plt.title("Detailed Cumulative Pipeline Stage Overheads")
            plt.xlabel("Total Cumulative Latency (sec)")
            plt.tight_layout()
            plt.savefig(self.plots_dir / "end_to_end_pipeline_breakdown.png")
            plt.close()


# ---------------------------------------------------------------------------
# CLI ENTRYPOINT
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evo2 7B Inference Benchmark & Profiling Framework")
    parser.add_argument("--model", default="evo2_7b", help="Model name or path to load")
    parser.add_argument("--device", default="cuda", help="Inference device: 'cuda' or 'cpu'")
    
    # Sweep Configuration Controls
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 16, 32], help="List of batch sizes to sweep")
    parser.add_argument("--windows", nargs="+", type=int, default=[256, 512, 1024], help="List of context half-windows to sweep")
    
    # Real Inputs vs Synthetics
    parser.add_argument("--input", help="Custom Variant CSV input path")
    parser.add_argument("--reference", help="Custom genomic FASTA reference path")
    parser.add_argument("--num-variants", type=int, default=100, help="Number of variants to generate for synthetic testing")
    
    # Execution Modality
    parser.add_argument("--mock", action="store_true", help="Use a lightweight mock runtime simulating Evo2 layers without weights")
    parser.add_argument("--output-dir", default="benchmark_results", help="Directory where benchmarks metrics/plots are exported")
    
    args = parser.parse_args()
    
    suite = Evo2BenchmarkSuite(args)
    suite.run_benchmark_sweep()


if __name__ == "__main__":
    main()
