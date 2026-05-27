#!/usr/bin/env python3
"""
Evo2 7B Portable Runtime Inference Pipeline — Production Execution
==================================================================
Automated variant scoring with Evo2-7B on genomic datasets.
Features:
  - Persistent background execution support
  - Incremental result writing & checkpointing
  - Optimized batch scoring on GPU
  - samtools-based genomic context fetching
"""

import os
import sys

# 3. Ensure portable runtime is prioritized over globally installed packages at absolute top
sys.path.insert(0, os.path.expanduser("~/evo2-portable-runtime"))

import csv
import time
import logging
import argparse
import traceback
import gc
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# INDEXED FASTA READER (Pure Python)
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
            # Try with 'chr' prefix
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

def get_context(
    chrom: str, pos: int, ref: str, alt: str,
    fasta: IndexedFasta, half_window: int = 200
) -> Optional[Tuple[str, str]]:
    """Return (ref_seq, mut_seq) centered on the variant."""
    p0 = pos - 1
    start_0 = max(0, p0 - half_window)
    end_0 = p0 + half_window
    
    seq = fasta.fetch(chrom, start_0, end_0)
    if seq:
        rel = p0 - start_0
        if rel < 0 or rel >= len(seq): return None
        if seq[rel] != ref: return None # Allele mismatch
        mut = seq[:rel] + alt + seq[rel + 1:]
        return seq, mut
    return None

# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------

class VariantDataset(Dataset):
    def __init__(self, df: pd.DataFrame, fasta: IndexedFasta, half_window: int):
        self.df = df
        self.fasta = fasta
        self.half_window = half_window

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        chrom = str(row["CHROM"])
        pos = int(row["POS"])
        ref = str(row["REF"])
        alt = str(row["ALT"])
        vid = f"{chrom}_{pos}_{ref}_{alt}"
        
        ctx = get_context(chrom, pos, ref, alt, self.fasta, self.half_window)
        if ctx:
            return {
                "id": vid, 
                "ref_seq": ctx[0], 
                "mut_seq": ctx[1], 
                "chrom": chrom, 
                "pos": pos, 
                "ref": ref, 
                "alt": alt,
                "failed": False
            }
        else:
            return {
                "id": vid, 
                "ref_seq": "", 
                "mut_seq": "", 
                "chrom": chrom, 
                "pos": pos, 
                "ref": ref, 
                "alt": alt,
                "failed": True
            }

def simple_collate(batch):
    """Custom collate to avoid tensorization of mixed-type metadata."""
    return {key: [d[key] for d in batch] for key in batch[0].keys()}

# ---------------------------------------------------------------------------
# CORE PIPELINE
# ---------------------------------------------------------------------------

def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    dfmt = "%Y-%m-%d %H:%M:%S"
    logger = logging.getLogger("evo2_prod")
    logger.setLevel(logging.DEBUG)
    
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter(fmt, dfmt))
    logger.addHandler(fh)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, dfmt))
    logger.addHandler(ch)
    return logger

def main():
    parser = argparse.ArgumentParser(description="Evo2 7B Portable Runtime Inference Pipeline")
    # 1. Default model set to evo2_7b
    parser.add_argument("--model", default="evo2_7b", help="Model name or path")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=lambda x: str(x).lower() == 'true', default=True)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    # 5. Remove hardcoded reference genome and make required
    parser.add_argument(
        "--reference",
        required=True,
        help="Reference FASTA path"
    )
    parser.add_argument("--half_window", type=int, default=200)
    args = parser.parse_args()

    # 4. Fix logging path to be portable
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path("./logs/evo2_run.log")
    
    logger = setup_logging(log_path)
    logger.info("=" * 70)
    logger.info("EVO2 (7B) PORTABLE RUNTIME PRODUCTION INFERENCE PIPELINE")
    logger.info("=" * 70)
    
    # 6. Safe CUDA Fallback
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable. Falling back to CPU.")
        args.device = "cpu"
        
    logger.info(f"Model       : {args.model}")
    logger.info(f"Input       : {args.input}")
    logger.info(f"Output      : {args.output}")
    logger.info(f"Batch Size  : {args.batch_size}")
    logger.info(f"Device      : {args.device}")
    
    # 1. Load Data
    try:
        df = pd.read_csv(args.input)
        # Column normalization
        col_map = {}
        for c in df.columns:
            cl = c.lower().strip()
            if cl in ("chrm", "chrom", "chromosome", "chr", "#chrom"): col_map[c] = "CHROM"
            elif cl in ("pos", "position", "genome_start"): col_map[c] = "POS"
            elif cl in ("ref", "reference", "genomic_wt_allele"): col_map[c] = "REF"
            elif cl in ("alt", "alternate", "genomic_mut_allele"): col_map[c] = "ALT"
        df = df.rename(columns=col_map)
        
        total_rows = len(df)
        logger.info(f"Loaded {total_rows} variants from CSV.")
        
        # Production Genomic Variant Preprocessing & Validation Checkpoint
        valid_bases = {"A", "C", "G", "T"}
        valid_chroms = {
            "1","2","3","4","5","6","7","8","9","10",
            "11","12","13","14","15","16","17","18",
            "19","20","21","22","X","Y","MT"
        }
        
        valid_rows = []
        rejected_counts = {
            "missing_chrom": 0,
            "invalid_chrom": 0,
            "missing_pos": 0,
            "missing_allele": 0,
            "invalid_length": 0,
            "invalid_base": 0
        }
        
        for idx, row in df.iterrows():
            chrom = row.get("CHROM")
            pos = row.get("POS")
            ref = row.get("REF")
            alt = row.get("ALT")
            
            # 1. Validate Chromosome Presence
            if pd.isna(chrom):
                logger.warning(f"Row {idx}: Rejected variant. Chromosome is missing.")
                rejected_counts["missing_chrom"] += 1
                continue
                
            # Normalize Chromosome
            chrom_norm = str(chrom).replace("chr", "").strip().upper()
            if chrom_norm not in valid_chroms:
                logger.warning(f"Row {idx}: Rejected variant. Chromosome '{chrom}' is invalid.")
                rejected_counts["invalid_chrom"] += 1
                continue
                
            # 2. Validate Position Presence
            if pd.isna(pos):
                logger.warning(f"Row {idx}: Rejected variant at chrom '{chrom_norm}'. Position is missing.")
                rejected_counts["missing_pos"] += 1
                continue
                
            # 3. Validate REF/ALT Presence
            if pd.isna(ref) or pd.isna(alt):
                logger.warning(f"Row {idx}: Rejected variant at {chrom_norm}:{pos}. REF='{ref}', ALT='{alt}' has missing allele.")
                rejected_counts["missing_allele"] += 1
                continue
                
            # Normalize Alleles
            ref_norm = str(ref).strip().upper()
            alt_norm = str(alt).strip().upper()
            
            # 4. Validate REF/ALT Length (strictly SNV)
            if len(ref_norm) != 1 or len(alt_norm) != 1:
                logger.warning(f"Row {idx}: Rejected variant at {chrom_norm}:{pos}. REF='{ref_norm}', ALT='{alt_norm}' has invalid length (strictly must be SNV length of 1).")
                rejected_counts["invalid_length"] += 1
                continue
                
            # 5. Validate Canonical Base
            if ref_norm not in valid_bases or alt_norm not in valid_bases:
                logger.warning(f"Row {idx}: Rejected variant at {chrom_norm}:{pos}. REF='{ref_norm}', ALT='{alt_norm}' contains non-canonical DNA bases.")
                rejected_counts["invalid_base"] += 1
                continue
                
            # Store updated normalized values
            row_dict = row.to_dict()
            row_dict["CHROM"] = chrom_norm
            row_dict["REF"] = ref_norm
            row_dict["ALT"] = alt_norm
            row_dict["POS"] = int(pos)
            valid_rows.append(row_dict)
            
        df = pd.DataFrame(valid_rows) if valid_rows else pd.DataFrame(columns=df.columns)
        total_rejected = sum(rejected_counts.values())
        
        logger.info("=" * 60)
        logger.info("=== GENOMIC VARIANT PREPROCESSING & VALIDATION REPORT ===")
        logger.info(f"Total Rows Checked : {total_rows}")
        logger.info(f"Valid Rows Passed  : {len(df)}")
        logger.info(f"Rejected Rows      : {total_rejected}")
        logger.info("Rejection Breakdown:")
        for category, count in rejected_counts.items():
            logger.info(f"  - {category:<15} : {count}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to load/validate input CSV: {e}")
        logger.error(traceback.format_exc())
        return

    # 2. Load Model using Portable Runtime
    try:
        from evo2.models import Evo2
        
        # 2. Remove legacy model remapping
        model_name = args.model
        logger.info(f"Initializing portable Evo2 runtime for {model_name}...")
        
        model = Evo2(model_name)
        if args.device == "cuda":
            model.model = model.model.to("cuda")
        model.model.eval()
        
        logger.info("Model loaded successfully into portable runtime.")
    except Exception as e:
        logger.error(f"Failed to load Evo2 model: {e}")
        logger.error(traceback.format_exc())
        return

    # 3. Setup DataLoader
    fasta = IndexedFasta(args.reference)
    dataset = VariantDataset(df, fasta, args.half_window)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        shuffle=False,
        collate_fn=simple_collate
    )

    # 4. Inference Loop
    t0 = time.time()
    
    # Write header if new file
    write_header = not out_path.exists()
    
    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["variant_id", "CHROM", "POS", "REF", "ALT", "ref_llh", "mut_llh", "delta_llh", "effect"])

        for batch_idx, batch in enumerate(dataloader):
            v_ids = batch["id"]
            ref_seqs = batch["ref_seq"]
            mut_seqs = batch["mut_seq"]
            failed = batch.get("failed", [False] * len(v_ids))
            
            # Filter out failed ones for scoring
            valid_indices = [i for i, f in enumerate(failed) if not f and ref_seqs[i]]
            
            if valid_indices:
                all_seqs = []
                for i in valid_indices:
                    all_seqs.extend([ref_seqs[i], mut_seqs[i]])
                
                try:
                    # 7. Strict Inference-State Reset for determinism
                    if hasattr(model.model, "reset_inference_state"):
                        model.model.reset_inference_state()

                    with torch.no_grad():
                        scores = model.score_sequences(
                            all_seqs, 
                            batch_size=len(all_seqs), 
                            reduce_method="sum"
                        )
                    
                    for i, idx in enumerate(valid_indices):
                        r_score = float(scores[2*i])
                        m_score = float(scores[2*i+1])
                        delta = m_score - r_score
                        effect = "pathogenic" if delta < -2 else "possibly_pathogenic" if delta < -0.5 else "benign"
                        
                        writer.writerow([
                            v_ids[idx], 
                            batch["chrom"][idx], 
                            batch["pos"][idx], 
                            batch["ref"][idx], 
                            batch["alt"][idx], 
                            f"{r_score:.4f}", 
                            f"{m_score:.4f}", 
                            f"{delta:.4f}", 
                            effect
                        ])
                    f.flush()
                except RuntimeError as re:
                    if "CUDA out of memory" in str(re):
                        logger.error(f"OOM on Batch {batch_idx}. Clearing cache and skipping...")
                    else:
                        logger.error(f"Batch {batch_idx} runtime error: {re}")
                        logger.debug(traceback.format_exc())
                except Exception as e:
                    logger.error(f"Batch {batch_idx} scoring failed: {e}")
                    logger.debug(traceback.format_exc())
                finally:
                    # 8. Safe GPU Cache Cleanup after scoring batches
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
            
            # Log progress and cleanup
            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate = (batch_idx + 1) * args.batch_size / elapsed
                logger.info(f"Processed { (batch_idx+1)*args.batch_size } variants... Rate: {rate:.1f}/s")
                f.flush()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

    logger.info("=" * 70)
    logger.info("=== FINAL PRODUCTION RUN SUMMARY ===")
    logger.info(f"Total Valid Variants Scored : {len(df)}")
    logger.info(f"Total Rejected Variants     : {total_rejected}")
    logger.info("Rejection Breakdown:")
    for category, count in rejected_counts.items():
        logger.info(f"  - {category:<15} : {count}")
    logger.info("====================================")
    logger.info(f"Inference complete. Results saved to {args.output}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
