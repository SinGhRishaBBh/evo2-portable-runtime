#!/usr/bin/env python3

import os
import sys

# ------------------------------------------------------------------
# PRIORITIZE PORTABLE EVO2 RUNTIME
# ------------------------------------------------------------------

sys.path.insert(0, os.path.expanduser("~/evo2-portable-runtime"))

import csv
import gc
import time
import logging
import argparse
import traceback
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------------
# INDEXED FASTA READER
# ------------------------------------------------------------------

class IndexedFasta:

    def __init__(self, fasta_path: str):

        self.path = fasta_path
        self.index: Dict[str, dict] = {}

        fai_path = fasta_path + ".fai"

        if not os.path.exists(fai_path):
            raise FileNotFoundError(
                f"Missing FASTA index: {fai_path}"
            )

        with open(fai_path) as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) < 5:
                    continue

                name, length, offset, linebases, linewidth = parts

                self.index[name] = {
                    "length": int(length),
                    "offset": int(offset),
                    "linebases": int(linebases),
                    "linewidth": int(linewidth),
                }

        self._fh = open(fasta_path, "r")

    def fetch(self, chrom: str, start: int, end: int):

        if chrom not in self.index:

            if f"chr{chrom}" in self.index:
                chrom = f"chr{chrom}"

            elif chrom.startswith("chr") and chrom[3:] in self.index:
                chrom = chrom[3:]

            else:
                return None

        idx = self.index[chrom]

        start = max(0, start)
        end = min(end, idx["length"])

        if start >= end:
            return None

        n = end - start

        lb = idx["linebases"]
        lw = idx["linewidth"]
        off = idx["offset"]

        byte0 = off + (start // lb) * lw + (start % lb)

        max_newlines = n // lb + 2

        self._fh.seek(byte0)

        raw = self._fh.read(
            n + max_newlines * (lw - lb)
        )

        return (
            raw.replace("\n", "")
            .replace("\r", "")
            [:n]
            .upper()
        )

    def close(self):
        self._fh.close()

# ------------------------------------------------------------------
# CONTEXT EXTRACTION
# ------------------------------------------------------------------

def get_context(
    chrom,
    pos,
    ref,
    alt,
    fasta,
    half_window=200,
    logger=None,
):

    p0 = pos - 1

    start_0 = max(0, p0 - half_window)

    end_0 = p0 + len(ref) + half_window

    seq = fasta.fetch(chrom, start_0, end_0)

    if seq is None:
        return None

    rel = p0 - start_0

    if rel < 0 or rel >= len(seq):
        return None

    genome_ref = seq[rel : rel + len(ref)]

    if genome_ref != ref:

        if logger:
            logger.warning(
                f"REF allele mismatch at "
                f"{chrom}:{pos}. "
                f"Genome={genome_ref}, "
                f"Input={ref}. Skipping."
            )

        return None

    mut = (
        seq[:rel]
        + alt
        + seq[rel + len(ref):]
    )

    return seq, mut

# ------------------------------------------------------------------
# DATASET
# ------------------------------------------------------------------

class VariantDataset(Dataset):

    def __init__(
        self,
        df,
        fasta,
        half_window,
        logger=None,
    ):

        self.df = df
        self.fasta = fasta
        self.half_window = half_window
        self.logger = logger

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        chrom = str(row["CHROM"])
        pos = int(row["POS"])
        ref = str(row["REF"])
        alt = str(row["ALT"])

        vid = f"{chrom}_{pos}_{ref}_{alt}"

        ctx = get_context(
            chrom,
            pos,
            ref,
            alt,
            self.fasta,
            self.half_window,
            self.logger,
        )

        if ctx:

            return {
                "id": vid,
                "ref_seq": ctx[0],
                "mut_seq": ctx[1],
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "failed": False,
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
                "failed": True,
            }

# ------------------------------------------------------------------
# COLLATE
# ------------------------------------------------------------------

def simple_collate(batch):

    return {
        key: [d[key] for d in batch]
        for key in batch[0].keys()
    }

# ------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------

def setup_logging(log_path: Path):

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fmt = (
        "%(asctime)s | "
        "%(levelname)-8s | "
        "%(message)s"
    )

    logger = logging.getLogger("evo2_prod")

    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_path)

    fh.setFormatter(logging.Formatter(fmt))

    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)

    ch.setLevel(logging.INFO)

    ch.setFormatter(logging.Formatter(fmt))

    logger.addHandler(ch)

    return logger

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Evo2 Portable Runtime"
    )

    parser.add_argument(
        "--model",
        default="evo2_7b",
    )

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--reference",
        required=True,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--half_window",
        type=int,
        default=200,
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # LOGGING
    # ------------------------------------------------------------------

    logger = setup_logging(
        Path("./logs/evo2_run.log")
    )

    logger.info("=" * 70)
    logger.info("EVO2 PORTABLE RUNTIME")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # CUDA CHECK
    # ------------------------------------------------------------------

    if (
        args.device == "cuda"
        and not torch.cuda.is_available()
    ):

        logger.warning(
            "CUDA unavailable. "
            "Falling back to CPU."
        )

        args.device = "cpu"

    logger.info(f"MODEL      : {args.model}")
    logger.info(f"INPUT      : {args.input}")
    logger.info(f"OUTPUT     : {args.output}")
    logger.info(f"REFERENCE  : {args.reference}")
    logger.info(f"BATCH SIZE : {args.batch_size}")
    logger.info(f"DEVICE     : {args.device}")

    # ------------------------------------------------------------------
    # LOAD CSV
    # ------------------------------------------------------------------

    try:

        df = pd.read_csv(args.input)

        logger.info(
            f"Loaded {len(df)} rows."
        )

        # ------------------------------------------------------------------
        # NORMALIZE SCHEMA
        # ------------------------------------------------------------------

        logger.info(
            "Normalizing input schema..."
        )

        df.columns = [
            str(c).strip()
            for c in df.columns
        ]

        col_map = {}

        for c in df.columns:

            cl = c.lower().strip()

            # CHROM
            if cl in (
                "chrom",
                "chromosome",
                "chr",
                "#chrom",
                "chrm",
            ):
                col_map[c] = "CHROM"

            # COSMIC
            elif cl == "genome_start":
                col_map[c] = "POS"

            elif cl == "genomic_wt_allele":
                col_map[c] = "REF"

            elif cl == "genomic_mut_allele":
                col_map[c] = "ALT"

            # GENERIC
            elif cl in ("pos", "position"):
                col_map[c] = "POS"

            elif cl in ("ref", "reference"):
                col_map[c] = "REF"

            elif cl in ("alt", "alternate"):
                col_map[c] = "ALT"

        df = df.rename(columns=col_map)

        required_cols = {
            "CHROM",
            "POS",
            "REF",
            "ALT",
        }

        missing = (
            required_cols
            - set(df.columns)
        )

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )

        # ------------------------------------------------------------------
        # STRICT VALIDATION
        # ------------------------------------------------------------------

        valid_bases = {
            "A",
            "C",
            "G",
            "T",
        }

        valid_chroms = {
            "1","2","3","4","5","6","7","8","9","10",
            "11","12","13","14","15","16","17","18",
            "19","20","21","22","X","Y","MT"
        }

        rejected = {
            "missing_allele": 0,
            "invalid_base": 0,
            "invalid_length": 0,
            "invalid_chrom": 0,
            "missing_pos": 0,
        }

        valid_rows = []

        for idx, row in df.iterrows():

            chrom = row["CHROM"]
            pos = row["POS"]
            ref = row["REF"]
            alt = row["ALT"]

            # CHROM
            if pd.isna(chrom):
                rejected["invalid_chrom"] += 1
                continue

            chrom = (
                str(chrom)
                .replace("chr", "")
                .strip()
                .upper()
            )

            if chrom not in valid_chroms:
                rejected["invalid_chrom"] += 1
                continue

            # POS
            if pd.isna(pos):
                rejected["missing_pos"] += 1
                continue

            # REF ALT
            if (
                pd.isna(ref)
                or pd.isna(alt)
            ):
                rejected["missing_allele"] += 1
                continue

            ref = (
                str(ref)
                .strip()
                .upper()
            )

            alt = (
                str(alt)
                .strip()
                .upper()
            )

            # STRICT SNV
            if (
                len(ref) != 1
                or len(alt) != 1
            ):
                rejected["invalid_length"] += 1
                continue

            # DNA BASES
            if (
                ref not in valid_bases
                or alt not in valid_bases
            ):
                rejected["invalid_base"] += 1
                continue

            row_dict = row.to_dict()

            row_dict["CHROM"] = chrom
            row_dict["POS"] = int(pos)
            row_dict["REF"] = ref
            row_dict["ALT"] = alt

            valid_rows.append(row_dict)

        df = pd.DataFrame(valid_rows)

        logger.info("=" * 60)
        logger.info("VALIDATION REPORT")
        logger.info("=" * 60)

        logger.info(
            f"VALID ROWS    : {len(df)}"
        )

        logger.info(
            f"REJECTED ROWS : "
            f"{sum(rejected.values())}"
        )

        for k, v in rejected.items():

            logger.info(
                f"{k:<20}: {v}"
            )

        logger.info("=" * 60)

    except Exception as e:

        logger.error(
            f"CSV loading failed: {e}"
        )

        logger.error(
            traceback.format_exc()
        )

        return

    # ------------------------------------------------------------------
    # LOAD MODEL
    # ------------------------------------------------------------------

    try:

        from evo2.models import Evo2

        logger.info(
            f"Loading model: {args.model}"
        )

        model = Evo2(args.model)

        if args.device == "cuda":

            model.model = (
                model.model.to("cuda")
            )

        model.model.eval()

        logger.info(
            "Model loaded successfully."
        )

    except Exception as e:

        logger.error(
            f"Model loading failed: {e}"
        )

        logger.error(
            traceback.format_exc()
        )

        return

    # ------------------------------------------------------------------
    # FASTA
    # ------------------------------------------------------------------

    fasta = IndexedFasta(
        args.reference
    )

    dataset = VariantDataset(
        df,
        fasta,
        args.half_window,
        logger=logger,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=simple_collate,
    )

    # ------------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------------

    output_columns = [
        "variant_id",
        "CHROM",
        "POS",
        "REF",
        "ALT",
        "ref_llh",
        "mut_llh",
        "delta_llh",
        "effect",
    ]

    extra_metadata_cols = [
        "GENE_SYMBOL",
        "MUTATION_AA",
        "MUTATION_DESCRIPTION",
        "MUTATION_SOMATIC_STATUS",
        "HGVSC",
        "HGVSP",
        "COSMIC_SAMPLE_ID",
    ]

    for c in extra_metadata_cols:

        if c in df.columns:
            output_columns.append(c)

    # ------------------------------------------------------------------
    # RUN
    # ------------------------------------------------------------------

    out_path = Path(args.output)

    t0 = time.time()

    with open(
        out_path,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(output_columns)

        for batch_idx, batch in enumerate(dataloader):

            failed = batch["failed"]

            valid_indices = [
                i for i, x in enumerate(failed)
                if not x
            ]

            if not valid_indices:
                continue

            all_seqs = []

            for i in valid_indices:

                all_seqs.extend([
                    batch["ref_seq"][i],
                    batch["mut_seq"][i],
                ])

            try:

                if hasattr(
                    model.model,
                    "reset_inference_state",
                ):
                    model.model.reset_inference_state()

                with torch.no_grad():

                    scores = (
                        model.score_sequences(
                            all_seqs,
                            batch_size=len(all_seqs),
                            reduce_method="sum",
                        )
                    )

                for i, idx in enumerate(valid_indices):

                    r_score = float(
                        scores[2 * i]
                    )

                    m_score = float(
                        scores[2 * i + 1]
                    )

                    delta = (
                        m_score - r_score
                    )

                    if delta < -2:
                        effect = "pathogenic"

                    elif delta < -0.5:
                        effect = (
                            "possibly_pathogenic"
                        )

                    else:
                        effect = "benign"

                    row_out = {
                        "variant_id":
                            batch["id"][idx],

                        "CHROM":
                            batch["chrom"][idx],

                        "POS":
                            batch["pos"][idx],

                        "REF":
                            batch["ref"][idx],

                        "ALT":
                            batch["alt"][idx],

                        "ref_llh":
                            f"{r_score:.4f}",

                        "mut_llh":
                            f"{m_score:.4f}",

                        "delta_llh":
                            f"{delta:.4f}",

                        "effect":
                            effect,
                    }

                    for c in extra_metadata_cols:

                        if c in df.columns:

                            try:

                                row_out[c] = str(
                                    dataset.df.iloc[
                                        batch_idx
                                        * args.batch_size
                                        + idx
                                    ][c]
                                )

                            except Exception:

                                row_out[c] = ""

                    writer.writerow([
                        row_out.get(col, "")
                        for col in output_columns
                    ])

                f.flush()

            except Exception as e:

                logger.error(
                    f"Batch {batch_idx} failed: {e}"
                )

                logger.error(
                    traceback.format_exc()
                )

            finally:

                if torch.cuda.is_available():

                    torch.cuda.empty_cache()

                gc.collect()

            # PROGRESS
            if (batch_idx + 1) % 10 == 0:

                elapsed = (
                    time.time() - t0
                )

                processed = (
                    (batch_idx + 1)
                    * args.batch_size
                )

                rate = (
                    processed / elapsed
                )

                logger.info(
                    f"Processed "
                    f"{processed} variants "
                    f"({rate:.2f} "
                    f"variants/sec)"
                )

    logger.info("=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)

    logger.info(
        f"Finished inference."
    )

    logger.info(
        f"Output: {args.output}"
    )

    logger.info("=" * 70)

# ------------------------------------------------------------------
# ENTRY
# ------------------------------------------------------------------

if __name__ == "__main__":
    main()
