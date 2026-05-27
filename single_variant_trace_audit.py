#!/usr/bin/env python3
"""Strict single-variant semantic trace audit for Evo2 likelihood scoring.

This utility is observational. It captures the input construction used by the
local validation script and the portable production pipeline, then optionally
runs the identical Evo2 scoring operation on those inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


CONFIG_MAP = {
    "evo2_7b": "evo2/configs/evo2-7b-1m.yml",
    "evo2_40b": "evo2/configs/evo2-40b-1m.yml",
    "evo2_20b": "evo2/configs/evo2-20b-1m.yml",
    "evo2_40b_base": "evo2/configs/evo2-40b-8k.yml",
    "evo2_7b_base": "evo2/configs/evo2-7b-8k.yml",
    "evo2_1b_base": "evo2/configs/evo2-1b-8k.yml",
    "evo2_7b_262k": "evo2/configs/evo2-7b-262k.yml",
    "evo2_7b_microviridae": "evo2/configs/evo2-7b-8k.yml",
}


class IndexedFasta:
    """Small .fai reader matching the portable production reader semantics."""

    def __init__(self, fasta_path: str) -> None:
        self.path = fasta_path
        self.index: Dict[str, Dict[str, int]] = {}
        fai_path = Path(fasta_path + ".fai")
        if not fai_path.exists():
            raise FileNotFoundError(f"FASTA index (.fai) not found at {fai_path}")
        with fai_path.open() as handle:
            for line in handle:
                name, length, offset, linebases, linewidth = line.split()[:5]
                self.index[name] = {
                    "length": int(length),
                    "offset": int(offset),
                    "linebases": int(linebases),
                    "linewidth": int(linewidth),
                }
        self._handle = open(fasta_path, "r")

    def fetch(self, chrom: str, start: int, end: int) -> Optional[str]:
        if chrom not in self.index:
            if f"chr{chrom}" in self.index:
                chrom = f"chr{chrom}"
            elif chrom.startswith("chr") and chrom[3:] in self.index:
                chrom = chrom[3:]
            else:
                return None
        record = self.index[chrom]
        start = max(0, start)
        end = min(end, record["length"])
        if start >= end:
            return None
        base_count = end - start
        linebases = record["linebases"]
        linewidth = record["linewidth"]
        byte_offset = (
            record["offset"]
            + (start // linebases) * linewidth
            + (start % linebases)
        )
        self._handle.seek(byte_offset)
        raw = self._handle.read(base_count + (base_count // linebases + 2) * (linewidth - linebases))
        return raw.replace("\n", "").replace("\r", "")[:base_count].upper()

    def close(self) -> None:
        self._handle.close()


def parse_variant(variant: str) -> Tuple[str, int, str, str]:
    chrom, pos, ref, alt = variant.rsplit("_", 3)
    return chrom, int(pos), ref.upper(), alt.upper()


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_context(
    fasta: IndexedFasta, chrom: str, pos: int, ref: str, alt: str, left: int, right: int
) -> Dict[str, Any]:
    """Mirror test/test_delta_scores.py: left + supplied allele + right."""
    index_0 = pos - 1
    left_seq = fasta.fetch(chrom, index_0 - left, index_0) or ""
    right_seq = fasta.fetch(chrom, index_0 + len(ref), index_0 + len(ref) + right) or ""
    observed_ref = fasta.fetch(chrom, index_0, index_0 + len(ref))
    return {
        "chromosome": chrom,
        "position": pos,
        "extraction_succeeded": bool(left_seq or right_seq),
        "left_flank_length": len(left_seq),
        "right_flank_length": len(right_seq),
        "observed_reference_allele": observed_ref,
        "allele_matches_reference": observed_ref == ref,
        "reference_sequence": left_seq + ref + right_seq,
        "mutant_sequence": left_seq + alt + right_seq,
        "total_context_length": len(left_seq) + len(ref) + len(right_seq),
        "source_semantics": "test/test_delta_scores.py: 512 left bases + allele + 512 right bases",
    }


def portable_context(
    fasta: IndexedFasta, chrom: str, pos: int, ref: str, alt: str, half_window: int
) -> Dict[str, Any]:
    """Mirror cosmic_pred.py:get_context, including its exclusive end coordinate."""
    index_0 = pos - 1
    start_0 = max(0, index_0 - half_window)
    end_0 = index_0 + half_window
    sequence = fasta.fetch(chrom, start_0, end_0) or ""
    relative_index = index_0 - start_0
    observed_ref = sequence[relative_index : relative_index + len(ref)]
    matches_reference = observed_ref == ref
    mutant = sequence[:relative_index] + alt + sequence[relative_index + len(ref) :]
    return {
        "chromosome": chrom,
        "position": pos,
        "extraction_succeeded": bool(sequence) and matches_reference,
        "left_flank_length": relative_index,
        "right_flank_length": len(sequence) - relative_index - len(ref),
        "observed_reference_allele": observed_ref,
        "allele_matches_reference": matches_reference,
        "reference_sequence": sequence if matches_reference else None,
        "mutant_sequence": mutant if matches_reference else None,
        "candidate_reference_sequence": sequence if not matches_reference else None,
        "total_context_length": len(sequence),
        "source_semantics": "cosmic_pred.py:get_context: [pos-half_window, pos+half_window) interval",
    }


def token_trace(sequence: str, prepend_bos: bool) -> Dict[str, Any]:
    tokens = list(sequence.encode("ascii"))
    bos_id = 0
    eos_id = 0
    input_ids = ([bos_id] if prepend_bos else []) + tokens
    target_ids = input_ids[1:]
    return {
        "tokenizer": "CharLevelTokenizer",
        "vocab_size": 512,
        "bos_token_id": bos_id,
        "eos_token_id": eos_id,
        "bos_inserted": prepend_bos,
        "eos_inserted": False,
        "full_token_ids": input_ids,
        "token_count": len(input_ids),
        "first_20_token_ids": input_ids[:20],
        "last_20_token_ids": input_ids[-20:],
        "shifted_target_ids": target_ids,
        "shifted_target_length": len(target_ids),
        "scored_token_count": len(sequence) - 1 + int(prepend_bos),
    }


def model_loading_trace(
    model_name: str, checkpoint: Optional[str], loaded_model: Optional[Any] = None
) -> Dict[str, Any]:
    root = Path(__file__).resolve().parent
    config_path = root / CONFIG_MAP[model_name]
    checkpoint_path = Path(checkpoint).resolve() if checkpoint else None
    return {
        "model_name": model_name,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else "HuggingFace resolution at load time",
        "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path else None,
        "tokenizer": "CharLevelTokenizer",
        "tokenizer_vocab_size": 512,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "parameter_count": (
            sum(parameter.numel() for parameter in loaded_model.parameters())
            if loaded_model is not None
            else None
        ),
    }


def forward_trace(
    model: Any, sequence: str, prepend_bos: bool, reduction_method: str, device: str
) -> Dict[str, Any]:
    import numpy as np
    import torch

    tokens = list(sequence.encode("ascii"))
    input_ids = ([0] if prepend_bos else []) + tokens
    tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    if hasattr(model, "reset_inference_state"):
        model.reset_inference_state()
    with torch.inference_mode():
        output = model(tensor)
        logits = output[0] if isinstance(output, tuple) else output
        shifted_logits = logits[:, :-1, :]
        targets = tensor[:, 1:]
        all_logprobs = torch.log_softmax(shifted_logits, dim=-1)
        selected = torch.gather(all_logprobs, 2, targets.unsqueeze(-1)).squeeze(-1)
    score_len = len(sequence) - 1 + int(prepend_bos)
    raw = selected[0, :score_len].float().cpu().numpy()
    sum_score = float(np.sum(raw))
    mean_score = float(np.mean(raw))
    return {
        "logits_shape": list(logits.shape),
        "attention_mask_shape": None,
        "attention_mask_note": "No attention mask is passed by evo2.scoring._score_sequences.",
        "sequence_length_entering_model": tensor.shape[1],
        "shifted_target_length": targets.shape[1],
        "scored_token_count": score_len,
        "raw_token_logprobs": raw.tolist(),
        "first_20_per_token_logprobs": raw[:20].tolist(),
        "last_20_per_token_logprobs": raw[-20:].tolist(),
        "cumulative_sum_progression": np.cumsum(raw).tolist(),
        "normalization_divisor": score_len,
        "sum_reduction_output": sum_score,
        "mean_reduction_output": mean_score,
        "reduction_method": reduction_method,
        "final_llh": sum_score if reduction_method == "sum" else mean_score,
    }


def first_difference(left: Any, right: Any, path: str) -> Optional[Dict[str, Any]]:
    if type(left) is not type(right):
        return {"field": path, "local": left, "portable": right}
    if isinstance(left, dict):
        for key in left:
            if key in right:
                difference = first_difference(left[key], right[key], f"{path}.{key}")
                if difference:
                    return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return {"field": f"{path}.length", "local": len(left), "portable": len(right)}
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            difference = first_difference(left_value, right_value, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if left != right:
        return {"field": path, "local": left, "portable": right}
    return None


def compare_runtime_traces(local: Dict[str, Any], portable: Dict[str, Any]) -> Dict[str, Any]:
    local_tokens = local.get("tokenization") or {}
    portable_tokens = portable.get("tokenization") or {}
    local_forward = local.get("forward_pass") or {}
    portable_forward = portable.get("forward_pass") or {}
    local_context_value = {
        key: value for key, value in local.get("context_extraction", {}).items()
        if key != "source_semantics"
    }
    portable_context_value = {
        key: value for key, value in portable.get("context_extraction", {}).items()
        if key != "source_semantics"
    }
    stages = [
        ("model_loading", local.get("model_loading"), portable.get("model_loading")),
        ("context_extraction", local_context_value, portable_context_value),
        ("tokenization.reference", local_tokens.get("reference"), portable_tokens.get("reference")),
        ("tokenization.mutant", local_tokens.get("mutant"), portable_tokens.get("mutant")),
        ("forward_pass.reference", local_forward.get("reference"), portable_forward.get("reference")),
        ("per_token_logprobs.reference", local_forward.get("reference", {}).get("raw_token_logprobs"), portable_forward.get("reference", {}).get("raw_token_logprobs")),
        ("reduction", local.get("reduction"), portable.get("reduction")),
    ]
    for stage, local_value, portable_value in stages:
        if local_value is None or portable_value is None:
            continue
        difference = first_difference(local_value, portable_value, stage)
        if difference:
            return {"diverged": True, "first_divergence_stage": stage, "difference": difference}
    return {"diverged": False, "first_divergence_stage": None, "difference": None}


def capture_trace(args: argparse.Namespace) -> Dict[str, Any]:
    chrom, pos, ref, alt = parse_variant(args.variant)
    fasta = IndexedFasta(args.reference)
    try:
        contexts = {
            "local": local_context(fasta, chrom, pos, ref, alt, args.local_left, args.local_right),
            "portable": portable_context(fasta, chrom, pos, ref, alt, args.portable_half_window),
        }
    finally:
        fasta.close()

    loaded = None
    if args.run_model:
        from evo2.models import Evo2

        loaded_wrapper = Evo2(args.model, local_path=args.checkpoint)
        loaded_wrapper.model.eval()
        loaded = loaded_wrapper.model.to(args.device)

    traces: Dict[str, Dict[str, Any]] = {}
    for runtime, context in contexts.items():
        tokenization = None
        if context["extraction_succeeded"]:
            tokenization = {
                "reference": token_trace(context["reference_sequence"], args.prepend_bos),
                "mutant": token_trace(context["mutant_sequence"], args.prepend_bos),
            }
        trace: Dict[str, Any] = {
            "runtime": runtime,
            "variant": args.variant,
            "model_loading": model_loading_trace(args.model, args.checkpoint, loaded),
            "context_extraction": context,
            "tokenization": tokenization,
        }
        if loaded is not None and context["extraction_succeeded"]:
            reference = forward_trace(loaded, context["reference_sequence"], args.prepend_bos, args.reduction, args.device)
            mutant = forward_trace(loaded, context["mutant_sequence"], args.prepend_bos, args.reduction, args.device)
            trace["forward_pass"] = {"reference": reference, "mutant": mutant}
            trace["reduction"] = {
                "method": args.reduction,
                "ref_llh": reference["final_llh"],
                "mut_llh": mutant["final_llh"],
                "delta_llh": mutant["final_llh"] - reference["final_llh"],
            }
        traces[runtime] = trace
    return {
        "local": traces["local"],
        "portable": traces["portable"],
        "comparison": compare_runtime_traces(traces["local"], traces["portable"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", help="Indexed reference FASTA; requires adjacent .fai file.")
    parser.add_argument("--variant", default="X_76430509_T_C", help="Variant identifier CHROM_POS_REF_ALT.")
    parser.add_argument("--model", default="evo2_7b", choices=sorted(CONFIG_MAP))
    parser.add_argument("--checkpoint", help="Explicit identical checkpoint file to load and hash.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-left", type=int, default=512)
    parser.add_argument("--local-right", type=int, default=512)
    parser.add_argument("--portable-half-window", type=int, default=200)
    parser.add_argument("--prepend-bos", action="store_true")
    parser.add_argument("--reduction", choices=("sum", "mean"), default="sum")
    parser.add_argument("--run-model", action="store_true", help="Load Evo2 and capture logits/logprob/reduction stages.")
    parser.add_argument("--compare", nargs=2, metavar=("LOCAL_JSON", "PORTABLE_JSON"), help="Compare traces captured on separate systems.")
    parser.add_argument("--output", help="Write complete JSON trace to this file.")
    parser.add_argument("--print-full", action="store_true", help="Print full JSON even when --output is supplied.")
    args = parser.parse_args()

    if args.compare:
        local_data = json.loads(Path(args.compare[0]).read_text())
        portable_data = json.loads(Path(args.compare[1]).read_text())
        local = local_data.get("local", local_data)
        portable = portable_data.get("portable", portable_data)
        report = compare_runtime_traces(local, portable)
    else:
        if not args.reference:
            parser.error("--reference is required unless --compare is supplied")
        report = capture_trace(args)

    output = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n")
    if args.output and not args.print_full:
        summary = report.get("comparison", report)
        print(json.dumps(summary, indent=2))
        print(f"Complete trace written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
