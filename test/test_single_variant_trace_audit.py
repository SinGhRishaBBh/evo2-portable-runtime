from argparse import Namespace

from single_variant_trace_audit import capture_trace


def test_portable_context_matches_local_window_after_correction(tmp_path):
    sequence = ("A" * 599) + "T" + ("G" * 600)
    reference = tmp_path / "reference.fa"
    reference.write_bytes((">X\n" + sequence + "\n").encode("ascii"))
    (tmp_path / "reference.fa.fai").write_text(f"X\t{len(sequence)}\t3\t{len(sequence)}\t{len(sequence) + 1}\n")

    result = capture_trace(
        Namespace(
            reference=str(reference),
            variant="X_600_T_C",
            model="evo2_7b",
            checkpoint=None,
            device="cpu",
            local_left=512,
            local_right=512,
            portable_half_window=512,
            prepend_bos=False,
            reduction="sum",
            run_model=False,
        )
    )

    assert result["local"]["context_extraction"]["total_context_length"] == 1025
    assert result["portable"]["context_extraction"]["total_context_length"] == 1025
    assert result["comparison"]["diverged"] is False


def test_portable_trace_stops_on_reference_allele_mismatch(tmp_path):
    sequence = ("A" * 599) + "C" + ("G" * 600)
    reference = tmp_path / "reference.fa"
    reference.write_bytes((">X\n" + sequence + "\n").encode("ascii"))
    (tmp_path / "reference.fa.fai").write_text(f"X\t{len(sequence)}\t3\t{len(sequence)}\t{len(sequence) + 1}\n")

    result = capture_trace(
        Namespace(
            reference=str(reference),
            variant="X_600_T_C",
            model="evo2_7b",
            checkpoint=None,
            device="cpu",
            local_left=512,
            local_right=512,
            portable_half_window=512,
            prepend_bos=False,
            reduction="sum",
            run_model=False,
        )
    )

    assert result["local"]["context_extraction"]["extraction_succeeded"]
    assert not result["portable"]["context_extraction"]["extraction_succeeded"]
    assert result["portable"]["tokenization"] is None
