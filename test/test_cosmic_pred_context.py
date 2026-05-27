import logging

from cosmic_pred import IndexedFasta, get_context


def write_indexed_fasta(tmp_path, variant_base):
    sequence = ("A" * 599) + variant_base + ("G" * 600)
    reference = tmp_path / "reference.fa"
    reference.write_bytes((">X\n" + sequence + "\n").encode("ascii"))
    (tmp_path / "reference.fa.fai").write_text(
        f"X\t{len(sequence)}\t3\t{len(sequence)}\t{len(sequence) + 1}\n"
    )
    return reference


def test_get_context_returns_512_allele_512_for_snv(tmp_path):
    fasta = IndexedFasta(str(write_indexed_fasta(tmp_path, "T")))
    try:
        ref_seq, mut_seq = get_context("X", 600, "T", "C", fasta)
    finally:
        fasta.close()

    assert len(ref_seq) == 1025
    assert len(mut_seq) == 1025
    assert ref_seq[512] == "T"
    assert mut_seq[512] == "C"


def test_get_context_logs_and_skips_ref_mismatch(tmp_path, caplog):
    fasta = IndexedFasta(str(write_indexed_fasta(tmp_path, "C")))
    try:
        with caplog.at_level(logging.WARNING):
            context = get_context("X", 600, "T", "C", fasta, logger=logging.getLogger("test"))
    finally:
        fasta.close()

    assert context is None
    assert "REF allele mismatch at X:600. Genome=C, Input=T. Skipping." in caplog.text
