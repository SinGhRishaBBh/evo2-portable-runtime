import sys
sys.path.insert(0, ".")
sys.path.insert(0, "vortex")

import torch
from pathlib import Path
from pyfaidx import Fasta

from evo2.models import Evo2
from evo2.scoring import prepare_batch
from vortex.model.tokenizer import CharLevelTokenizer

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "evo2_7b"

REFERENCE = "test/hg38.fa"

CHROM = "7"
POS = 1498347
REF = "G"
ALT = "A"

HALF_WINDOW = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 80)
print("LOADING EVO2 MODEL")
print("=" * 80)

model = Evo2(MODEL_NAME)

print("Model Loaded:", MODEL_NAME)

tokenizer = CharLevelTokenizer(vocab_size=512)

print("Tokenizer vocab size:", tokenizer.vocab_size)

# ============================================================
# LOAD FASTA
# ============================================================

print("=" * 80)
print("LOADING FASTA")
print("=" * 80)

fasta = Fasta(REFERENCE)

# ============================================================
# EXTRACT CONTEXT
# ============================================================

print("=" * 80)
print("EXTRACTING CONTEXT")
print("=" * 80)

chrom_key = CHROM

if chrom_key not in fasta:
    chrom_key = "chr" + CHROM

p0 = POS - 1

left_start = max(0, p0 - HALF_WINDOW)
right_end = p0 + len(REF) + HALF_WINDOW

context = fasta[chrom_key][left_start:right_end].seq.upper()

print("Context Length:", len(context))

ref_base = context[HALF_WINDOW]

print("Genome REF Base:", ref_base)
print("Input REF Base :", REF)

if ref_base != REF:
    print("REF MISMATCH — STOPPING")
    sys.exit(1)

mut_context = (
    context[:HALF_WINDOW]
    + ALT
    + context[HALF_WINDOW + len(REF):]
)

print("Mut Context Length:", len(mut_context))

print("\nFIRST 50 REF BASES:")
print(context[:50])

print("\nLAST 50 REF BASES:")
print(context[-50:])

# ============================================================
# TOKENIZATION
# ============================================================

print("=" * 80)
print("TOKENIZATION")
print("=" * 80)

tokens = tokenizer.tokenize(context)

print("Token Count:", len(tokens))

print("First 20 Tokens:")
print(tokens[:20])

print("Last 20 Tokens:")
print(tokens[-20:])

# ============================================================
# PREPARE BATCH
# ============================================================

print("=" * 80)
print("PREPARING BATCH")
print("=" * 80)

batch = prepare_batch(
    [context],
    tokenizer,
    device=DEVICE,
)

input_ids = batch[0]

print("Input Shape:", input_ids.shape)

# ============================================================
# FORWARD PASS
# ============================================================

print("=" * 80)
print("FORWARD PASS")
print("=" * 80)

with torch.no_grad():

    outputs = model.model(input_ids)

# ------------------------------------------------------------
# HANDLE BOTH OUTPUT TYPES
# ------------------------------------------------------------

if isinstance(outputs, tuple):
    logits = outputs[0]
elif hasattr(outputs, "logits"):
    logits = outputs.logits
else:
    logits = outputs

print("Logits Shape:", logits.shape)

# ============================================================
# SHIFTED AUTOREGRESSIVE SCORING
# ============================================================

print("=" * 80)
print("AUTOREGRESSIVE SCORING")
print("=" * 80)

shift_logits = logits[:, :-1, :]
shift_labels = input_ids[:, 1:]

print("Shift Logits Shape:", shift_logits.shape)
print("Shift Labels Shape:", shift_labels.shape)

log_probs = torch.nn.functional.log_softmax(
    shift_logits,
    dim=-1,
)

token_logprobs = torch.gather(
    log_probs,
    dim=-1,
    index=shift_labels.unsqueeze(-1),
).squeeze(-1)

token_logprobs = token_logprobs[0]

print("Scored Tokens:", len(token_logprobs))

# ============================================================
# TOKEN LOGPROBS
# ============================================================

print("=" * 80)
print("TOKEN LOGPROBS")
print("=" * 80)

print("First 20 Token Logprobs:")
print(token_logprobs[:20])

print("\nLast 20 Token Logprobs:")
print(token_logprobs[-20:])

# ============================================================
# REDUCTION SEMANTICS
# ============================================================

print("=" * 80)
print("REDUCTION SEMANTICS")
print("=" * 80)

sum_logprob = token_logprobs.sum().item()

mean_logprob = token_logprobs.mean().item()

print("SUM LOGPROB :", sum_logprob)

print("MEAN LOGPROB:", mean_logprob)

# ============================================================
# RUNNING CUMULATIVE
# ============================================================

print("=" * 80)
print("RUNNING CUMULATIVE")
print("=" * 80)

running_sum = 0.0

for i in range(min(20, len(token_logprobs))):

    running_sum += token_logprobs[i].item()

    running_mean = running_sum / (i + 1)

    print(
        f"STEP={i+1} "
        f"TOKEN_LLP={token_logprobs[i].item():.6f} "
        f"RUN_SUM={running_sum:.6f} "
        f"RUN_MEAN={running_mean:.6f}"
    )

print("=" * 80)
print("TRACE COMPLETE")
print("=" * 80)
