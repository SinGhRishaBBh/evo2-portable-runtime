import random
import time
from datetime import datetime

# =========================================================
# FIXED SEED FOR REPRODUCIBLE OUTPUT
# =========================================================

random.seed(42)

DNA_BASES = ["A", "C", "G", "T"]


# =========================================================
# LOGGER
# =========================================================

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


# =========================================================
# DNA GENERATION
# =========================================================

def generate_dna(length=1000):
    return "".join(random.choice(DNA_BASES) for _ in range(length))


# =========================================================
# DNA MUTATION
# =========================================================

def mutate_sequence(seq, num_mutations=5):

    seq = list(seq)

    for _ in range(num_mutations):

        idx = random.randint(0, len(seq) - 1)

        original = seq[idx]

        choices = [b for b in DNA_BASES if b != original]

        seq[idx] = random.choice(choices)

    return "".join(seq)


# =========================================================
# MAIN
# =========================================================

start_time = time.time()

print("========================================")
log("DNA GENERATION TEST STARTED")
print("========================================")

reference = generate_dna(1000)

mutated = mutate_sequence(reference, 5)

print()

log("Reference DNA:")
print(reference)

print()

log("Mutated DNA:")
print(mutated)

print()

log(f"Sequence Length: {len(reference)}")

end_time = time.time()

runtime = end_time - start_time

print()
print("========================================")
log("DNA GENERATION TEST FINISHED")
log(f"TOTAL RUNTIME: {runtime:.6f} seconds")
print("========================================")
