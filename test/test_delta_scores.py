import os
import pandas as pd
from pyfaidx import Fasta
from evo2 import Evo2

def main():
    # 1. Load Evo2 (Uses adaptive runtime automatically)
    print("Loading Evo2...")
    try:
        model = Evo2("evo2_7b")
    except Exception as e:
        print(f"Failed to load Evo2: {e}")
        return

    # 2. Setup inputs
    csv_path = "test/cosmic_variants_10.csv"
    fasta_path = "hg38.fa"

    if not os.path.exists(csv_path):
        print(f"Error: Dataset {csv_path} not found.")
        return

    if not os.path.exists(fasta_path):
        print(f"Warning: {fasta_path} not found. Please ensure hg38.fa is available in the current directory.")

    print(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    try:
        fasta = Fasta(fasta_path)
    except Exception as e:
        print(f"Error loading FASTA ({fasta_path}): {e}")
        fasta = None

    # 3. Process variants
    results = []
    
    # Context window sizing (512 bp on each side -> 1024 bp total window)
    context_left = 512
    context_right = 512

    for index, row in df.iterrows():
        chrom = str(row['CHROMOSOME'])
        if not chrom.startswith('chr'):
            chrom = 'chr' + chrom
            
        pos = int(row['GENOME_START'])
        wt_allele = str(row['GENOMIC_WT_ALLELE'])
        mut_allele = str(row['GENOMIC_MUT_ALLELE'])

        print(f"\n[{index}] Processing {chrom}:{pos} {wt_allele} > {mut_allele}")

        if fasta is None:
            continue

        # Convert to 0-indexed for pyfaidx slicing
        idx = pos - 1

        try:
            # 4. Construct sequences
            left_seq = str(fasta[chrom][idx - context_left : idx])
            right_seq = str(fasta[chrom][idx + len(wt_allele) : idx + len(wt_allele) + context_right])
            
            wt_seq = left_seq + wt_allele + right_seq
            mut_seq = left_seq + mut_allele + right_seq

            # 5. Score sequences
            # Using reduce_method='sum' to compute total log probability across the sequence
            scores = model.score_sequences(
                [wt_seq, mut_seq],
                batch_size=2,
                reduce_method='sum'
            )
            
            wt_score = scores[0]
            mut_score = scores[1]
            delta_score = mut_score - wt_score

            print(f"  WT Score:    {wt_score:.4f}")
            print(f"  MUT Score:   {mut_score:.4f}")
            print(f"  Delta Score: {delta_score:.4f}")

            results.append({
                'chromosome': chrom,
                'position': pos,
                'ref': wt_allele,
                'alt': mut_allele,
                'wt_score': wt_score,
                'mut_score': mut_score,
                'delta_score': delta_score
            })

        except Exception as e:
            print(f"  Error processing variant: {e}")

    # 6. Save results
    if results:
        out_path = "test/delta_scores_output.csv"
        out_df = pd.DataFrame(results)
        # Ensure output directory exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"\nSaved {len(results)} variant scores to {out_path}")

if __name__ == "__main__":
    main()
