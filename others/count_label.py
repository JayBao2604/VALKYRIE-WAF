import pandas as pd
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True, help='Path to original reward_dataset.csv')
    parser.add_argument('--output_csv', type=str, default='reward_dataset_balanced.csv')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)

    # Load dataset
    df = pd.read_csv(args.input_csv)

    if "payload" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'payload' and 'label' columns.")

    # Convert label to float explicitly (important!)
    df["label"] = df["label"].astype(float)

    # Split by label
    positives = df[df["label"] == 1.0]
    negatives = df[df["label"] == 0.0]

    n_pos, n_neg = len(positives), len(negatives)
    print(f"[i] Found {n_pos} positives and {n_neg} negatives")

    if n_pos != n_neg:
        n_samples = min(n_pos, n_neg)
        positives = positives.sample(n=n_samples, random_state=args.seed)
        negatives = negatives.sample(n=n_samples, random_state=args.seed)
        print(f"[!] Balanced dataset to {n_samples} samples per class")
    else:
        print(f"[✓] Dataset is already balanced.")

    # Combine and shuffle
    balanced_df = pd.concat([positives, negatives]).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Ensure label remains float (not coerced back to int)
    balanced_df["label"] = balanced_df["label"].astype(float)

    # Save to CSV
    balanced_df.to_csv(args.output_csv, index=False)
    print(f"[✔] Saved balanced dataset to: {args.output_csv} ({len(balanced_df)} samples)")

if __name__ == "__main__":
    main()
