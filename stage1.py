import argparse
import os
import csv
from tools.reward_model import score_payload
from tools.grammar_sampler import refine_if_needed
from tools.waf_tester import test_against_waf

def load_payloads(payload_file):
    with open(payload_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def run_refinement_loop(args):
    payloads = load_payloads(args.payload_file)
    output_csv = f"data/stage1/reward_dataset_{args.attack_type}.csv"

    with open(output_csv, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["payload", "label"])

        for idx, payload in enumerate(payloads):
            score = score_payload(payload)
            print(f"[DEBUG] Score: {score:.4f} → {payload}")

            if score < args.threshold:
                refined_payload, new_score = refine_if_needed(
                    original_payload=payload,
                    original_score=score,
                    score_threshold=args.threshold,
                    max_attempts=args.max_refine
                )
                print(f"[DEBUG] Refined: {new_score:.4f} → {refined_payload}")
                payload = refined_payload
                score = new_score

            if score >= args.threshold:
                label = int(test_against_waf(payload))
                print(f"[WAF] Payload: {payload} → Label: {label}")
                writer.writerow([payload, label])
                print(f"[{idx+1}/{len(payloads)}] ✅ Saved.")

    print(f"\n✅ Done. Output saved to: {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--payload_file', required=True,
                        help='Path to .txt file containing one payload per line')
    parser.add_argument('--attack_type', required=True,
                        help='Attack type (used in output filename)')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='Refine if score < this threshold (default: 0.8)')
    parser.add_argument('--max_refine', type=int, default=15,
                        help='Max refinement attempts per payload')
    args = parser.parse_args()

    run_refinement_loop(args)
