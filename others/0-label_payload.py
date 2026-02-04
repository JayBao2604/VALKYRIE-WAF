import os
import csv
import argparse
from tools.waf_tester import test_against_waf


def label_payloads_from_txt(input_txt: str, output_csv: str):
    output_rows = []

    with open(input_txt, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for i, payload in enumerate(lines):
        bypassed = test_against_waf(payload)
        label = 1 if bypassed else 0  # 1 = bypass, 0 = blocked

        output_rows.append([payload, label])
        print(f"[{i+1}/{len(lines)}] Label = {label} → {payload}")

    # Write to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["payload", "label"])  # header
        writer.writerows(output_rows)

    print(f"\n[+] Saved {len(output_rows)} labeled payloads to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_txt", type=str, required=True, help="Input .txt file with one payload per line")
    parser.add_argument("--output_csv", type=str, required=True, help="Output .csv file with labels")
    args = parser.parse_args()

    label_payloads_from_txt(args.input_txt, args.output_csv)
