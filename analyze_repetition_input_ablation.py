"""Compare hard majority voting and soft LLR summation offline."""

from __future__ import print_function

import argparse
import csv
import json
import os

import numpy as np


def main(args):
    records = []
    with open(os.path.join(args.run_dir, "results.jsonl"), "r") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    rows = []
    for record in records:
        rel = record.get("carrier_llrs")
        if not rel:
            raise ValueError("record {} lacks carrier_llrs".format(
                record.get("index")))
        llrs = np.load(os.path.join(args.run_dir, rel)).reshape(
            args.payload_bits, args.repeat)
        secret = np.asarray(record["secret_bits"], dtype=np.int8)
        hard_carriers = (llrs < 0.0).astype(np.int8)
        hard_bits = (hard_carriers.sum(axis=1) > args.repeat / 2.0).astype(
            np.int8)
        soft_bits = (llrs.sum(axis=1) < 0.0).astype(np.int8)
        for mode, decoded in (("hard_vote", hard_bits),
                              ("soft_sum", soft_bits)):
            errors = int(np.sum(decoded != secret))
            rows.append({
                "index": int(record["index"]),
                "mode": mode,
                "bit_errors": errors,
                "ber": errors / float(len(secret)),
                "exact_recovery": errors == 0,
            })
    output_csv = args.output_csv or os.path.join(
        args.run_dir, "repetition_input_ablation.csv")
    output_summary = args.output_summary or os.path.join(
        args.run_dir, "repetition_input_ablation_summary.json")
    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "index", "mode", "bit_errors", "ber", "exact_recovery"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for mode in ("hard_vote", "soft_sum"):
        selected = [row for row in rows if row["mode"] == mode]
        summary[mode] = {
            "images": len(selected),
            "mean_secret_ber": float(np.mean([
                row["ber"] for row in selected])),
            "total_secret_bit_errors": int(sum(
                row["bit_errors"] for row in selected)),
            "exact_recovery_count": int(sum(
                row["exact_recovery"] for row in selected)),
            "exact_recovery_rate": float(np.mean([
                row["exact_recovery"] for row in selected])),
        }
    with open(output_summary, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("wrote: {}".format(output_csv))
    print("wrote: {}".format(output_summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--payload_bits", type=int, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_summary", default=None)
    main(parser.parse_args())
