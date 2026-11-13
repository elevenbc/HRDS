"""Compare soft and hard-input LDPC decoding on one completed run.

This is an offline ablation: it reuses each saved message-LLR vector and
therefore holds the generated image, attack, codeword, and carrier errors
fixed. It does not run Stable Diffusion.
"""

from __future__ import print_function

import argparse
import csv
import json
import os

import numpy as np

from ecc_utils import LDPCCode


def load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def load_records(path):
    rows = []
    with open(path, "r") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError as exc:
                    raise ValueError("invalid JSON at line {}: {}".format(
                        line_no, exc))
    return rows


def decode_one(code, record, llrs, mode, hard_magnitude, iterations, scaling):
    received = np.asarray(record["decoded_coded_bits"], dtype=np.int8)
    if mode == "soft":
        channel = np.asarray(llrs, dtype=np.float64)
    else:
        channel = np.where(received == 0, hard_magnitude, -hard_magnitude)
    decoded = code.decode(
        received, channel, max_iterations=iterations, scaling=scaling)
    secret = np.asarray(record["secret_bits"], dtype=np.int8)
    bit_errors = int(np.sum(decoded["secret_bits"] != secret))
    return {
        "bit_errors": bit_errors,
        "ber": bit_errors / float(len(secret)),
        "exact_recovery": bit_errors == 0,
        "checks_satisfied": bool(decoded["checks_satisfied"]),
        "iterations_used": int(decoded["iterations_used"]),
    }


def main(args):
    manifest = load_json(os.path.join(args.run_dir, "manifest.json"))
    summary = load_json(os.path.join(args.run_dir, "summary.json"))
    records = load_records(os.path.join(args.run_dir, "results.jsonl"))
    ldpc = summary.get("ldpc")
    if not ldpc:
        raise ValueError("run is not an LDPC experiment")
    code = LDPCCode(
        n=int(ldpc["n"]), k=int(ldpc["k"]), seed=int(ldpc["seed"]),
        column_weight=int(ldpc["column_weight"]))
    iterations = args.max_iterations or int(ldpc["max_iterations"])
    scaling = args.scaling if args.scaling is not None else float(ldpc["scaling"])

    rows = []
    for record in records:
        rel = record.get("message_llrs")
        if not rel:
            raise ValueError(
                "record {} has no message_llrs; rerun with --save_llrs".format(
                    record.get("index")))
        llrs = np.load(os.path.join(args.run_dir, rel))
        for mode in ("soft", "hard"):
            result = decode_one(
                code, record, llrs, mode, args.hard_magnitude,
                iterations, scaling)
            result.update({
                "index": int(record["index"]),
                "seed": int(record["seed"]),
                "mode": mode,
            })
            rows.append(result)

    output_csv = args.output_csv or os.path.join(
        args.run_dir, "ldpc_input_ablation.csv")
    output_summary = args.output_summary or os.path.join(
        args.run_dir, "ldpc_input_ablation_summary.json")
    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "index", "seed", "mode", "bit_errors", "ber",
            "exact_recovery", "checks_satisfied", "iterations_used"])
        writer.writeheader()
        writer.writerows(rows)

    aggregates = {}
    for mode in ("soft", "hard"):
        selected = [row for row in rows if row["mode"] == mode]
        aggregates[mode] = {
            "images": len(selected),
            "total_secret_bit_errors": int(sum(
                row["bit_errors"] for row in selected)),
            "mean_secret_ber": float(np.mean([
                row["ber"] for row in selected])),
            "exact_recovery_count": int(sum(
                row["exact_recovery"] for row in selected)),
            "exact_recovery_rate": float(np.mean([
                row["exact_recovery"] for row in selected])),
            "checks_satisfied_count": int(sum(
                row["checks_satisfied"] for row in selected)),
            "mean_iterations": float(np.mean([
                row["iterations_used"] for row in selected])),
        }
    output = {
        "run_dir": os.path.abspath(args.run_dir),
        "source_prompt_sha256": manifest.get("prompt_sha256"),
        "hard_magnitude": args.hard_magnitude,
        "max_iterations": iterations,
        "scaling": scaling,
        "interpretation": (
            "Soft and hard-input decoding reuse identical saved channel "
            "observations. Hard input uses only the received bit sign and "
            "a constant reliability magnitude."),
        "modes": aggregates,
    }
    with open(output_summary, "w") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    print("wrote: {}".format(output_csv))
    print("wrote: {}".format(output_summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--hard_magnitude", type=float, default=1.0)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--scaling", type=float, default=None)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_summary", default=None)
    main(parser.parse_args())
