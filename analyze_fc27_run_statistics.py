"""Create publication-ready communication summaries with uncertainty."""

from __future__ import print_function

import argparse
import csv
import json
import math
import os

import numpy as np


def parse_run(value):
    if "=" not in value:
        raise ValueError("--run must be LABEL=RUN_DIR")
    return value.split("=", 1)


def load_records(run_dir):
    records = []
    with open(os.path.join(run_dir, "results.jsonl"), "r") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except ValueError as exc:
                    raise ValueError("invalid JSON at line {}: {}".format(
                        line_no, exc))
    return records


def wilson(successes, total, z=1.959963984540054):
    if total == 0:
        return [None, None]
    p = successes / float(total)
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        p * (1.0 - p) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def bootstrap_mean(values, iterations, seed):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.RandomState(seed)
    means = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 256):
        count = min(256, iterations - start)
        indices = rng.randint(0, values.size, size=(count, values.size))
        means[start:start + count] = values[indices].mean(axis=1)
    return [float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5))]


def main(args):
    outputs = {}
    flat = []
    for offset, value in enumerate(args.run):
        label, run_dir = parse_run(value)
        records = load_records(run_dir)
        if not records:
            raise ValueError("empty run: {}".format(run_dir))
        exact = [bool(row.get("exact_recovery", False)) for row in records]
        checks_available = all(
            "ldpc_checks_satisfied" in row for row in records)
        checks = ([bool(row["ldpc_checks_satisfied"]) for row in records]
                  if checks_available else None)
        bers = [float(row.get("ber", row.get("secret_ber", 0.0)))
                for row in records]
        coded = [float(row["coded_ber"]) for row in records
                 if row.get("coded_ber") is not None]
        exact_count = int(sum(exact))
        result = {
            "run_dir": os.path.abspath(run_dir),
            "images": len(records),
            "exact_recovery_count": exact_count,
            "exact_recovery_rate": exact_count / float(len(records)),
            "exact_recovery_wilson_ci95": wilson(
                exact_count, len(records)),
            "mean_secret_ber": float(np.mean(bers)),
            "mean_secret_ber_image_bootstrap_ci95": bootstrap_mean(
                bers, args.bootstrap_iterations, args.bootstrap_seed + offset),
            "total_secret_bit_errors": int(sum(
                int(row.get("bit_errors", 0)) for row in records)),
            "failed_indices": [int(row.get("index", i))
                               for i, row in enumerate(records)
                               if not bool(row.get("exact_recovery", False))],
        }
        if coded:
            result["mean_coded_ber"] = float(np.mean(coded))
            result["mean_coded_ber_image_bootstrap_ci95"] = bootstrap_mean(
                coded, args.bootstrap_iterations,
                args.bootstrap_seed + 1000 + offset)
            result["total_coded_bit_errors"] = int(sum(
                int(row.get("coded_bit_errors", 0)) for row in records))
        if checks is not None:
            check_count = int(sum(checks))
            result["checks_satisfied_count"] = check_count
            result["checks_satisfied_rate"] = check_count / float(len(records))
            result["checks_satisfied_wilson_ci95"] = wilson(
                check_count, len(records))
        outputs[label] = result
        flat.append({
            "label": label,
            "images": result["images"],
            "exact_count": exact_count,
            "exact_rate": result["exact_recovery_rate"],
            "exact_ci95_low": result["exact_recovery_wilson_ci95"][0],
            "exact_ci95_high": result["exact_recovery_wilson_ci95"][1],
            "mean_secret_ber": result["mean_secret_ber"],
            "secret_ber_ci95_low": (
                result["mean_secret_ber_image_bootstrap_ci95"][0]),
            "secret_ber_ci95_high": (
                result["mean_secret_ber_image_bootstrap_ci95"][1]),
            "mean_coded_ber": result.get("mean_coded_ber"),
        })

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "communication_statistics.json")
    csv_path = os.path.join(args.output_dir, "communication_statistics.csv")
    with open(json_path, "w") as handle:
        json.dump(outputs, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0].keys()))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps(outputs, indent=2, sort_keys=True))
    print("wrote: {}".format(json_path))
    print("wrote: {}".format(csv_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True,
                        help="repeat LABEL=RUN_DIR")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260917)
    main(parser.parse_args())
