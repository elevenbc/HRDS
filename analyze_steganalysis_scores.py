"""Summarize SRNet/LWENet/Siamese detector prediction CSV files.

Each input CSV must contain split, binary label, and continuous stego score
columns. Threshold selection uses validation data only; all reported primary
metrics use the test split. If pair_id is present, confidence intervals
bootstrap complete cover/stego pairs together.
"""

from __future__ import print_function

import argparse
import csv
import json
import os

import numpy as np


def parse_prediction(value):
    if "=" not in value:
        raise ValueError("--prediction must be MODEL=CSV_PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise ValueError("--prediction must be MODEL=CSV_PATH")
    return name, path


def read_csv(path, args):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {args.split_col, args.label_col, args.score_col}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("{} missing columns: {}".format(
                path, sorted(missing)))
        for row_number, row in enumerate(reader, 2):
            try:
                rows.append({
                    "split": row[args.split_col].strip().lower(),
                    "label": int(row[args.label_col]),
                    "score": float(row[args.score_col]),
                    "pair_id": row.get(args.pair_col) or str(row_number),
                })
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid row {} in {}: {}".format(
                    row_number, path, exc))
    return rows


def roc_curve(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional and aligned")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("labels must contain only 0 and 1")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("ROC requires both cover and stego samples")

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    distinct = np.where(np.diff(sorted_scores))[0]
    threshold_indices = np.r_[distinct, labels.size - 1]
    true_positives = np.cumsum(sorted_labels)[threshold_indices]
    false_positives = (
        1 + threshold_indices - true_positives)
    tpr = np.r_[0.0, true_positives / float(positives)]
    fpr = np.r_[0.0, false_positives / float(negatives)]
    thresholds = np.r_[np.inf, sorted_scores[threshold_indices]]
    return fpr, tpr, thresholds


def roc_auc(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    widths = np.diff(fpr)
    heights = (tpr[:-1] + tpr[1:]) / 2.0
    return float(np.sum(widths * heights))


def select_threshold(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    finite = np.isfinite(thresholds)
    candidates = np.flatnonzero(finite)
    if candidates.size == 0:
        return 0.5
    objective = tpr[candidates] - fpr[candidates]
    return float(thresholds[candidates[int(np.argmax(objective))]])


def tpr_at_fpr(labels, scores, target_fpr):
    fpr, tpr, _ = roc_curve(labels, scores)
    eligible = np.flatnonzero(fpr <= target_fpr)
    return float(tpr[eligible].max()) if eligible.size else 0.0


def metrics(labels, scores, threshold, target_fpr):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores >= threshold).astype(np.int64)
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    accuracy = (tp + tn) / float(labels.size)
    true_positive_rate = tp / float(tp + fn) if tp + fn else 0.0
    true_negative_rate = tn / float(tn + fp) if tn + fp else 0.0
    precision = tp / float(tp + fp) if tp + fp else 0.0
    f1 = (2.0 * precision * true_positive_rate /
          (precision + true_positive_rate)
          if precision + true_positive_rate else 0.0)
    return {
        "images": int(labels.size),
        "cover_images": int(np.sum(labels == 0)),
        "stego_images": int(np.sum(labels == 1)),
        "threshold": float(threshold),
        "roc_auc": roc_auc(labels, scores),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(
            (true_positive_rate + true_negative_rate) / 2.0),
        "precision": float(precision),
        "recall": float(true_positive_rate),
        "f1": float(f1),
        "tpr_at_fpr_{:.4f}".format(target_fpr): tpr_at_fpr(
            labels, scores, target_fpr),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap(rows, threshold, target_fpr, iterations, seed):
    groups = {}
    for row in rows:
        groups.setdefault(row["pair_id"], []).append(row)
    group_ids = sorted(groups)
    rng = np.random.RandomState(seed)
    values = {"roc_auc": [], "balanced_accuracy": []}
    tpr_key = "tpr_at_fpr_{:.4f}".format(target_fpr)
    values[tpr_key] = []
    for _ in range(iterations):
        sampled = rng.randint(0, len(group_ids), size=len(group_ids))
        selected = []
        for index in sampled:
            selected.extend(groups[group_ids[index]])
        labels = [row["label"] for row in selected]
        if len(set(labels)) < 2:
            continue
        scores = [row["score"] for row in selected]
        result = metrics(labels, scores, threshold, target_fpr)
        for name in values:
            values[name].append(result[name])
    intervals = {}
    for name, samples in values.items():
        if samples:
            intervals[name] = {
                "low": float(np.percentile(samples, 2.5)),
                "high": float(np.percentile(samples, 97.5)),
                "valid_iterations": len(samples),
            }
        else:
            intervals[name] = {
                "low": None, "high": None, "valid_iterations": 0}
    return {
        "method": "pair_cluster_percentile_bootstrap",
        "requested_iterations": iterations,
        "seed": seed,
        "intervals": intervals,
    }


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    outputs = {}
    flat_rows = []
    for offset, prediction in enumerate(args.prediction):
        name, path = parse_prediction(prediction)
        rows = read_csv(path, args)
        val = [row for row in rows if row["split"] == args.val_split]
        test = [row for row in rows if row["split"] == args.test_split]
        if not val or not test:
            raise ValueError(
                "{} needs non-empty {} and {} splits".format(
                    name, args.val_split, args.test_split))
        val_labels = [row["label"] for row in val]
        test_labels = [row["label"] for row in test]
        if len(set(val_labels)) < 2 or len(set(test_labels)) < 2:
            raise ValueError("{} val/test must each contain both labels".format(
                name))
        threshold = select_threshold(
            val_labels, [row["score"] for row in val])
        result = metrics(
            test_labels, [row["score"] for row in test], threshold,
            args.target_fpr)
        result["threshold_selection"] = (
            "max validation Youden J (TPR-FPR)")
        result["validation_images"] = len(val)
        result["prediction_csv"] = os.path.abspath(path)
        result["confidence_intervals_95"] = bootstrap(
            test, threshold, args.target_fpr, args.bootstrap_iterations,
            args.bootstrap_seed + offset)
        outputs[name] = result

        fpr, tpr, thresholds = roc_curve(
            test_labels, [row["score"] for row in test])
        roc_path = os.path.join(args.output_dir, "{}_roc.csv".format(name))
        with open(roc_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["fpr", "tpr", "threshold"])
            writer.writerows(zip(fpr, tpr, thresholds))
        flat_rows.append({
            "model": name,
            "test_images": result["images"],
            "threshold": result["threshold"],
            "roc_auc": result["roc_auc"],
            "balanced_accuracy": result["balanced_accuracy"],
            "accuracy": result["accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            tpr_key(args.target_fpr): result[tpr_key(args.target_fpr)],
        })

    json_path = os.path.join(args.output_dir, "steganalysis_summary.json")
    csv_path = os.path.join(args.output_dir, "steganalysis_summary.csv")
    with open(json_path, "w") as handle:
        json.dump(outputs, handle, indent=2, sort_keys=True)
        handle.write("\n")
    fieldnames = list(flat_rows[0].keys())
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps(outputs, indent=2, sort_keys=True))
    print("wrote: {}".format(json_path))
    print("wrote: {}".format(csv_path))


def tpr_key(target_fpr):
    return "tpr_at_fpr_{:.4f}".format(target_fpr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction", action="append", required=True,
        help="repeat MODEL=CSV_PATH for SRNet, LWENet, Siamese, etc.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_col", default="split")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--score_col", default="score")
    parser.add_argument("--pair_col", default="pair_id")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--target_fpr", type=float, default=0.01)
    parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260917)
    main(parser.parse_args())
