"""Evaluate a paired Tree-Ring clean/watermarked image dataset.

Python 3.7 compatible. This script never generates diffusion images. It pairs
existing images, computes paired visual metrics and prepares exact FID folders.
"""
from __future__ import print_function

import argparse
import csv
import fnmatch
import hashlib
import json
import math
import os
import re
import shutil

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def list_images(folder, pattern="*"):
    paths = []
    for root, _, names in os.walk(folder):
        for name in names:
            if (name.lower().endswith(IMAGE_EXTENSIONS)
                    and fnmatch.fnmatch(name, pattern)):
                paths.append(os.path.join(root, name))
    return sorted(paths)


def image_index(path):
    numbers = re.findall(r"\d+", os.path.splitext(os.path.basename(path))[0])
    if not numbers:
        raise ValueError("filename has no numeric index: {}".format(path))
    return int(numbers[-1])


def indexed(paths, label):
    result = {}
    for path in paths:
        index = image_index(path)
        if index in result:
            raise ValueError("duplicate {} index {}: {} and {}".format(
                label, index, result[index], path))
        result[index] = path
    return result


def load_prompts(path):
    with open(path, "r") as handle:
        return [line.strip() for line in handle if line.strip()]


def psnr(a, b):
    import numpy as np
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    mse = float(np.mean((aa - bb) ** 2))
    return float("inf") if mse == 0.0 else 10.0 * math.log10(255.0 * 255.0 / mse)


def bootstrap_mean_ci(values, iterations, seed, confidence=0.95):
    import numpy as np
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if values.size == 0 or iterations <= 0:
        return None
    rng = np.random.RandomState(seed)
    means = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 256):
        count = min(256, iterations - start)
        indices = rng.randint(0, values.size, size=(count, values.size))
        means[start:start + count] = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "method": "image_level_percentile_bootstrap",
        "confidence": confidence,
        "iterations": int(iterations),
        "seed": int(seed),
        "low": float(np.percentile(means, 100.0 * alpha)),
        "high": float(np.percentile(means, 100.0 * (1.0 - alpha))),
    }


def prepare_fid_set(output_dir, pairs):
    root = os.path.join(output_dir, "fid_sets")
    clean_out = os.path.join(root, "clean")
    marked_out = os.path.join(root, "watermarked")
    if os.path.exists(root) and any(os.scandir(root)):
        raise RuntimeError("FID output already exists and is nonempty: {}".format(root))
    os.makedirs(clean_out, exist_ok=True)
    os.makedirs(marked_out, exist_ok=True)
    records = []
    for ordinal, pair in enumerate(pairs):
        ext_a = os.path.splitext(pair["clean"])[1].lower()
        ext_b = os.path.splitext(pair["watermarked"])[1].lower()
        clean_name = "{:06d}{}".format(ordinal, ext_a)
        marked_name = "{:06d}{}".format(ordinal, ext_b)
        clean_dest = os.path.join(clean_out, clean_name)
        marked_dest = os.path.join(marked_out, marked_name)
        shutil.copy2(pair["clean"], clean_dest)
        shutil.copy2(pair["watermarked"], marked_dest)
        records.append({
            "ordinal": ordinal, "index": pair["index"],
            "clean_source": os.path.abspath(pair["clean"]),
            "watermarked_source": os.path.abspath(pair["watermarked"]),
            "clean_file": clean_name, "watermarked_file": marked_name,
            "clean_sha256": sha256(clean_dest),
            "watermarked_sha256": sha256(marked_dest),
        })
    manifest = {
        "pairs": len(records), "pairing": "numeric filename index",
        "clean_dir": os.path.abspath(clean_out),
        "watermarked_dir": os.path.abspath(marked_out), "records": records,
    }
    with open(os.path.join(root, "fid_manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return clean_out, marked_out


def main(args):
    prompts = load_prompts(args.prompt_file)
    clean = indexed(list_images(args.clean_dir, args.clean_pattern), "clean")
    marked = indexed(list_images(args.watermarked_dir,
                                 args.watermarked_pattern), "watermarked")
    common = sorted(set(clean) & set(marked))
    missing_clean = sorted(set(marked) - set(clean))
    missing_marked = sorted(set(clean) - set(marked))
    if missing_clean or missing_marked:
        raise RuntimeError("unpaired files; missing clean={} missing watermarked={}".format(
            missing_clean[:20], missing_marked[:20]))
    if args.expected_count and len(common) != args.expected_count:
        raise RuntimeError("expected {} pairs, found {}".format(
            args.expected_count, len(common)))
    if not common:
        raise RuntimeError("no paired images found")
    pairs = []
    for index in common:
        prompt_index = index - args.index_offset
        if prompt_index < 0 or prompt_index >= len(prompts):
            raise RuntimeError("index {} maps outside prompt file".format(index))
        pairs.append({"index": index, "prompt": prompts[prompt_index],
                      "clean": clean[index], "watermarked": marked[index]})

    print("pairing preflight: pairs={} first_index={} last_index={} index_offset={}".format(
        len(pairs), pairs[0]["index"], pairs[-1]["index"], args.index_offset))
    if args.preflight_only:
        print("preflight only: no output directory created and no metrics computed")
        return
    if not args.clip_checkpoint:
        raise RuntimeError("--clip_checkpoint is required unless --preflight_only is used")

    from tqdm import tqdm
    from analyze_visual_quality import (
        clip_scores, compute_lpips, compute_ssim, finite_stats, init_clip,
        init_lpips, load_rgb,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    fid_clean, fid_marked = prepare_fid_set(args.output_dir, pairs)
    lpips_model = init_lpips(args.lpips_net, args.device)
    rows, clean_images, marked_images, pair_prompts = [], [], [], []
    for pair in tqdm(pairs, desc="Tree-Ring LPIPS/SSIM/PSNR"):
        a, b = load_rgb(pair["clean"]), load_rgb(pair["watermarked"])
        if a.size != b.size:
            raise ValueError("shape mismatch at index {}".format(pair["index"]))
        rows.append({
            "index": pair["index"], "prompt": pair["prompt"],
            "clean_path": os.path.abspath(pair["clean"]),
            "watermarked_path": os.path.abspath(pair["watermarked"]),
            "psnr": psnr(a, b), "ssim": compute_ssim(a, b),
            "lpips": compute_lpips(lpips_model, a, b, args.device),
            "clip_clean": None, "clip_watermarked": None, "delta_clip": None,
        })
        clean_images.append(a); marked_images.append(b); pair_prompts.append(pair["prompt"])

    backend, model, preprocess, tokenizer = init_clip(
        args.clip_model, None, args.clip_checkpoint, args.clip_backend, args.device)
    clean_clip = clip_scores(backend, model, preprocess, tokenizer,
                             clean_images, pair_prompts, args.device, args.batch_size)
    marked_clip = clip_scores(backend, model, preprocess, tokenizer,
                              marked_images, pair_prompts, args.device, args.batch_size)
    for row, ca, cb in zip(rows, clean_clip, marked_clip):
        row["clip_clean"] = float(ca)
        row["clip_watermarked"] = float(cb)
        row["delta_clip"] = float(cb - ca)

    csv_path = os.path.join(args.output_dir, "treering_visual_metrics.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    metrics = {}
    for offset, name in enumerate(("psnr", "lpips", "ssim", "clip_clean",
                                   "clip_watermarked", "delta_clip")):
        values = [row[name] for row in rows]
        metrics[name] = finite_stats(values)
        metrics[name]["mean_ci95"] = bootstrap_mean_ci(
            values, args.bootstrap_iterations, args.bootstrap_seed + offset)
    summary = {
        "method": "Tree-Ring", "model": "Stable Diffusion 2.1-base",
        "pairs": len(rows), "prompt_file": os.path.abspath(args.prompt_file),
        "clean_dir": os.path.abspath(args.clean_dir),
        "watermarked_dir": os.path.abspath(args.watermarked_dir),
        "pairing": {"rule": "last numeric token in filename",
                    "index_offset": args.index_offset,
                    "clean_pattern": args.clean_pattern,
                    "watermarked_pattern": args.watermarked_pattern},
        "lpips_net": args.lpips_net, "clip_backend": backend,
        "clip_checkpoint": os.path.abspath(args.clip_checkpoint),
        "metrics": metrics,
        "fid_command_inputs": {"clean": fid_clean, "watermarked": fid_marked},
        "interpretation": "Quality-only comparison. Tree-Ring detection is not bit-message recovery.",
    }
    summary_path = os.path.join(args.output_dir, "treering_visual_metrics_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("pairs: {}".format(len(rows)))
    for name, item in metrics.items():
        print("{} mean={:.6f} p05={:.6f} p95={:.6f}".format(
            name, item["mean"], item["p05"], item["p95"]))
    print("wrote: {}".format(csv_path))
    print("wrote: {}".format(summary_path))
    print("FID clean: {}".format(fid_clean))
    print("FID watermarked: {}".format(fid_marked))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="paired Tree-Ring visual comparison")
    parser.add_argument("--clean_dir", required=True)
    parser.add_argument("--watermarked_dir", required=True)
    parser.add_argument("--clean_pattern", default="*")
    parser.add_argument("--watermarked_pattern", default="*")
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_count", type=int, default=172)
    parser.add_argument("--index_offset", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--lpips_net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--clip_backend", default="transformers",
                        choices=["auto", "transformers", "open_clip"])
    parser.add_argument("--clip_model", default="ViT-L-14")
    parser.add_argument("--clip_checkpoint", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20261023)
    parser.add_argument("--preflight_only", action="store_true")
    main(parser.parse_args())
