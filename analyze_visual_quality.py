"""Offline visual-quality metrics for a completed DCT-QIM run.

Computes, per result record:
  - LPIPS(clean, stego), lower is better
  - SSIM(clean, stego), higher is better
  - CLIPScore(prompt, clean)
  - CLIPScore(prompt, stego)
  - delta CLIP = CLIPScore(stego) - CLIPScore(clean)

The script does not run Stable Diffusion. It only reads images referenced by
results.jsonl in an existing run directory.
"""

from __future__ import print_function

import argparse
import csv
import json
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def load_records(run_dir):
    path = os.path.join(run_dir, "results.jsonl")
    if not os.path.isfile(path):
        raise IOError("missing results.jsonl: {}".format(path))
    records = []
    with open(path, "r") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError as exc:
                raise ValueError("invalid JSON at {}:{}: {}".format(
                    path, line_no, exc))
    return records


def resolve_image(run_dir, record, key, fallback_name):
    rel = record.get(key)
    path = os.path.join(run_dir, rel) if rel else os.path.join(
        run_dir, "images", fallback_name)
    if not os.path.isfile(path):
        return None
    return path


def load_rgb(path):
    return Image.open(path).convert("RGB")


def compute_ssim(clean, stego):
    a = np.asarray(clean, dtype=np.float32)
    b = np.asarray(stego, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError("image shape mismatch: {} vs {}".format(a.shape, b.shape))
    try:
        from skimage.metrics import structural_similarity
        try:
            return float(structural_similarity(
                a, b, data_range=255.0, channel_axis=2))
        except TypeError:
            return float(structural_similarity(
                a, b, data_range=255.0, multichannel=True))
    except ImportError:
        # Small fallback for older environments without scikit-image.
        from scipy.ndimage import convolve
        radius = 5
        sigma = 1.5
        axis = np.arange(-radius, radius + 1, dtype=np.float32)
        xx, yy = np.meshgrid(axis, axis)
        kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
        kernel /= kernel.sum()
        ga, gb = a.mean(axis=2), b.mean(axis=2)
        mu_a = convolve(ga, kernel, mode="constant", cval=0)
        mu_b = convolve(gb, kernel, mode="constant", cval=0)
        va = convolve(ga * ga, kernel, mode="constant", cval=0) - mu_a * mu_a
        vb = convolve(gb * gb, kernel, mode="constant", cval=0) - mu_b * mu_b
        cab = convolve(ga * gb, kernel, mode="constant", cval=0) - mu_a * mu_b
        c1 = (0.01 * 255.0) ** 2
        c2 = (0.03 * 255.0) ** 2
        value = ((2 * mu_a * mu_b + c1) * (2 * cab + c2)) / (
            (mu_a * mu_a + mu_b * mu_b + c1) * (va + vb + c2))
        return float(value.mean())


def init_lpips(net_name, device):
    try:
        import lpips
    except ImportError:
        raise RuntimeError(
            "LPIPS is unavailable. Install lpips or make the repository's "
            "LPIPS package visible on PYTHONPATH.")
    model = lpips.LPIPS(net=net_name).to(device)
    model.eval()
    return model


def image_tensor(image, device):
    arr = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def compute_lpips(model, clean, stego, device):
    with torch.no_grad():
        value = model(image_tensor(clean, device), image_tensor(stego, device))
    return float(value.reshape(-1)[0].item())


def init_clip(model_name, pretrained, checkpoint_path, backend, device):
    # Registry names would trigger a Hugging Face download. The remote
    # machine must use an already downloaded checkpoint or model directory.
    local_checkpoint = checkpoint_path
    if local_checkpoint is None and pretrained and os.path.exists(pretrained):
        local_checkpoint = pretrained
    if not local_checkpoint:
        raise RuntimeError(
            "No local CLIP checkpoint supplied. Use --clip_checkpoint "
            "/path/to/local_clip_model; network downloads are disabled.")
    if not os.path.exists(local_checkpoint):
        raise RuntimeError("CLIP checkpoint not found: {}".format(local_checkpoint))

    if backend in ("auto", "transformers") and os.path.isdir(local_checkpoint):
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            raise RuntimeError(
                "transformers is required for a Hugging Face CLIP model directory")
        model = CLIPModel.from_pretrained(
            local_checkpoint, local_files_only=True).to(device)
        processor = CLIPProcessor.from_pretrained(
            local_checkpoint, local_files_only=True)
        model.eval()
        return "transformers", model, processor, None

    if backend == "transformers":
        raise RuntimeError(
            "--clip_backend transformers requires a local model directory")
    try:
        import open_clip
    except ImportError:
        raise RuntimeError(
            "open_clip is unavailable. Install open_clip_torch and provide "
            "a local OpenCLIP checkpoint file.")
    if not os.path.isfile(local_checkpoint):
        raise RuntimeError("OpenCLIP checkpoint must be a file: {}".format(
            local_checkpoint))
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=None, checkpoint_path=local_checkpoint,
        device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return "open_clip", model, preprocess, tokenizer


def clip_scores(backend, model, preprocess, tokenizer, images, prompts,
                device, batch_size):
    scores = []
    for start in tqdm(range(0, len(images), batch_size), desc="CLIP"):
        image_slice = images[start:start + batch_size]
        prompt_slice = prompts[start:start + batch_size]
        with torch.no_grad():
            if backend == "transformers":
                inputs = preprocess(
                    text=prompt_slice, images=image_slice,
                    return_tensors="pt", padding=True).to(device)
                outputs = model(**inputs)
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
            else:
                image_batch = torch.stack([
                    preprocess(image) for image in image_slice
                ]).to(device)
                text_batch = tokenizer(prompt_slice).to(device)
                image_features = model.encode_image(image_batch)
                text_features = model.encode_text(text_batch)
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True).clamp_min(1e-12)
            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True).clamp_min(1e-12)
            batch_scores = (image_features * text_features).sum(dim=-1)
        scores.extend(batch_scores.detach().float().cpu().tolist())
    return scores


def finite_stats(values):
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def bootstrap_mean_ci(values, iterations, seed, confidence=0.95):
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if values.size == 0 or iterations <= 0:
        return None
    rng = np.random.RandomState(seed)
    means = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 256):
        count = min(256, iterations - start)
        sample_indices = rng.randint(
            0, values.size, size=(count, values.size))
        means[start:start + count] = values[sample_indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "method": "image_level_percentile_bootstrap",
        "confidence": confidence,
        "iterations": int(iterations),
        "seed": int(seed),
        "low": float(np.percentile(means, 100.0 * alpha)),
        "high": float(np.percentile(means, 100.0 * (1.0 - alpha))),
    }


def main(args):
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    records = load_records(args.run_dir)
    pairs = []
    skipped = []
    for record in records:
        index = int(record.get("index", len(pairs)))
        clean_path = resolve_image(
            args.run_dir, record, "clean_image", "{:06d}_clean.png".format(index))
        stego_path = resolve_image(
            args.run_dir, record, "stego_image", "{:06d}_stego.png".format(index))
        if clean_path is None or stego_path is None:
            skipped.append({
                "index": index,
                "reason": "missing clean or stego image",
                "clean_image": clean_path,
                "stego_image": stego_path,
            })
            continue
        pairs.append({
            "index": index,
            "prompt": record.get("prompt", ""),
            "record": record,
            "clean_path": clean_path,
            "stego_path": stego_path,
        })

    if not pairs:
        raise RuntimeError("no clean/stego image pairs found in {}".format(
            args.run_dir))

    print("records: {}  pairs: {}  skipped: {}".format(
        len(records), len(pairs), len(skipped)))
    lpips_model = init_lpips(args.lpips_net, device)

    rows = []
    clean_images = []
    stego_images = []
    prompts = []
    for pair in tqdm(pairs, desc="LPIPS/SSIM"):
        record = pair["record"]
        clean = load_rgb(pair["clean_path"])
        stego = load_rgb(pair["stego_path"])
        lpips_value = compute_lpips(lpips_model, clean, stego, device)
        ssim_value = compute_ssim(clean, stego)
        rows.append({
            "index": pair["index"],
            "prompt": pair["prompt"],
            "exact_recovery": record.get("exact_recovery"),
            "ldpc_checks_satisfied": record.get("ldpc_checks_satisfied"),
            "coded_ber": record.get("coded_ber"),
            "ldpc_iterations_used": record.get("ldpc_iterations_used"),
            "llr_margin_mean": record.get("llr_margin_mean"),
            "psnr_recorded": record.get("psnr"),
            "lpips": lpips_value,
            "ssim": ssim_value,
            "clip_clean": None,
            "clip_stego": None,
            "delta_clip": None,
        })
        clean_images.append(clean)
        stego_images.append(stego)
        prompts.append(pair["prompt"])

    clip_backend, clip_model, clip_preprocess, clip_tokenizer = init_clip(
        args.clip_model, args.clip_pretrained, args.clip_checkpoint,
        args.clip_backend, device)
    clean_scores = clip_scores(
        clip_backend, clip_model, clip_preprocess, clip_tokenizer,
        clean_images, prompts, device, args.batch_size)
    stego_scores = clip_scores(
        clip_backend, clip_model, clip_preprocess, clip_tokenizer,
        stego_images, prompts, device, args.batch_size)
    for row, clean_score, stego_score in zip(rows, clean_scores, stego_scores):
        row["clip_clean"] = float(clean_score)
        row["clip_stego"] = float(stego_score)
        row["delta_clip"] = float(stego_score - clean_score)

    output_csv = args.output_csv or os.path.join(args.run_dir, "visual_metrics.csv")
    output_summary = args.output_summary or os.path.join(
        args.run_dir, "visual_metrics_summary.json")
    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "index", "prompt", "exact_recovery", "ldpc_checks_satisfied",
            "coded_ber", "ldpc_iterations_used", "llr_margin_mean",
            "psnr_recorded", "lpips", "ssim", "clip_clean", "clip_stego",
            "delta_clip",
        ])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "run_dir": os.path.abspath(args.run_dir),
        "records": len(records),
        "paired_images": len(rows),
        "skipped": skipped,
        "lpips_net": args.lpips_net,
        "clip_model": args.clip_model,
        "clip_backend": clip_backend,
        "clip_pretrained": args.clip_pretrained,
        "clip_checkpoint": args.clip_checkpoint,
        "metrics": {
            "lpips": finite_stats([row["lpips"] for row in rows]),
            "ssim": finite_stats([row["ssim"] for row in rows]),
            "clip_clean": finite_stats([row["clip_clean"] for row in rows]),
            "clip_stego": finite_stats([row["clip_stego"] for row in rows]),
            "delta_clip": finite_stats([row["delta_clip"] for row in rows]),
        },
    }
    for offset, (name, _) in enumerate(summary["metrics"].items()):
        metric_values = [row[name] for row in rows]
        summary["metrics"][name]["mean_ci95"] = bootstrap_mean_ci(
            metric_values, args.bootstrap_iterations,
            args.bootstrap_seed + offset)
    with open(output_summary, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("wrote: {}".format(output_csv))
    print("wrote: {}".format(output_summary))
    for name, values in summary["metrics"].items():
        print("{}: mean={:.6f} std={:.6f} p05={:.6f} p95={:.6f}".format(
            name, values.get("mean", float("nan")),
            values.get("std", float("nan")), values.get("p05", float("nan")),
            values.get("p95", float("nan"))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="compute LPIPS, SSIM and paired CLIP scores offline")
    parser.add_argument("--run_dir", required=True,
                        help="completed run directory containing results.jsonl and images/")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--lpips_net", default="alex",
                        choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--clip_model", default="ViT-B-32")
    parser.add_argument("--clip_backend", default="auto",
                        choices=["auto", "transformers", "open_clip"])
    parser.add_argument("--clip_pretrained", default=None,
                        help="legacy registry name; never downloaded by this script")
    parser.add_argument("--clip_checkpoint", default=None,
                        help="local OpenCLIP checkpoint (.pt/.pth/.bin)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_summary", default=None)
    parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260917)
    main(parser.parse_args())
