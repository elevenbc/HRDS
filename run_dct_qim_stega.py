"""End-to-end DCT-QIM steganography experiments with durable run records.

Pipeline:
    initial latent -> 2-D DCT -> dithered-QIM -> IDCT
    -> Stable Diffusion generation -> optional attack
    -> VAE encode -> empty-prompt DDIM inversion -> DCT -> QIM extraction

The runner writes a timestamped, self-contained audit directory containing:
    manifest.json       experiment configuration and source hashes
    results.jsonl       one durable record per generated image
    summary.json        aggregate BER / recovery / quality statistics
    images/             optional clean, stego, and received images
    latents/            optional initial and inverted latents
    llrs/               optional per-carrier and per-message LLRs

Example:
    python run_dct_qim_stega.py \
        --model_id /path/to/stable-diffusion-2-1-base \
        --prompt_file prompts.txt \
        --start 0 --end 20 \
        --stega_n_bits 283 --stega_repeat 11 --qim_step 3.0 \
        --coding_mode ldpc --ldpc_rate 0.6666666666666666 \
        --save_clean_images --save_llrs --save_latents \
        --run_name ldpc_283_rate_2_3
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from optim_utils import (
    set_random_seed,
    transform_img,
    image_distortion,
    get_dataset,
)
from qim_utils import (
    count_carriers,
    embed_message_in_latents,
    extract_message_details_from_latents,
)
from ecc_utils import LDPCCode


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_record_dir(args: argparse.Namespace, dataset: List[Any]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_dir) / f"{args.run_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    if args.save_images or args.save_clean_images:
        (run_dir / "images").mkdir()
    if args.save_latents:
        (run_dir / "latents").mkdir()
    if args.save_llrs:
        (run_dir / "llrs").mkdir()

    prompt_text = "\n".join(
        item if isinstance(item, str) else str(item) for item in dataset
    )
    manifest: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "torch": getattr(torch, "__version__", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "arguments": vars(args),
        "prompt_source": args.prompt_file or args.dataset,
        "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "prompt_count": len(dataset),
        "source_sha256": {
            name: _sha256_file(name)
            for name in ("run_dct_qim_stega.py", "qim_utils.py", "ecc_utils.py")
            if Path(name).is_file()
        },
        "record_format": (
            "results.jsonl: one durable record per generated image; "
            "summary.json: aggregate metrics"
        ),
    }
    _write_json(run_dir / "manifest.json", manifest)
    return run_dir


def _psnr(clean: Any, stego: Any, max_val: float = 255.0) -> float:
    a = np.asarray(clean, dtype=np.float64)
    b = np.asarray(stego, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("clean and stego images must have the same shape")
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


def _latent_metrics(clean_latent: torch.Tensor, stego_latent: torch.Tensor) -> Dict[str, float]:
    clean = clean_latent.detach().float().cpu().numpy()
    stego = stego_latent.detach().float().cpu().numpy()
    delta = stego - clean
    mse = float(np.mean(delta ** 2))
    l2 = float(np.linalg.norm(delta))
    return {
        "latent_mse": mse,
        "latent_l2": l2,
        "latent_rmse": math.sqrt(mse),
        "latent_relative_l2": l2 / max(float(np.linalg.norm(clean)), 1e-12),
    }


def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        args.model_id, subfolder="scheduler"
    )
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision="fp16",
    ).to(device)

    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            dataset = [line.strip() for line in f if line.strip()]
        prompt_key = None
        if not dataset:
            raise ValueError("prompt_file is empty")
    else:
        dataset, prompt_key = get_dataset(args)

    run_dir = _prepare_record_dir(args, dataset)
    results_path = run_dir / "results.jsonl"
    print(f"[record] writing immutable experiment record to {run_dir}")

    text_embeddings = pipe.get_text_embedding("")

    latent_size = args.image_length // 8
    available_carriers = count_carriers(
        channels=4,
        size=latent_size,
        min_radius=args.qim_min_radius,
        max_radius=args.qim_max_radius,
    )
    used_carriers = args.stega_n_bits * args.stega_repeat
    capacity_utilization = used_carriers / available_carriers
    print(
        f"[capacity] available_carriers={available_carriers}, "
        f"used_carriers={used_carriers}, "
        f"utilization={capacity_utilization:.4%}"
    )
    if used_carriers > available_carriers:
        raise ValueError(
            "requested payload exceeds carrier capacity: "
            f"used={used_carriers}, available={available_carriers}"
        )

    ldpc_code = None
    secret_length = args.stega_n_bits
    if args.coding_mode == "ldpc":
        ldpc_code = LDPCCode.from_rate(
            n=args.stega_n_bits,
            rate=args.ldpc_rate,
            seed=args.ldpc_seed,
            column_weight=args.ldpc_column_weight,
        )
        secret_length = ldpc_code.k
        print(
            f"[ldpc] n={ldpc_code.n}, k={ldpc_code.k}, "
            f"m={ldpc_code.m}, rate={ldpc_code.code_rate:.6f}, "
            f"seed={args.ldpc_seed}, column_weight={args.ldpc_column_weight}"
        )

    # Phase 0: lossless sanity check before spending diffusion GPU time.
    probe = torch.randn(1, 4, latent_size, latent_size)
    probe_msg = np.random.default_rng(0).integers(0, 2, secret_length)
    if ldpc_code is not None:
        probe_coded = ldpc_code.encode(probe_msg)
    else:
        probe_coded = probe_msg

    probe_w = embed_message_in_latents(
        probe,
        probe_coded,
        args.stega_key,
        step=args.qim_step,
        repeat=args.stega_repeat,
        min_radius=args.qim_min_radius,
        max_radius=args.qim_max_radius,
        embedding_mode=args.qim_embedding_mode,
        rounding_seed=args.qim_rounding_seed,
        step_mode=args.qim_step_mode,
        step_min_scale=args.qim_step_min_scale,
        step_max_scale=args.qim_step_max_scale,
        cdf_step=args.qim_cdf_step,
        cdf_mean=args.qim_cdf_mean,
        cdf_std=args.qim_cdf_std,
        cdf_clip=args.qim_cdf_clip,
    )
    probe_details = extract_message_details_from_latents(
        probe_w,
        args.stega_n_bits,
        args.stega_key,
        step=args.qim_step,
        repeat=args.stega_repeat,
        min_radius=args.qim_min_radius,
        max_radius=args.qim_max_radius,
        step_mode=args.qim_step_mode,
        step_min_scale=args.qim_step_min_scale,
        step_max_scale=args.qim_step_max_scale,
        embedding_mode=args.qim_embedding_mode,
        cdf_step=args.qim_cdf_step,
        cdf_mean=args.qim_cdf_mean,
        cdf_std=args.qim_cdf_std,
        cdf_clip=args.qim_cdf_clip,
    )
    if ldpc_code is not None:
        probe_ldpc = ldpc_code.decode(
            probe_details["soft_bits"],
            probe_details["message_llrs"],
            max_iterations=args.ldpc_iterations,
            scaling=args.ldpc_scaling,
        )
        probe_vote_ber = float(
            (probe_ldpc["secret_bits"] != probe_msg).mean()
        )
        probe_soft_ber = probe_vote_ber
    else:
        probe_vote_ber = float((probe_details["vote_bits"] != probe_msg).mean())
        probe_soft_ber = float((probe_details["soft_bits"] != probe_msg).mean())
    print(
        f"[sanity] no-diffusion BER: hard={probe_vote_ber:.4f}, "
        f"soft={probe_soft_ber:.4f} (expected 0.0)"
    )
    if probe_vote_ber > 0.0 or probe_soft_ber > 0.0:
        raise RuntimeError("QIM self-consistency check failed; aborting diffusion run")

    message_rng = np.random.default_rng(args.stega_msg_seed)
    results: List[Dict[str, Any]] = []

    with results_path.open("w", encoding="utf-8") as record_file:
        for i in tqdm(range(args.start, args.end), desc="DCT-QIM steganography"):
            seed = i + args.gen_seed
            current_prompt = (
                dataset[i % len(dataset)]
                if prompt_key is None
                else dataset[i][prompt_key]
            )

            set_random_seed(seed)
            init_latents_no_w = pipe.get_random_latents()

            message = message_rng.integers(0, 2, secret_length)
            if ldpc_code is not None:
                coded_message = ldpc_code.encode(message)
            else:
                coded_message = message

            init_latents_w = embed_message_in_latents(
                copy.deepcopy(init_latents_no_w).float(),
                coded_message,
                args.stega_key,
                step=args.qim_step,
                repeat=args.stega_repeat,
                min_radius=args.qim_min_radius,
                max_radius=args.qim_max_radius,
                embedding_mode=args.qim_embedding_mode,
                rounding_seed=args.qim_rounding_seed + seed,
                step_mode=args.qim_step_mode,
                step_min_scale=args.qim_step_min_scale,
                step_max_scale=args.qim_step_max_scale,
                cdf_step=args.qim_cdf_step,
                cdf_mean=args.qim_cdf_mean,
                cdf_std=args.qim_cdf_std,
                cdf_clip=args.qim_cdf_clip,
            ).to(init_latents_no_w.dtype)

            clean_image = None
            if args.save_clean_images or args.compute_image_metrics:
                outputs_clean = pipe(
                    current_prompt,
                    num_images_per_prompt=args.num_images,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    height=args.image_length,
                    width=args.image_length,
                    latents=init_latents_no_w,
                )
                clean_image = outputs_clean.images[0]

            outputs_w = pipe(
                current_prompt,
                num_images_per_prompt=args.num_images,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=args.image_length,
                width=args.image_length,
                latents=init_latents_w,
            )
            stego_image = outputs_w.images[0]

            _, stego_aug = image_distortion(stego_image, stego_image, seed, args)

            img_t = transform_img(stego_aug).unsqueeze(0).to(
                text_embeddings.dtype
            ).to(device)
            image_latents = pipe.get_image_latents(img_t, sample=False)
            reversed_latents = pipe.forward_diffusion(
                latents=image_latents,
                text_embeddings=text_embeddings,
                guidance_scale=1,
                num_inference_steps=args.test_num_inference_steps,
            )

            details = extract_message_details_from_latents(
                reversed_latents.float().cpu(),
                args.stega_n_bits,
                args.stega_key,
                step=args.qim_step,
                repeat=args.stega_repeat,
                min_radius=args.qim_min_radius,
                max_radius=args.qim_max_radius,
                step_mode=args.qim_step_mode,
                step_min_scale=args.qim_step_min_scale,
                step_max_scale=args.qim_step_max_scale,
                embedding_mode=args.qim_embedding_mode,
                cdf_step=args.qim_cdf_step,
                cdf_mean=args.qim_cdf_mean,
                cdf_std=args.qim_cdf_std,
                cdf_clip=args.qim_cdf_clip,
            )

            coded_hard = details["soft_bits"]
            coded_llrs = details["message_llrs"]

            if args.coding_mode == "hard_repetition":
                decoded = details["vote_bits"]
            elif args.coding_mode == "soft_repetition":
                decoded = details["soft_bits"]
            elif args.coding_mode == "ldpc":
                ldpc_result = ldpc_code.decode(
                    coded_hard,
                    coded_llrs,
                    max_iterations=args.ldpc_iterations,
                    scaling=args.ldpc_scaling,
                )
                decoded = ldpc_result["secret_bits"]
            else:
                raise ValueError(f"unsupported coding_mode: {args.coding_mode}")

            bit_errors = int((decoded != message).sum())
            ber = bit_errors / secret_length
            coded_bit_errors = int((coded_hard != coded_message).sum())
            coded_ber = coded_bit_errors / args.stega_n_bits
            latent_stats = _latent_metrics(init_latents_no_w, init_latents_w)

            item: Dict[str, Any] = {
                "index": i,
                "seed": seed,
                "prompt": current_prompt,
                "secret_bits": message.tolist(),
                "message_bits": message.tolist(),
                "coded_bits": coded_message.tolist(),
                "decoded_bits": decoded.tolist(),
                "decoded_coded_bits": coded_hard.tolist(),
                "bit_errors": bit_errors,
                "ber": ber,
                "coded_bit_errors": coded_bit_errors,
                "coded_ber": coded_ber,
                "exact_recovery": bit_errors == 0,
                "coding_mode": args.coding_mode,
                **latent_stats,
                "llr_abs_mean": float(np.mean(np.abs(details["carrier_llrs"]))),
                "llr_margin_mean": float(
                    np.mean(np.abs(details["message_llrs"]))
                ),
            }

            if ldpc_code is not None:
                item.update(
                    {
                        "ldpc_iterations_used": int(
                            ldpc_result["iterations_used"]
                        ),
                        "ldpc_checks_satisfied": bool(
                            ldpc_result["checks_satisfied"]
                        ),
                        "ldpc_coded_errors_after_decode": int(
                            np.sum(ldpc_result["coded_bits"] != coded_message)
                        ),
                    }
                )

            if clean_image is not None and args.compute_image_metrics:
                item["psnr"] = _psnr(clean_image, stego_image)

            if args.save_clean_images and clean_image is not None:
                clean_name = f"{i:06d}_clean.png"
                clean_image.save(run_dir / "images" / clean_name)
                item["clean_image"] = f"images/{clean_name}"

            if args.save_images:
                stego_name = f"{i:06d}_stego.png"
                received_name = f"{i:06d}_received.png"
                stego_image.save(run_dir / "images" / stego_name)
                stego_aug.save(run_dir / "images" / received_name)
                item["stego_image"] = f"images/{stego_name}"
                item["received_image"] = f"images/{received_name}"

            if args.save_latents:
                clean_latent_path = run_dir / "latents" / f"{i:06d}_clean_init.npy"
                stego_latent_path = run_dir / "latents" / f"{i:06d}_stego_init.npy"
                reversed_latent_path = run_dir / "latents" / f"{i:06d}_reversed.npy"
                np.save(clean_latent_path, init_latents_no_w.detach().float().cpu().numpy())
                np.save(stego_latent_path, init_latents_w.detach().float().cpu().numpy())
                np.save(
                    reversed_latent_path,
                    reversed_latents.detach().float().cpu().numpy(),
                )
                item.update(
                    {
                        "clean_latent": f"latents/{clean_latent_path.name}",
                        "stego_latent": f"latents/{stego_latent_path.name}",
                        "reversed_latent": f"latents/{reversed_latent_path.name}",
                    }
                )

            if args.save_llrs:
                carrier_path = run_dir / "llrs" / f"{i:06d}_carrier_llrs.npy"
                message_path = run_dir / "llrs" / f"{i:06d}_message_llrs.npy"
                np.save(carrier_path, details["carrier_llrs"])
                np.save(message_path, details["message_llrs"])
                item.update(
                    {
                        "carrier_llrs": f"llrs/{carrier_path.name}",
                        "message_llrs": f"llrs/{message_path.name}",
                    }
                )

            record_file.write(json.dumps(item, ensure_ascii=False) + "\n")
            record_file.flush()
            results.append(
                {
                    "i": i,
                    "ber": ber,
                    "coded_ber": coded_ber,
                    "ok": bit_errors == 0,
                    "latent_mse": latent_stats["latent_mse"],
                    "psnr": item.get("psnr"),
                    "llr_abs_mean": item["llr_abs_mean"],
                }
            )
            tqdm.write(
                f"[{i}] secret BER={ber:.4f} "
                f"({bit_errors}/{secret_length}), "
                f"coded BER={coded_ber:.4f} "
                f"({coded_bit_errors}/{args.stega_n_bits}) "
                f"exact={bit_errors == 0}"
            )

    bers = np.asarray([r["ber"] for r in results], dtype=np.float64)
    coded_bers = np.asarray(
        [r["coded_ber"] for r in results], dtype=np.float64
    )
    exact = np.asarray([r["ok"] for r in results], dtype=bool)
    latent_mses = np.asarray([r["latent_mse"] for r in results], dtype=np.float64)
    llr_abs = np.asarray([r["llr_abs_mean"] for r in results], dtype=np.float64)
    psnrs = np.asarray(
        [r["psnr"] for r in results if r["psnr"] is not None], dtype=np.float64
    )

    summary: Dict[str, Any] = {
        "images": len(results),
        "payload_bits": secret_length,
        "secret_bits": secret_length,
        "coded_bits": args.stega_n_bits,
        "available_carriers": available_carriers,
        "used_carriers": used_carriers,
        "capacity_utilization": capacity_utilization,
        "coding_mode": args.coding_mode,
        "code_rate": (
            ldpc_code.code_rate if ldpc_code is not None else 1.0
        ),
        "mean_ber": float(bers.mean()) if len(bers) else None,
        "mean_secret_ber": float(bers.mean()) if len(bers) else None,
        "mean_coded_ber": float(coded_bers.mean()) if len(coded_bers) else None,
        "std_ber": float(bers.std()) if len(bers) else None,
        "exact_recovery_count": int(exact.sum()),
        "exact_recovery_rate": float(exact.mean()) if len(exact) else None,
        "mean_latent_mse": float(latent_mses.mean()) if len(latent_mses) else None,
        "mean_llr_abs": float(llr_abs.mean()) if len(llr_abs) else None,
        "mean_psnr": float(psnrs.mean()) if len(psnrs) else None,
        "std_psnr": float(psnrs.std()) if len(psnrs) else None,
        "attack": {
            "jpeg_ratio": args.jpeg_ratio,
            "r_degree": args.r_degree,
            "crop_scale": args.crop_scale,
            "crop_ratio": args.crop_ratio,
            "gaussian_blur_r": args.gaussian_blur_r,
            "gaussian_std": args.gaussian_std,
            "brightness_factor": args.brightness_factor,
        },
    }
    if ldpc_code is not None:
        summary["ldpc"] = {
            "n": ldpc_code.n,
            "k": ldpc_code.k,
            "m": ldpc_code.m,
            "rate": ldpc_code.code_rate,
            "seed": args.ldpc_seed,
            "column_weight": args.ldpc_column_weight,
            "max_iterations": args.ldpc_iterations,
            "scaling": args.ldpc_scaling,
        }
    _write_json(run_dir / "summary.json", summary)

    print("================ summary ================")
    print(f"images          : {len(results)}")
    print(
        f"payload         : {secret_length} secret bits "
        f"({args.stega_n_bits} coded bits) "
        f"(carriers={used_carriers}/{available_carriers}, "
        f"utilization={capacity_utilization:.2%})"
    )
    if ldpc_code is not None:
        print(
            f"LDPC            : n={ldpc_code.n}, k={ldpc_code.k}, "
            f"rate={ldpc_code.code_rate:.6f}"
        )
    print(f"mean BER        : {bers.mean():.4f}" if len(bers) else "mean BER        : n/a")
    print(
        f"exact recovery  : {int(exact.sum())}/{len(exact)} "
        f"= {exact.mean():.2%}"
        if len(exact)
        else "exact recovery  : n/a"
    )
    if len(psnrs):
        print(f"mean PSNR       : {psnrs.mean():.2f} ± {psnrs.std():.2f} dB")
    print(f"record directory: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCT-QIM diffusion steganography")
    parser.add_argument("--run_name", default="stega")
    parser.add_argument("--output_dir", default="outputs/dct_qim_stega")
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--save_clean_images", action="store_true")
    parser.add_argument("--save_latents", action="store_true")
    parser.add_argument("--save_llrs", action="store_true")
    parser.add_argument("--compute_image_metrics", action="store_true")
    parser.add_argument(
        "--coding_mode",
        choices=("hard_repetition", "soft_repetition", "ldpc"),
        default="hard_repetition",
    )
    parser.add_argument(
        "--ldpc_rate",
        type=float,
        default=2.0 / 3.0,
        help="LDPC secret/coded bit rate",
    )
    parser.add_argument(
        "--ldpc_seed",
        type=int,
        default=20260826,
        help="deterministic LDPC construction seed",
    )
    parser.add_argument(
        "--ldpc_column_weight",
        type=int,
        default=3,
        help="number of checks connected to each coded bit",
    )
    parser.add_argument(
        "--ldpc_iterations",
        type=int,
        default=100,
        help="maximum min-sum belief-propagation iterations",
    )
    parser.add_argument(
        "--ldpc_scaling",
        type=float,
        default=0.80,
        help="normalized min-sum scaling factor",
    )

    parser.add_argument("--dataset", default="Gustavosta/Stable-Diffusion-Prompts")
    parser.add_argument("--prompt_file", default="", type=str)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=10, type=int)
    parser.add_argument("--image_length", default=512, type=int)
    parser.add_argument(
        "--model_id", default="stabilityai/stable-diffusion-2-1-base"
    )
    parser.add_argument("--num_images", default=1, type=int)
    parser.add_argument("--guidance_scale", default=7.5, type=float)
    parser.add_argument("--num_inference_steps", default=50, type=int)
    parser.add_argument("--test_num_inference_steps", default=None, type=int)
    parser.add_argument("--gen_seed", default=0, type=int)

    parser.add_argument("--stega_key", default=999999, type=int)
    parser.add_argument("--stega_msg_seed", default=2024, type=int)
    parser.add_argument("--stega_n_bits", default=8, type=int)
    parser.add_argument("--stega_repeat", default=11, type=int)
    parser.add_argument("--qim_step", default=0.8, type=float)
    parser.add_argument("--qim_min_radius", default=8.0, type=float)
    parser.add_argument("--qim_max_radius", default=26.0, type=float)
    parser.add_argument(
        "--qim_embedding_mode",
        choices=("nearest", "stochastic", "cdf"),
        default="nearest",
        help=(
            "nearest uses the original QIM; stochastic uses unbiased "
            "randomized rounding; cdf quantizes in Gaussian CDF space"
        ),
    )
    parser.add_argument(
        "--qim_rounding_seed",
        default=20260928,
        type=int,
        help="base RNG seed used only by stochastic QIM",
    )
    parser.add_argument(
        "--qim_step_mode",
        choices=("global", "radial"),
        default="global",
        help="global step or deterministic radial per-carrier steps",
    )
    parser.add_argument("--qim_step_min_scale", default=0.75, type=float)
    parser.add_argument("--qim_step_max_scale", default=1.25, type=float)
    parser.add_argument("--qim_cdf_step", default=0.08, type=float)
    parser.add_argument("--qim_cdf_mean", default=0.0, type=float)
    parser.add_argument("--qim_cdf_std", default=1.0, type=float)
    parser.add_argument("--qim_cdf_clip", default=1.0e-4, type=float)

    parser.add_argument("--r_degree", default=None, type=float)
    parser.add_argument("--jpeg_ratio", default=None, type=int)
    parser.add_argument("--crop_scale", default=None, type=float)
    parser.add_argument("--crop_ratio", default=None, type=float)
    parser.add_argument("--gaussian_blur_r", default=None, type=int)
    parser.add_argument("--gaussian_std", default=None, type=float)
    parser.add_argument("--brightness_factor", default=None, type=float)
    parser.add_argument("--rand_aug", default=0, type=int)

    args = parser.parse_args()
    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps
    main(args)
