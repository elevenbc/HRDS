"""Re-invert saved clean/stego images for the DCT coefficient diagnostic.

No image generation, training, HF access, or source-run modification. Execute
manually on the GPU server. Results are cached per sample for resumption.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
from pathlib import Path

import numpy as np

from analyze_dct_coefficients import (check_base, dct2, geometry, load_latent,
    load_run, pair_runs, parse_named, qim_observation, read_json, write_json)

SUPPORTED = ("none", "jpeg75", "blur1", "blur2", "brightness020", "gaussian001", "gaussian003")


def apply_attack(image, condition, seed):
    from PIL import Image, ImageEnhance, ImageFilter
    if condition == "none":
        return image.copy(), {}
    if condition == "jpeg75":
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=75)
        buf.seek(0)
        with Image.open(buf) as decoded:
            return decoded.convert("RGB"), {"jpeg_quality": 75}
    if condition in ("blur1", "blur2"):
        radius = int(condition[-1])
        return image.filter(ImageFilter.GaussianBlur(radius=radius)), {"gaussian_blur_radius": radius}
    rng = np.random.default_rng(seed)
    if condition == "brightness020":
        factor = float(rng.uniform(0.8, 1.2))
        return ImageEnhance.Brightness(image).enhance(factor), {"brightness_multiplier": factor}
    if condition in ("gaussian001", "gaussian003"):
        sigma = 0.01 if condition == "gaussian001" else 0.03
        # Keep signed noise in float until clipping. Casting noise to uint8
        # before addition would cause wrapping and is NOT Gaussian corruption.
        x = np.asarray(image, dtype=np.float32) / 255.0
        y = np.clip(x + rng.normal(0.0, sigma, x.shape), 0.0, 1.0)
        return Image.fromarray(np.rint(y * 255).astype(np.uint8)), {"gaussian_std_0_1": sigma}
    raise ValueError("Unsupported condition: %s" % condition)


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--reuse_run", action="append", default=[], help="jpeg75=/path/to/existing/run")
    p.add_argument("--reuse_collection", help="Reuse completed conditions from an earlier collection")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--conditions", default="none,jpeg75,blur1,blur2")
    p.add_argument("--include_clean", action="store_true", help="Add the unembedded-image control")
    p.add_argument("--expected_images", type=int, default=172)
    p.add_argument("--device", default="cuda")
    p.add_argument("--attack_seed", type=int, default=20261026)
    args = p.parse_args()
    conditions = args.conditions.split(",")
    if len(set(conditions)) != len(conditions) or "none" not in conditions or any(c not in SUPPORTED for c in conditions):
        p.error("Unique conditions must include none; supported: " + ",".join(SUPPORTED))
    base = load_run(args.run_dir)
    bp, bm, br = base
    check_base(bm)
    if len(br) != args.expected_images:
        raise ValueError("Expected %s images, found %s" % (args.expected_images, len(br)))
    a = bm["arguments"]
    model_path = Path(a["model_id"])
    if not model_path.is_dir():
        raise ValueError("Local model directory not found: %s" % model_path)
    reuse = {"none": base}
    for label, path in parse_named(args.reuse_run).items():
        if label != "jpeg75":
            raise ValueError("Only exact existing jpeg75 runs can currently be reused")
        other = load_run(path)
        pair_runs(base, other)
        oa = other[1]["arguments"]
        active = {k: oa.get(k) for k in ("jpeg_ratio", "gaussian_blur_r", "gaussian_std", "brightness_factor", "r_degree", "crop_scale", "crop_ratio") if oa.get(k) is not None}
        if active != {"jpeg_ratio": 75} or oa.get("rand_aug", 0):
            raise ValueError("The reused run is not a JPEG75-only run")
        reuse[label] = other
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    arms = ["clean", "stego"] if args.include_clean else ["stego"]
    source_dir = Path(__file__).resolve().parent
    dependencies = ["inverse_stable_diffusion.py", "modified_stable_diffusion.py", "ecc_utils.py", "collect_dct_inversions.py", "analyze_dct_coefficients.py"]
    identity = {"schema": 1, "base_manifest": bm, "conditions": conditions, "arms": arms,
        "attack_seed": args.attack_seed, "reused_runs": {k: str(v[0]) for k, v in reuse.items()},
        "source_sha256": {name: file_digest(source_dir / name) for name in dependencies},
        "inversion": {"model_id": str(model_path), "dtype": "float16", "empty_prompt": True,
            "guidance_scale": 1, "steps": a["test_num_inference_steps"], "vae_sample": False,
            "preprocess": "torchvision Resize(512), CenterCrop(512), ToTensor, 2*x-1"},
        "attack_definitions": {"none": "identity", "jpeg75": "PIL JPEG quality=75",
            "blur1": "PIL GaussianBlur radius=1", "blur2": "PIL GaussianBlur radius=2",
            "brightness020": "PIL brightness factor U[0.8,1.2], paired across arms",
            "gaussian001": "additive float RGB Gaussian noise std=0.01 on [0,1], clip then round",
            "gaussian003": "additive float RGB Gaussian noise std=0.03 on [0,1], clip then round"}}
    prior = None
    if args.reuse_collection:
        prior_path = Path(args.reuse_collection).resolve()
        prior_manifest = read_json(prior_path / "manifest.json")
        for key in ("base_manifest", "attack_seed", "source_sha256", "inversion", "attack_definitions"):
            if prior_manifest["identity"][key] != identity[key]:
                raise ValueError("Incompatible previous collection: %s" % key)
        prior = (prior_path, prior_manifest)
        identity["reuse_collection"] = str(prior_path)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["identity"] != identity:
            raise ValueError("Existing collection has different arguments/sources. Use another out_dir.")
    else:
        manifest = {"identity": identity, "base_manifest": bm, "files": {arm: {c: {} for c in conditions} for arm in arms}}
        write_json(manifest_path, manifest)

    # Import GPU/model dependencies only after all read-only protocol checks.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    import torch
    from torchvision import transforms
    from PIL import Image
    from diffusers import DPMSolverMultistepScheduler
    from inverse_stable_diffusion import InversableStableDiffusionPipeline
    from ecc_utils import LDPCCode
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    missing = sum(1 for arm in arms for condition in conditions for idx in br
                  if not (out / "latents" / arm / condition / ("%06d.npy" % idx)).is_file())
    print("Samples: %s | arms: %s | conditions: %s | uncached records: %s" % (len(br), arms, conditions, missing), flush=True)
    pipe = None
    decoder = None
    transform = transforms.Compose([transforms.Resize(a["image_length"]), transforms.CenterCrop(a["image_length"]), transforms.ToTensor()])
    pos = geometry(dct2(load_latent(bp / br[min(br)]["stego_latent"])).shape, a)[0]
    for arm in arms:
        for condition in conditions:
            dest = out / "latents" / arm / condition
            dest.mkdir(parents=True, exist_ok=True)
            for number, idx in enumerate(sorted(br), 1):
                path = dest / ("%06d.npy" % idx)
                cached_record = manifest["files"][arm][condition].get(str(idx))
                if path.is_file() and cached_record is not None:
                    load_latent(path)  # A corrupted cache must not be silently accepted.
                    continue
                record = {"index": idx, "seed": br[idx]["seed"], "arm": arm, "condition": condition,
                          "latent": str(path.relative_to(out)).replace("\\", "/")}
                prior_record = prior[1]["files"].get(arm, {}).get(condition, {}).get(str(idx)) if prior else None
                if prior_record is not None:
                    latent = load_latent(prior[0] / prior_record["latent"]).astype(np.float32)
                    record.update({k: v for k, v in prior_record.items() if k != "latent"})
                    record["origin"] = "reused_collection"
                elif arm == "stego" and condition in reuse:
                    rp, _, rr = reuse[condition]
                    latent = load_latent(rp / rr[idx]["reversed_latent"]).astype(np.float32)
                    record.update({k: rr[idx][k] for k in ("ber", "coded_ber", "exact_recovery", "ldpc_checks_satisfied")})
                    record["origin"] = "reused_saved_inversion"
                else:
                    recovered = load_latent(path).astype(np.float32) if path.is_file() else None
                    if pipe is None and recovered is None:
                        print("Loading local model for inversion only:", model_path, flush=True)
                        scheduler = DPMSolverMultistepScheduler.from_pretrained(str(model_path), subfolder="scheduler", local_files_only=True)
                        pipe = InversableStableDiffusionPipeline.from_pretrained(str(model_path), scheduler=scheduler,
                            torch_dtype=torch.float16, revision="fp16", local_files_only=True).to(args.device)
                        pipe.set_progress_bar_config(disable=True)
                        text_embedding = pipe.get_text_embedding("")
                        # The first replay verifies consistency with the historical
                        # no-attack inversion before collecting any new observation.
                        first_idx = min(br)
                        with Image.open(bp / br[first_idx]["stego_image"]) as im:
                            control = 2.0 * transform(im.convert("RGB")) - 1.0
                        with torch.no_grad():
                            z0 = pipe.get_image_latents(control.unsqueeze(0).to(args.device, dtype=text_embedding.dtype), sample=False)
                            check = pipe.forward_diffusion(latents=z0, text_embeddings=text_embedding, guidance_scale=1,
                                num_inference_steps=a["test_num_inference_steps"]).detach().float().cpu().numpy()
                        saved = load_latent(bp / br[first_idx]["reversed_latent"])
                        replay_rmse = float(np.sqrt(np.mean((check - saved) ** 2)))
                        if not np.allclose(check, saved, rtol=0.001, atol=0.002):
                            raise RuntimeError("No-attack replay differs from the saved run (RMSE %.6g). Check model/inversion versions before mixing observations." % replay_rmse)
                        print("No-attack replay RMSE:", replay_rmse, flush=True)
                        manifest["replay_validation"] = {"index": first_idx, "rmse": replay_rmse,
                            "atol": 0.002, "rtol": 0.001}
                    # Identical random perturbations across clean/stego arms.
                    seed = int((args.attack_seed + 1000003 * idx + 1009 * SUPPORTED.index(condition)) % (2 ** 32))
                    with Image.open(bp / br[idx][arm + "_image"]) as im:
                        received, actual = apply_attack(im.convert("RGB"), condition, seed)
                    if recovered is not None:
                        latent = recovered
                    else:
                        x = (2.0 * transform(received) - 1.0).unsqueeze(0).to(args.device, dtype=text_embedding.dtype)
                        with torch.no_grad():
                            z = pipe.get_image_latents(x, sample=False)
                            latent = pipe.forward_diffusion(latents=z, text_embeddings=text_embedding, guidance_scale=1,
                                num_inference_steps=a["test_num_inference_steps"]).detach().float().cpu().numpy()
                    record.update({"origin": "new_inversion_of_saved_image", "attack_seed": seed, "attack_parameters": actual})
                    if arm == "stego":
                        q = qim_observation(dct2(latent.astype(np.float64)), pos, a, br[idx])
                        if decoder is None:
                            decoder = LDPCCode.from_rate(n=a["stega_n_bits"], rate=a["ldpc_rate"], seed=a["ldpc_seed"], column_weight=a["ldpc_column_weight"])
                        decoded = decoder.decode(q["hard"], q["llr"], max_iterations=a["ldpc_iterations"], scaling=a["ldpc_scaling"])
                        errors = int(np.sum(np.asarray(decoded["secret_bits"]) != np.asarray(br[idx]["secret_bits"])))
                        record.update({"ber": errors / float(len(br[idx]["secret_bits"])), "coded_ber": q["coded_ber"],
                            "exact_recovery": errors == 0, "ldpc_checks_satisfied": bool(decoded["checks_satisfied"])})
                if not np.isfinite(latent).all():
                    raise RuntimeError("Nonfinite inversion: %s/%s/%s" % (arm, condition, idx))
                # Replace only task-owned temporary files; never touch source runs.
                temp = path.with_suffix(".tmp.npy")
                np.save(str(temp), latent.astype(np.float32))
                temp.replace(path)
                manifest["files"][arm][condition][str(idx)] = record
                temp_manifest = manifest_path.with_suffix(".tmp.json")
                write_json(temp_manifest, manifest)
                temp_manifest.replace(manifest_path)
                print("%s/%s %s/%s (index %s)" % (arm, condition, number, len(br), idx), flush=True)
    print("COLLECTION:", manifest_path, flush=True)


if __name__ == "__main__":
    main()
