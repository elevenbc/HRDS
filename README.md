# High-Capacity Robust Diffusion Steganography via Quantization Index Modulation

Research implementation of bit-message steganography in the initial latent of a diffusion model. The sender applies binary dithered QIM to selected DCT coefficients. The receiver inverts the received image, combines repeated soft observations, and decodes the message with LDPC.

## Contents

| File | Purpose |
| --- | --- |
| `run_paper.py` | Portable entry point for the paper configurations |
| `run_dct_qim_stega.py` | Generation, optional image attack, inversion, extraction and run records |
| `qim_utils.py`, `ecc_utils.py` | DCT/QIM and LDPC implementation |
| `inverse_stable_diffusion.py`, `modified_stable_diffusion.py`, `optim_utils.py` | Diffusion pipeline and image processing utilities |
| `analyze_visual_quality.py` | LPIPS, SSIM, CLIP cosine similarity and paired CLIP change |
| `prepare_fid_dirs.py`, `calculate_fid_score.py`, `inception.py` | Separate image sets and compute FID |
| `collect_dct_inversions.py`, `analyze_dct_coefficients.py` | DCT coefficient error distributions and recovery statistics |
| `analyze_ldpc_input_ablation.py`, `analyze_repetition_input_ablation.py` | Receiver input ablations |
| `analyze_fc27_run_statistics.py` | Run-level statistics and uncertainty (historical filename) |
| `analyze_treering_dataset.py` | Paired quality evaluation of an existing watermark baseline dataset |
| `analyze_steganalysis_scores.py` | Evaluation of externally supplied detector scores |

The package contains source and prompt lists. Obtain model weights separately. Detector training and the external RingID-style baseline generator are not included. Further experimental branches and manuscript-editing scripts are outside this release.

## Environment

The recorded main experiments used Linux, Python 3.7.16, PyTorch 1.13.0+cu117 and an NVIDIA RTX 4090. The mounted project specifies Transformers 4.23.1 and Diffusers 0.11.1. These legacy pipelines use APIs that changed in recent Diffusers releases.

```bash
conda create -n dctqim python=3.7.16 -y
conda activate dctqim
python -m pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
```

The remaining package pins are compatibility constraints reconstructed for this release, not an archived full environment lock. End-to-end generation must be validated on the target GPU environment. No GPU experiment was run during packaging.

Prepare a local Diffusers model directory containing `model_index.json`, UNet, VAE, tokenizer, text encoder and scheduler files. Use the corresponding SD 1.4, SD 1.5, SD 2.0-base or SD 2.1-base checkpoint. Use the **base 512-pixel** checkpoint when comparing the SD 2.x configurations.

## Generate and extract

From this repository root:

```bash
export MODEL_DIR=/path/to/stable-diffusion-2-1-base
CUDA_VISIBLE_DEVICES=0 python -u run_paper.py --model "$MODEL_DIR" --images 5
CUDA_VISIBLE_DEVICES=0 python -u run_paper.py --model "$MODEL_DIR" --images 172 --preset E5 --attack none
CUDA_VISIBLE_DEVICES=0 python -u run_paper.py --model "$MODEL_DIR" --images 172 --preset E5 --attack jpeg75
```

Add `--dry_run` to print a command without loading models. To repeat on another model, change `--model` and `--model_tag sd14` (or `sd15`, `sd20base`); keep prompts, capacity and generation seed fixed.

| Preset | Coded bits n | Message bits k | Repeats r | Used DCT carriers n × r |
| --- | ---: | ---: | ---: | ---: |
| k65 | 98 | 65 | 5 | 490 |
| E2 | 283 | 188 | 11 | 3113 |
| E4 | 445 | 296 | 7 | 3115 |
| E5 | 623 | 415 | 5 | 3115 |
| E6 | 779 | 519 | 4 | 3116 |
| E7 | 1038 | 692 | 3 | 3114 |

All presets use DCT radii 8–32, QIM step 3.0, LDPC rate 2/3, column weight 3, 100 decoder iterations, scaling 0.8, 50 generation/inversion steps, and 512 × 512 images. `n` is the coded length; `k=floor(2n/3)` is the message capacity. Available carriers: 3116.

The wrapper defaults to the 172-prompt main evaluation and generation seed 1500. The historical 20-image preliminary sweep used a different 10-prompt list cycled over 20 images and seed 0; using the main prompt list does not reproduce that historical sweep. Additional QIM variants in the core are experimental; the paper configuration is `nearest` with `global` steps.

Supported wrapper conditions: `none`, `jpeg95`, `jpeg85`, `jpeg75`, `blur1`, `blur2`, `brightness010`, `brightness020`. Brightness values specify ColorJitter amplitudes, not fixed multipliers. Run conditions sequentially; the historical JPEG implementation uses temporary files keyed by run name. Additional attacks are exposed by the original runner, but are not all validated paper configurations.

Each run prints a timestamped directory with `manifest.json`, `results.jsonl`, `summary.json`, `images/`, `latents/` and `llrs/`. Keep this directory intact. Use exact message recovery and message BER alongside coded BER; parity-check satisfaction alone is not exact recovery.

## Image quality without regenerating images

Use the completed run directory printed above. A local Transformers CLIP directory is recommended; paper comparisons used CLIP ViT-L/14, not the default OpenCLIP model-name label. LPIPS requires its calibration weights and cached torchvision AlexNet weights; prepare these before using a disconnected server.

```bash
export RUN_DIR=outputs/dct_qim/REPLACE_WITH_COMPLETED_RUN
export CLIP_DIR=/path/to/clip-vit-large-patch14
python -u analyze_visual_quality.py --run_dir "$RUN_DIR" --device cuda --lpips_net alex --clip_backend transformers --clip_checkpoint "$CLIP_DIR" --batch_size 16
```

The comparison is **clean versus stego before channel attacks**. CLIP scores are unscaled normalized embedding cosine similarities; delta CLIP is stego minus clean. The actual model is determined by `clip_backend` and `clip_checkpoint`, not the legacy `clip_model` label.

For FID, place the local `pt_inception-2015-12-05-6726825d.pth` in the repository root (a symlink is sufficient). Expected SHA-256: `6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2`.

```bash
ln -s /path/to/pt_inception-2015-12-05-6726825d.pth ./pt_inception-2015-12-05-6726825d.pth
export FID_DIR=outputs/fid_sets/my_run
python prepare_fid_dirs.py --run_dir "$RUN_DIR" --out_dir "$FID_DIR" --include_received
python -u calculate_fid_score.py --device cuda --batch-size 4 --dims 2048 "$FID_DIR/clean" "$FID_DIR/stego"
python -u calculate_fid_score.py --device cuda --batch-size 4 --dims 2048 "$FID_DIR/clean" "$FID_DIR/received"
```

Choose a fresh FID directory for each run. FID is a collection-level comparison; use the same sample count and checkpoint across methods. The second comparison includes channel distortion. Do not mix clean, stego and received images in one FID input directory.

## DCT coefficient diagnostics and ablations

`BASE_RUN` must be a completed no-attack run with saved clean/stego images and latents. The following collection executes additional inversions on the selected GPU.

```bash
export BASE_RUN=outputs/dct_qim/REPLACE_WITH_NOATTACK_RUN
python -u collect_dct_inversions.py --run_dir "$BASE_RUN" --out_dir outputs/dct_collection --conditions none,jpeg75,blur1,blur2 --include_clean --expected_images 172
python analyze_dct_coefficients.py --run_dir "$BASE_RUN" --collection_dir outputs/dct_collection --out_dir outputs/dct_analysis --expected_images 172
python analyze_ldpc_input_ablation.py --run_dir "$RUN_DIR" --hard_magnitude 1.0
python analyze_fc27_run_statistics.py --run main="$RUN_DIR" --output_dir outputs/statistics
```

The inversion collector uses the model path recorded in the original run manifest. If moving runs between machines, ensure that path remains accessible. Coefficient analysis provides SVG/CSV/JSON even when Matplotlib is unavailable. Other evaluation scripts expose their input schemas with `--help`.

## Data and provenance

`prompts/paper172.txt` is the fixed 172-prompt evaluation set retained from earlier screening, not an independent random sample. `prompts/five_styles.txt` is for five illustrative examples. Preserve order and seed for paired model comparisons. The prompt source SHA in a run manifest hashes parsed prompt text and can differ from the byte-level file hash.

The three core files were copied byte-for-byte from the mounted experiment project and matched the recorded source hashes of the SD 2.1-base main run dated 2026-10-12. `SOURCE_MANIFEST.json` records packaged file hashes, origins and this match. Auxiliary evaluation files reflect the local working versions at packaging; no matching historical hash is claimed for them. Run `python verify_release.py` to check the package and `python test_codec.py` for a CPU-only codec check.

## Attribution and licensing

See `THIRD_PARTY_NOTICES.md` and `licenses/` for upstream code notices. The project authors have not selected a blanket license for their original additions in this package; existing third-party licenses retain their scope. Model weights and their licenses are managed separately. This repository is prepared for upload; it does not imply publication acceptance.

For Chinese instructions and upload steps, see [README_zh.md](README_zh.md).
