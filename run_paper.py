"""Portable paper configuration launcher; use --dry_run to inspect commands."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PRESETS = {'k65': (98, 5), 'E2': (283, 11), 'E4': (445, 7),
           'E5': (623, 5), 'E6': (779, 4), 'E7': (1038, 3)}
ATTACKS = {'none': [], 'jpeg95': ['--jpeg_ratio', '95'],
           'jpeg85': ['--jpeg_ratio', '85'], 'jpeg75': ['--jpeg_ratio', '75'],
           'blur1': ['--gaussian_blur_r', '1'], 'blur2': ['--gaussian_blur_r', '2'],
           'brightness010': ['--brightness_factor', '0.10'],
           'brightness020': ['--brightness_factor', '0.20']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, help='Local Diffusers model directory')
    p.add_argument('--model_tag', default='sd21base')
    p.add_argument('--preset', choices=PRESETS, default='E5')
    p.add_argument('--attack', choices=ATTACKS, default='none')
    p.add_argument('--prompt_file', default=str(ROOT / 'prompts/paper172.txt'))
    p.add_argument('--images', type=int, default=172)
    p.add_argument('--seed', type=int, default=1500)
    p.add_argument('--output_dir', default=str(ROOT / 'outputs/dct_qim'))
    p.add_argument('--dry_run', action='store_true')
    a = p.parse_args()
    prompts = Path(a.prompt_file).resolve()
    lines = [x.strip() for x in prompts.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    if not 1 <= a.images <= len(lines):
        p.error('--images must be between 1 and the number of prompt lines')
    model = Path(a.model).resolve()
    if not a.dry_run and not (model / 'model_index.json').is_file():
        p.error('--model must contain model_index.json')
    n, repeat = PRESETS[a.preset]
    command = [sys.executable, '-u', str(ROOT / 'run_dct_qim_stega.py'),
        '--model_id', str(model), '--prompt_file', str(prompts),
        '--start', '0', '--end', str(a.images), '--gen_seed', str(a.seed),
        '--coding_mode', 'ldpc', '--ldpc_rate', '0.6666666666666666',
        '--ldpc_column_weight', '3', '--ldpc_seed', '20260826',
        '--ldpc_iterations', '100', '--ldpc_scaling', '0.8',
        '--stega_key', '999999', '--stega_msg_seed', '2024',
        '--stega_n_bits', str(n), '--stega_repeat', str(repeat),
        '--qim_min_radius', '8', '--qim_max_radius', '32', '--qim_step', '3.0',
        '--qim_embedding_mode', 'nearest', '--qim_step_mode', 'global',
        '--image_length', '512', '--guidance_scale', '7.5',
        '--num_inference_steps', '50', '--test_num_inference_steps', '50',
        '--save_images', '--save_clean_images', '--save_latents', '--save_llrs',
        '--compute_image_metrics', '--output_dir', str(Path(a.output_dir).resolve()),
        '--run_name', '{}_{}_{}_seed{}'.format(a.model_tag, a.preset, a.attack, a.seed)]
    command += ATTACKS[a.attack]
    print(' '.join(shlex.quote(x) for x in command), flush=True)
    if not a.dry_run:
        subprocess.run(command, cwd=str(ROOT), check=True)


if __name__ == '__main__':
    main()
