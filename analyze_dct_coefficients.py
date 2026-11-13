"""CPU analysis of paired, saved DCT-QIM latents (Python 3.7 compatible).

This script never loads a diffusion model or generates/inverts an image.
The reference for stego errors is stego_init, NOT clean_init.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ATTACK_KEYS = ("jpeg_ratio", "gaussian_blur_r", "gaussian_std", "brightness_factor",
               "r_degree", "crop_scale", "crop_ratio")
CONFIG_KEYS = ("model_id", "image_length", "guidance_scale", "num_inference_steps",
               "test_num_inference_steps", "gen_seed", "stega_key", "stega_msg_seed",
               "stega_n_bits", "stega_repeat", "qim_step", "qim_min_radius",
               "qim_max_radius", "coding_mode", "ldpc_rate", "ldpc_seed",
               "ldpc_column_weight", "ldpc_iterations", "ldpc_scaling")
COLORS = {"none": "#0072B2", "jpeg75": "#D55E00", "blur1": "#009E73",
          "blur2": "#CC79A7", "brightness020": "#E69F00",
          "gaussian001": "#56B4E9", "gaussian003": "#333333"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_run(path):
    path = Path(path).resolve()
    manifest = read_json(path / "manifest.json")
    rows = {}
    with (path / "results.jsonl").open(encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                idx = int(row["index"])
                if idx in rows:
                    raise ValueError("Duplicate sample index: %s" % idx)
                rows[idx] = row
    if not rows:
        raise ValueError("No records: %s" % path)
    expected = read_json(path / "summary.json")["images"]
    if len(rows) != expected:
        raise ValueError("results.jsonl count does not match summary.json")
    return path, manifest, rows


def check_base(manifest):
    a = manifest["arguments"]
    if any(a.get(k) is not None for k in ATTACK_KEYS) or a.get("rand_aug", 0):
        raise ValueError("The base run must be a no-attack run")
    if a.get("qim_embedding_mode", "nearest") != "nearest" or a.get("qim_step_mode", "global") != "global":
        raise ValueError("This diagnostic currently supports nearest/global QIM")
    if a["coding_mode"] != "ldpc":
        raise ValueError("Use the paper's LDPC run for this diagnostic")


def pair_runs(base, other):
    bp, bm, br = base
    op, om, rr = other
    for key in CONFIG_KEYS:
        if bm["arguments"].get(key) != om["arguments"].get(key):
            raise ValueError("Incompatible %s: %s / %s" % (key, bp, op))
    for key, default in (("qim_embedding_mode", "nearest"), ("qim_step_mode", "global")):
        if bm["arguments"].get(key, default) != om["arguments"].get(key, default):
            raise ValueError("Different QIM mode")
    for key in ("qim_utils.py", "ecc_utils.py"):
        if bm.get("source_sha256", {}).get(key) != om.get("source_sha256", {}).get(key):
            raise ValueError("Different source snapshot for %s" % key)
    if set(br) != set(rr):
        raise ValueError("Different sample indices; do not silently intersect sets")
    for idx in sorted(br):
        for key in ("index", "seed", "prompt", "coded_bits", "secret_bits"):
            if br[idx][key] != rr[idx][key]:
                raise ValueError("Unpaired sample %s: %s" % (idx, key))
        for key in ("clean_latent", "stego_latent"):
            if not np.array_equal(load_latent(bp / br[idx][key]), load_latent(op / rr[idx][key])):
                raise ValueError("Initial latent differs at sample %s (%s)" % (idx, key))


def load_latent(path):
    x = np.load(str(path), allow_pickle=False)
    if x.ndim != 4 or x.shape[0] != 1 or x.shape[-1] != x.shape[-2]:
        raise ValueError("Expected [1,C,H,H] at %s" % path)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite latent at %s" % path)
    return x.astype(np.float64, copy=False)


def dct_basis(size):
    k = np.arange(size, dtype=np.float64)[:, None]
    t = np.arange(size, dtype=np.float64)[None, :]
    b = np.cos(np.pi * (t + 0.5) * k / size) * np.sqrt(2.0 / size)
    b[0] /= np.sqrt(2.0)
    return b


def dct2(x, basis=None):
    b = dct_basis(x.shape[-1]) if basis is None else basis
    return np.matmul(np.matmul(b, x), b.T)[0]


def geometry(shape, args):
    channels, size, _ = shape
    yy, xx = np.mgrid[:size, :size]
    spatial = (np.hypot(yy, xx) >= args["qim_min_radius"]) & (np.hypot(yy, xx) <= args["qim_max_radius"])
    coords = np.argwhere(spatial)
    pos = np.array([(c, y, x) for c in range(channels) for y, x in coords], dtype=np.int64)
    pos = pos[np.random.default_rng(args["stega_key"]).permutation(len(pos))]
    count = int(args["stega_n_bits"]) * int(args["stega_repeat"])
    if count > len(pos):
        raise ValueError("Carrier budget exceeded")
    used = np.zeros(shape, dtype=bool)
    used[tuple(pos[:count].T)] = True
    band = np.broadcast_to(spatial, shape).copy()
    return pos[:count], {"used": used, "in_band": band, "outside_band": ~band, "all": np.ones(shape, bool)}


def qim_observation(coeff, pos, args, row):
    n, repeat = int(args["stega_n_bits"]), int(args["stega_repeat"])
    step = float(args["qim_step"])
    offsets = np.random.default_rng(int(args["stega_key"]) ^ 0x5EED).uniform(0.0, 1.0, len(pos)) * step
    x = coeff[tuple(pos.T)]
    q0 = step * np.rint((x - offsets) / step) + offsets
    q1 = step * np.rint((x - offsets - step / 2) / step) + offsets + step / 2
    llr = (np.abs(x - q1) - np.abs(x - q0)).astype(np.float32)
    combined = llr.reshape(n, repeat).sum(axis=1)
    coded = np.asarray(row["coded_bits"], dtype=np.int8)
    hard = (combined < 0).astype(np.int8)
    return {"carrier_ber": float(np.mean((llr < 0) != np.repeat(coded, repeat))),
            "coded_ber": float(np.mean(hard != coded)), "hard": hard, "llr": combined}


def statistics(x, reference, observation, margin):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    ax = np.abs(x)
    ref, obs = reference.reshape(-1), observation.reshape(-1)
    corr = float(np.corrcoef(ref, obs)[0, 1]) if ref.std() > 0 and obs.std() > 0 else None
    return {"coefficient_count": int(x.size), "bias": float(x.mean()), "std": float(x.std()),
            "mae": float(ax.mean()), "rmse": float(np.sqrt(np.mean(x ** 2))),
            "abs_p50": float(np.quantile(ax, 0.50)), "abs_p95": float(np.quantile(ax, 0.95)),
            "abs_p99": float(np.quantile(ax, 0.99)),
            "fraction_abs_lt_delta_over_4": float(np.mean(ax < margin)), "correlation": corr}


def parse_named(values):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("Use label=/path/to/run, got %s" % value)
        label, path = value.split("=", 1)
        if not label or label == "none" or label in result:
            raise ValueError("Invalid/duplicate comparison label: %s" % label)
        result[label] = path
    return result


def plot_svg(curves, args, margin, out):
    """Dependency-free vector plots, also usable on the offline server."""
    import xml.etree.ElementTree as ET
    namespace = "http://www.w3.org/2000/svg"
    ET.register_namespace("", namespace)
    arms = sorted({k[0] for k in curves})
    hist_rows = []
    for kind in ("total", "additional"):
        candidates = [x for (arm, condition, kd), x in curves.items() if kd == kind]
        if not candidates:
            continue
        limit = args.x_limit or max(2 * margin, max(float(np.quantile(np.abs(x), 0.995)) for x in candidates))
        edges = np.linspace(-limit, limit, args.bins + 1)
        densities = {k: np.histogram(x, bins=edges)[0] / (float(len(x)) * np.diff(edges))
                     for k, x in curves.items() if k[2] == kind}
        for cdf in (False, True):
            root = ET.Element("{%s}svg" % namespace, {"width": "7.1in", "height": "2.85in", "viewBox": "0 0 1100 442"})
            def shape(tag, **attrs):
                return ET.SubElement(root, "{%s}%s" % (namespace, tag), {k.replace("_", "-"): str(v) for k, v in attrs.items()})
            def text(x, y, value, size=18, **attrs):
                node = shape("text", x=x, y=y, font_family="Times New Roman, Liberation Serif, serif", font_size=size, fill="#222222", **attrs)
                node.text = value
            shape("rect", x=0, y=0, width=1100, height=442, fill="white")
            text(550, 23, "Total inversion error" if kind == "total" else "Additional attack difference", 20, text_anchor="middle")
            ymax = 1.0 if cdf else max(float(d.max()) for d in densities.values()) * 1.12
            ymax = ymax if ymax > 0 else 1.0
            for panel, arm in enumerate(arms):
                pw = 1100 / len(arms)
                left, top, width, height = panel * pw + 80, 64, pw - 110, 300
                xmin, xmax = (0, limit) if cdf else (-limit, limit)
                sx = lambda x: left + (x - xmin) / (xmax - xmin) * width
                sy = lambda y: top + height - y / ymax * height
                title = "No embedding" if arm == "clean" else "QIM embedding"
                text(left + width / 2, top - 14, "(%s) %s" % (chr(97 + panel), title), 19, text_anchor="middle")
                for t in np.linspace(0, ymax, 5):
                    shape("line", x1=left, x2=left + width, y1=sy(t), y2=sy(t), stroke="#eeeeee", stroke_width=1)
                    text(left - 10, sy(t) + 5, "%.2g" % t, 16, text_anchor="end")
                for t in np.linspace(xmin, xmax, 5):
                    shape("line", x1=sx(t), x2=sx(t), y1=top + height, y2=top + height + 5, stroke="#333333")
                    text(sx(t), top + height + 25, "%.2g" % t, 16, text_anchor="middle")
                for threshold in ([margin] if cdf else [-margin, margin]):
                    shape("line", x1=sx(threshold), x2=sx(threshold), y1=top, y2=top + height,
                          stroke="#777777", stroke_dasharray="5 5", stroke_width=1)
                entries = [(condition, x) for (ar, condition, kd), x in curves.items() if ar == arm and kd == kind]
                for j, (condition, x) in enumerate(entries):
                    color = COLORS.get(condition, "#333333")
                    if cdf:
                        grid = np.linspace(0, limit, 600)
                        values = np.searchsorted(np.sort(np.abs(x)), grid, side="right") / float(len(x))
                        points = zip(grid, values)
                    else:
                        density = densities[arm, condition, kind]
                        points = [(edges[k + d], density[k]) for k in range(len(density)) for d in (0, 1)]
                        counts, _ = np.histogram(x, bins=edges)
                        for k, count in enumerate(counts):
                            hist_rows.append({"arm": arm, "difference": kind, "condition": condition,
                                "bin_left": edges[k], "bin_right": edges[k + 1], "count": int(count),
                                "density": float(density[k]), "total_coefficients": int(x.size),
                                "underflow": int(np.sum(x < edges[0])), "overflow": int(np.sum(x > edges[-1]))})
                    shape("polyline", points=" ".join("%.3f,%.3f" % (sx(xv), sy(yv)) for xv, yv in points),
                          fill="none", stroke=color, stroke_width=2)
                    lx, ly = left + width - 135, top + 18 + j * 23
                    shape("line", x1=lx, x2=lx + 23, y1=ly - 5, y2=ly - 5, stroke=color, stroke_width=2.5)
                    text(lx + 29, ly, condition, 16)
                shape("line", x1=left, x2=left + width, y1=top + height, y2=top + height, stroke="#333333", stroke_width=1.4)
                shape("line", x1=left, x2=left, y1=top, y2=top + height, stroke="#333333", stroke_width=1.4)
                text(left + width / 2, 417, "Absolute DCT coefficient difference" if cdf else "DCT coefficient difference", 18, text_anchor="middle")
                tx, ty = left - 55, top + height / 2
                text(tx, ty, "Cumulative probability" if cdf else "Probability density", 18,
                     text_anchor="middle", transform="rotate(-90 %s %s)" % (tx, ty))
            stem = "dct_%s_%s" % (kind, "cdf" if cdf else "histogram")
            ET.ElementTree(root).write(str(out / (stem + ".svg")), encoding="utf-8", xml_declaration=True)
    write_csv(out / "histogram_bins.csv", hist_rows)


def plot_curves(curves, args, margin, out):
    plot_svg(curves, args, margin, out)
    try:
        import matplotlib
    except ImportError:
        print("Matplotlib unavailable: saved editable SVG figures and all CSV/JSON outputs; no installation required.")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 10, "axes.titlesize": 11, "pdf.fonttype": 42})
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#333333"]
    palette = {name: COLORS.get(name, colors[i % len(colors)]) for i, name in enumerate(sorted({k[1] for k in curves}))}
    arms = sorted({k[0] for k in curves})
    histogram_rows = []
    for kind in ("total", "additional"):
        for cumulative in (False, True):
            fig, axes = plt.subplots(1, len(arms), figsize=(7.1, 2.8), squeeze=False, sharex=True, sharey=True)
            candidates = [v for (arm, cond, kd), v in curves.items() if kd == kind]
            if not candidates:
                plt.close(fig)
                continue
            limit = args.x_limit or max(2 * margin, max(float(np.quantile(np.abs(v), 0.995)) for v in candidates))
            edges = np.linspace(-limit, limit, args.bins + 1)
            for ax, arm in zip(axes[0], arms):
                entries = [(cond, x) for (a, cond, kd), x in curves.items() if a == arm and kd == kind]
                for cond, x in entries:
                    if cumulative:
                        grid = np.linspace(0, limit, 600)
                        values = np.searchsorted(np.sort(np.abs(x)), grid, side="right") / float(len(x))
                        ax.plot(grid, values, label=cond, color=palette[cond], linewidth=1.5)
                    else:
                        counts, _ = np.histogram(x, bins=edges)
                        density = counts / (float(x.size) * np.diff(edges))
                        ax.step((edges[:-1] + edges[1:]) / 2, density, where="mid", label=cond, color=palette[cond], linewidth=1.2)
                        for j, count in enumerate(counts):
                            histogram_rows.append({"arm": arm, "difference": kind, "condition": cond,
                                "bin_left": edges[j], "bin_right": edges[j + 1], "count": int(count),
                                "density": float(density[j]), "total_coefficients": int(x.size),
                                "underflow": int(np.sum(x < edges[0])), "overflow": int(np.sum(x > edges[-1]))})
                ax.axvline(margin, color="0.45", linestyle="--", linewidth=0.8)
                if not cumulative:
                    ax.axvline(-margin, color="0.45", linestyle="--", linewidth=0.8)
                ax.set_title("(a) No embedding" if arm == "clean" else "(b) QIM embedding" if len(arms) > 1 else "QIM embedding")
                ax.set_xlabel("Absolute DCT coefficient difference" if cumulative else "DCT coefficient difference")
                ax.set_ylabel("Empirical cumulative probability" if cumulative else "Probability density")
                ax.grid(alpha=0.15)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                if entries:
                    ax.legend(frameon=False, fontsize=8)
                ax.set_xlim((0, limit) if cumulative else (-limit, limit))
                if cumulative:
                    ax.set_ylim(0, 1.01)
            fig.tight_layout(pad=0.6)
            stem = "dct_%s_%s" % (kind, "cdf" if cumulative else "histogram")
            fig.savefig(str(out / (stem + ".png")), dpi=300, bbox_inches="tight")
            fig.savefig(str(out / (stem + ".pdf")), bbox_inches="tight")
            plt.close(fig)
    write_csv(out / "histogram_bins.csv", histogram_rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True, help="No-attack run with saved latents")
    p.add_argument("--compare_run", action="append", default=[], help="label=attack_run; repeatable")
    p.add_argument("--collection_dir", help="Optional output of collect_dct_inversions.py")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--expected_images", type=int, default=172)
    p.add_argument("--bins", type=int, default=120)
    p.add_argument("--x_limit", type=float, default=None)
    p.add_argument("--no_plots", action="store_true")
    args = p.parse_args()
    if args.bins < 2 or (args.x_limit is not None and args.x_limit <= 0):
        p.error("bins >= 2 and x_limit > 0 required")
    base = load_run(args.run_dir)
    bp, bm, br = base
    check_base(bm)
    if len(br) != args.expected_images:
        raise ValueError("Expected %s images, found %s" % (args.expected_images, len(br)))
    observations = {"stego": {"none": {i: (bp / r["reversed_latent"], r) for i, r in br.items()}}}
    sources = {"none": str(bp)}
    for label, path in parse_named(args.compare_run).items():
        other = load_run(path)
        pair_runs(base, other)
        op, om, rr = other
        if not any(om["arguments"].get(k) is not None for k in ATTACK_KEYS):
            raise ValueError("Comparison has no attack: %s" % label)
        observations["stego"][label] = {i: (op / r["reversed_latent"], r) for i, r in rr.items()}
        sources[label] = {"run": str(op), "attack": {k: om["arguments"].get(k) for k in ATTACK_KEYS}}
    if args.collection_dir:
        cp = Path(args.collection_dir).resolve()
        cm = read_json(cp / "manifest.json")
        if cm["base_manifest"] != bm:
            raise ValueError("Collection was created for a different base manifest")
        for arm, conditions in cm["files"].items():
            observations.setdefault(arm, {})
            for label, records in conditions.items():
                if set(map(int, records)) != set(br):
                    raise ValueError("Incomplete collection: %s/%s" % (arm, label))
                observations[arm][label] = {int(i): (cp / r["latent"], r) for i, r in records.items()}
        sources["collection"] = cm
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    a = bm["arguments"]
    step, margin = float(a["qim_step"]), float(a["qim_step"]) / 4
    first = dct2(load_latent(bp / br[min(br)]["stego_latent"]))
    pos, masks = geometry(first.shape, a)
    image_rows, recovery, aggregate, curves, message_rows = [], [], [], {}, []
    for arm, conditions in observations.items():
        if "none" not in conditions:
            raise ValueError("Each arm needs a none condition for paired additional differences")
        for condition, records in conditions.items():
            comm, pooled = [], {}
            for idx in sorted(br):
                original = dct2(load_latent(bp / br[idx]["stego_latent" if arm == "stego" else "clean_latent"]))
                observed = dct2(load_latent(records[idx][0]))
                noattack = dct2(load_latent(conditions["none"][idx][0]))
                if observed.shape != original.shape or noattack.shape != original.shape:
                    raise ValueError("Latent shapes differ")
                diffs = {"total": (observed - original, original)}
                if condition != "none":
                    diffs["additional"] = (observed - noattack, noattack)
                for kind, (difference, reference) in diffs.items():
                    for region, mask in masks.items():
                        key = (arm, condition, kind, region)
                        pooled.setdefault(key, []).append((difference[mask], reference[mask], observed[mask]))
                        st = statistics(difference[mask], reference[mask], observed[mask], margin)
                        image_rows.append(dict(arm=arm, condition=condition, difference=kind, region=region, index=idx, seed=br[idx]["seed"], **st))
                if arm == "stego":
                    q = qim_observation(observed, pos, a, br[idx])
                    record = records[idx][1]
                    metrics = {k: record.get(k) for k in ("ber", "exact_recovery", "ldpc_checks_satisfied")}
                    stored = record.get("coded_ber")
                    if stored is not None and abs(q["coded_ber"] - float(stored)) > 1e-7:
                        raise ValueError("Recomputed/stored coded BER differ at %s/%s. Check source versions." % (condition, idx))
                    comm.append(dict(carrier_ber=q["carrier_ber"], coded_ber=q["coded_ber"], **metrics))
                    message_rows.append(dict(condition=condition, index=idx, seed=br[idx]["seed"],
                        carrier_ber=q["carrier_ber"], coded_ber=q["coded_ber"],
                        secret_ber=metrics["ber"], exact_recovery=metrics["exact_recovery"],
                        ldpc_checks_satisfied=metrics["ldpc_checks_satisfied"]))
            # Release non-plotted raw arrays after each condition, rather than
            # retaining every coefficient from every condition in memory.
            for (ar, cond, kind, region), triples in pooled.items():
                x, ref, obs = [np.concatenate([t[j] for t in triples]) for j in range(3)]
                aggregate.append(dict(arm=ar, condition=cond, difference=kind, region=region,
                                      images=len(br), **statistics(x, ref, obs, margin)))
                if region == "used":
                    curves[ar, cond, kind] = x
            if arm == "stego":
                recovery.append({"condition": condition, "images": len(comm),
                    "carrier_ber": float(np.mean([x["carrier_ber"] for x in comm])),
                    "coded_ber": float(np.mean([x["coded_ber"] for x in comm])),
                    "secret_ber": float(np.mean([x["ber"] for x in comm])) if all(x["ber"] is not None for x in comm) else None,
                    "exact_recovery_count": sum(bool(x["exact_recovery"]) for x in comm) if all(x["exact_recovery"] is not None for x in comm) else None,
                    "exact_recovery_rate": float(np.mean([x["exact_recovery"] for x in comm])) if all(x["exact_recovery"] is not None for x in comm) else None})
            print("Analyzed %s / %s: %s paired images" % (arm, condition, len(br)), flush=True)
    write_csv(out / "per_image_dct_statistics.csv", image_rows)
    write_csv(out / "aggregate_dct_statistics.csv", aggregate)
    write_csv(out / "message_recovery.csv", recovery)
    write_csv(out / "per_image_message_recovery.csv", message_rows)
    write_json(out / "dct_analysis_summary.json", {"images": len(br), "qim_step": step,
        "nominal_qim_margin": margin, "source_runs": sources, "statistics": aggregate,
        "message_recovery": recovery,
        "definitions": {"total": "DCT(inversion(image_after_condition)) - DCT(initial_latent_for_that_image)",
            "additional": "DCT(inversion(image_after_condition)) - DCT(inversion(the_same_image_without_attack))",
            "clean_arm": "original, unembedded clean image and clean_init reference",
            "stego_arm": "QIM embedded image and stego_init reference",
            "margin": "Delta/4 is a nominal nearest binary-QIM single-carrier margin; its coverage is not BER or LDPC recovery",
            "sampling": "coefficients are pooled descriptively; images remain the sample unit; no coefficient-level significance test"}})
    if not args.no_plots:
        plot_curves(curves, args, margin, out)
    for row in recovery:
        print("%s: carrier BER=%.6f coded BER=%.6f secret BER=%s exact=%s/%s" % (
            row["condition"], row["carrier_ber"], row["coded_ber"], row["secret_ber"], row["exact_recovery_count"], row["images"]))
    print("SUMMARY:", out / "dct_analysis_summary.json", flush=True)


if __name__ == "__main__":
    main()
