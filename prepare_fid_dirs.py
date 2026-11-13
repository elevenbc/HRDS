"""Prepare clean/stego/received image folders from one experiment run.

The source run is never modified. Each output image is copied according to the
paths recorded in results.jsonl, and the script validates one complete record
per image before writing the folders.
"""
import argparse
import json
import shutil
from pathlib import Path


def load_records(run_dir):
    path = run_dir / "results.jsonl"
    if not path.is_file():
        raise FileNotFoundError("missing results.jsonl: {}".format(path))
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("invalid JSON at line {}: {}".format(line_no, exc))
            rows.append(row)
    if not rows:
        raise ValueError("results.jsonl contains no records")
    indices = [int(row["index"]) for row in rows]
    if len(set(indices)) != len(indices):
        raise ValueError("duplicate record indices found")
    return sorted(rows, key=lambda row: int(row["index"]))


def copy_set(rows, run_dir, out_dir, field):
    target = out_dir / field
    target.mkdir(parents=True, exist_ok=False)
    for row in rows:
        rel = row.get(field + "_image")
        if not rel:
            raise ValueError("record {} has no {}_image field".format(row["index"], field))
        source = (run_dir / rel).resolve()
        if not source.is_file():
            raise FileNotFoundError("record {} points to missing file: {}".format(row["index"], source))
        suffix = source.suffix.lower() or ".png"
        destination = target / ("{:06d}{}".format(int(row["index"]), suffix))
        shutil.copy2(str(source), str(destination))
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--include_received", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError("output already exists; choose a new path: {}".format(out_dir))
    rows = load_records(run_dir)
    out_dir.mkdir(parents=True)
    created = [copy_set(rows, run_dir, out_dir, "clean"), copy_set(rows, run_dir, out_dir, "stego")]
    if args.include_received:
        created.append(copy_set(rows, run_dir, out_dir, "received"))

    manifest = {
        "run_dir": str(run_dir),
        "records": len(rows),
        "sets": {name: len(list((out_dir / name).iterdir())) for name in ("clean", "stego")},
        "included_received": bool(args.include_received),
    }
    if args.include_received:
        manifest["sets"]["received"] = len(list((out_dir / "received").iterdir()))
    with (out_dir / "fid_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
