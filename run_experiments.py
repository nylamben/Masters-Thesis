"""Repository entry point. Validate by default; train only with --train."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys

import numpy as np
import benchmark_engine as engine

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "XJTU_SY_features.csv"
EXPECTED_DATA_SHA256 = "31993f6277f598e99defc14d152e36aebf5d6d458a14a5cb26b2cc4f061a02d6"


def validate_data(path=DEFAULT_DATA):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Supply data/XJTU_SY_features.csv separately or run the optional raw download/extraction scripts; see docs/DOCUMENTATION.md")
    digest = engine.hash_file(path)
    if digest != EXPECTED_DATA_SHA256:
        raise ValueError("Feature CSV checksum does not match the recorded benchmark")
    df = engine.load_dataset(path)
    variants, eligible = engine.prepare_features(df, engine.DEFAULT_CONFIG)
    if len(df) != 9216 or df.shape[1] != 58 or int(eligible.sum()) != 9066:
        raise ValueError("Unexpected benchmark dimensions")
    for _, group in df.groupby("bearing_id"):
        cycles = group.cycle.to_numpy()
        if not np.array_equal(cycles, np.arange(1, cycles.max() + 1)):
            raise ValueError("Incomplete bearing cycle coverage")
        if not np.array_equal(group.rul_minutes, cycles.max() - cycles):
            raise ValueError("Invalid recorded-endpoint labels")
    counts = {name: frame.shape[1] for name, frame in variants.items()}
    if list(counts.values()) != [9, 50, 50]:
        raise ValueError("Unexpected feature counts")
    return {"dataset_sha256": digest, "rows": len(df), "bearings": int(df.bearing_id.nunique()),
            "eligible_rows": int(eligible.sum()), "predictor_counts": counts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "run_v1")
    parser.add_argument("--drive-path", type=Path, help="Run directory on an already mounted Drive filesystem")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--train", action="store_true", help="Execute/resume the fixed full benchmark")
    action.add_argument("--verify-recovery", action="store_true", help="Fit tiny models to check interruption/recovery only")
    action.add_argument("--validate-only", action="store_true", help="Check data without fitting models (default)")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.is_relative_to(ROOT / "artifacts"):
        parser.error("Reference artifacts are read-only; choose an output under runs/")
    if args.drive_path:
        drive = args.drive_path.resolve()
        if output == drive or output.is_relative_to(drive) or drive.is_relative_to(output):
            parser.error("Local output and Drive backup must be separate, non-nested directories")
        if not drive.parent.is_dir():
            parser.error("Mount Drive first and choose an existing parent for --drive-path")
    print(json.dumps(validate_data(args.data), indent=2), flush=True)
    if not (args.train or args.verify_recovery):
        print("Validation passed. No models fitted. Use --train to run experiments.")
        return 0
    if args.drive_path and args.drive_path.exists() and not output.exists():
        shutil.copytree(args.drive_path, output)
        print("Restored the saved run from Drive", flush=True)
    if args.verify_recovery:
        df = engine.load_dataset(args.data)
        variants, eligible = engine.prepare_features(df, engine.DEFAULT_CONFIG)
        engine.verify_resume(df, variants, eligible, output, engine.DEFAULT_CONFIG)
        print((output / "results" / "resume_verification.json").read_text())
        return 0
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    engine.atomic_json(output / "invocation.json", {
        "repository_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "command": sys.argv, "source_sha256": engine.hash_file(ROOT / "benchmark_engine.py")})
    summary = engine.run_all(args.data, output, drive_path=args.drive_path)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
