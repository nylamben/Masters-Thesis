"""Read-only audit of recorded predictions, metrics, splits and selections."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def validate_run(run):
    run = Path(run)
    if not json.loads((run / "completed.json").read_text())["success"]:
        raise ValueError("The run has no successful completion marker")
    summaries = []
    for scope in ["main", "loco_1", "loco_2", "loco_3"]:
        folder = run / "results"
        split = json.loads((folder / f"{scope}_split.json").read_text())
        assert set(split["train"]).isdisjoint(split["validation"])
        assert set(split["final_fit"]).isdisjoint(split["test"])
        assert len(split["test"]) == (3 if scope == "main" else 5)
        search = pd.read_json(folder / f"{scope}_search.json")
        for selected in json.loads((folder / f"{scope}_selection.json").read_text()):
            candidates = search[search.variant.eq(selected["variant"])]
            assert len(candidates) == 6
            winner = candidates.sort_values(["validation_macro_MAE", "candidate"]).iloc[0]
            assert selected["candidate"] == winner.candidate
            assert np.isclose(selected["validation_macro_MAE"], winner.validation_macro_MAE)
        pred = pd.read_csv(folder / f"{scope}_predictions.csv")
        assert np.isfinite(pred[["rul_minutes", "prediction"]]).all().all()
        assert pred.prediction.ge(0).all()
        assert not pred.duplicated(["variant", "bearing_id", "cycle"]).any()
        for _, group in pred.groupby("variant"):
            assert set(group.bearing_id) == set(split["test"])
        metrics = pd.read_csv(folder / f"{scope}_metrics.csv")
        assert len(metrics) == 5 * len(split["test"])
        assert not metrics.duplicated(["variant", "bearing_id"]).any()
        for row in metrics.itertuples():
            group = pred[pred.variant.eq(row.variant) & pred.bearing_id.eq(row.bearing_id)]
            assert len(group) == row.n_snapshots
            err = group.rul_minutes - group.prediction
            assert np.isclose(row.MAE, err.abs().mean())
            assert np.isclose(row.RMSE, np.sqrt(np.square(err).mean()))
        summaries.append(metrics.groupby(["scope", "variant"])[["MAE", "RMSE"]].mean().reset_index())
    recomputed = pd.concat(summaries).sort_values(["scope", "variant"]).reset_index(drop=True)
    saved = pd.read_csv(run / "results" / "macro_summary.csv").sort_values(["scope", "variant"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(saved, recomputed, check_exact=False, rtol=1e-5)
    assert all(x["passed"] for x in json.loads((run / "results" / "resume_verification.json").read_text()))
    assert json.loads((run / "results" / "shap_audit.json").read_text())["max_additivity_error"] < 1e-3
    assert len(list((run / "figures").glob("*.png"))) == 5
    return {"passed": True, "scopes": 4, "macro_rows": len(saved), "metric_recalculation": "passed"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "artifacts" / "reference_run")
    print(json.dumps(validate_run(parser.parse_args().run_dir), indent=2))
