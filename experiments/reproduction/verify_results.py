"""Portable audit of sealed PACT outputs. No network, training or model selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal_numbers(actual, expected, context):
    for key, value in actual.items():
        if key not in expected:
            continue
        if not np.isclose(value, expected[key], rtol=1e-11, atol=1e-12):
            raise AssertionError((context, key, value, expected[key]))


def assigned(keys, seed):
    order = np.argsort(keys, kind="stable")
    mixture = np.arange(len(keys)) % 2
    np.random.default_rng(seed).shuffle(mixture)
    result = np.empty(len(keys), int)
    result[order] = mixture
    return result


def replay(data, observation, choice, seed, force=False, uniform=False):
    from budget import allocate_shared_budget

    n = len(data["sample"])
    rows = np.arange(n)
    e = assigned(data["sample"], seed)
    o = {k: v[e, rows] for k, v in observation.items()}
    b = data["baseline"]
    eligible = bool(choice["enabled"]) & (b > 0) & (o["proposal"] > 0) & (o["value"] > 0)
    values = np.where(o["value"] > 0, 1.0, o["value"]) if uniform else o["value"]
    labels = [str(key) + "#" + str(j) for j, key in enumerate(data["sample"])]
    trial = allocate_shared_budget(
        values, eligible, np.zeros(n, bool), np.full(n, 0.25), labels, seed=seed
    )["geometry_trials"]
    threshold = choice["thresholds"]
    accepted = np.ones(n, bool) if force else (
        (o["u"] > threshold[0]) & (o["j"] >= threshold[1])
    )
    action = np.where(trial & accepted, o["proposal"], b)
    return action, trial


def verify_qwen(root):
    import frozen_math as math

    with np.load(root / "data/qwen_native_data.npz", allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    results = read(root / "data/qwen_result.json")
    predictions = read(root / "data/qwen_predictions.json")["predictions"]
    candidates = read(root / "data/qwen_frozen_candidates.json")["candidates"]
    controls = read(root / "data/qwen_training_controls.json")
    n = len(data["sample"])
    assert n == 1024 and len(set(data["image"])) == 512
    assert int(data["positive"].sum()) == 640 and int(data["valid"].sum()) == 638
    assert len(results["results"]) == 19
    native_iou = np.divide(
        data["intersections"], data["unions"],
        out=np.zeros_like(data["iou"]), where=data["unions"] > 0,
    )
    np.testing.assert_allclose(native_iou[data["valid"]], data["iou"][data["valid"]],
                               rtol=0, atol=1e-14)
    ii = np.arange(n)
    observations, choices = {}, {}
    for candidate, item in candidates.items():
        changed, whole = item["models"]["changed"], item["models"]["whole"]
        same = math.observed(data, changed["experts"], ii)
        for name in ["PACT", "preMean", "Whole39", "Whole39SameProposal",
                     "ForcedAcceptSameTrial", "UniformPositiveBudget"]:
            arm = candidate + "_" + name
            representation = whole if name in ["Whole39", "Whole39SameProposal"] else changed
            observations[arm] = math.observed(
                data, representation["experts"], ii,
                "mean" if name == "preMean" else "delivery",
                same if name == "Whole39SameProposal" else None,
            )
            choices[arm] = representation["choice"]
    for representation, prefix in [("changed", ""), ("whole", "Whole_")]:
        for name, experts in controls[representation].items():
            arm = "wide_" + prefix + name
            observations[arm] = math.observed(data, experts, ii)
            choices[arm] = {"enabled": True, "thresholds": [0.0, 0.0]}
    assert set(observations) == set(results["results"]) - {"SFT"}
    metric_rows, replayed = 0, 0
    for arm, stats in results["results"].items():
        rows = []
        for saved in stats["per_seed"]:
            seed = saved["seed"]
            sealed = predictions[arm][str(seed)]
            if arm == "SFT":
                action, trial = data["baseline"].copy(), np.zeros(n, bool)
            else:
                action, trial = replay(
                    data, observations[arm], choices[arm], seed,
                    force=arm.endswith("ForcedAcceptSameTrial"),
                    uniform=arm.endswith("UniformPositiveBudget"),
                )
                replayed += 1
            np.testing.assert_array_equal(action, sealed["action"])
            np.testing.assert_array_equal(trial, sealed["trial"])
            assert trial.sum() <= n // 4
            row = math.metrics(data, ii, action, trial)
            equal_numbers(row, saved, (arm, seed))
            equal_numbers(row, sealed["metrics"], (arm, seed, "sealed"))
            rows.append(row)
            metric_rows += 1
        mean = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
        equal_numbers(mean, stats["mean"], (arm, "mean"))
    return {"arms": 19, "seed_metric_rows": metric_rows,
            "nonbaseline_policies_reconstructed": replayed,
            "native_valid_domain_integer_IoU_checked": True,
            "bootstrap_intervals": "Preserved; not recomputed by this quick audit"}


def verify_internvl(root):
    with np.load(root / "data/internvl_metrics_arrays.npz", allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    results = read(root / "data/internvl_result.json")
    predictions = read(root / "data/internvl_predictions.json")["predictions"]
    b, y, valid = data["baseline"], data["positive"], data["valid"]
    ii = np.arange(len(b))
    fields_checked, metric_rows = set(), 0
    assert len(b) == 1024 and len(set(data["image"])) == 512
    assert len(results["results"]) == 7
    for arm, stats in results["results"].items():
        rows = []
        for saved in stats["per_seed"]:
            seed = saved["seed"]
            sealed = predictions[arm][str(seed)]
            action, trial = np.asarray(sealed["action"]), np.asarray(sealed["trial"], bool)
            assert np.array_equal(action > 0, b > 0) and trial.sum() <= len(b) // 4
            du = data["utility"][ii, action] - data["utility"][ii, b]
            dj = data["iou"][ii, action] - data["iou"][ii, b]
            row = {
                "FP": int(((action > 0) & ~y).sum()),
                "FN": int(((action == 0) & y).sum()),
                "negative": int((~y).sum()), "positive": int(y.sum()),
                "FPR": float(((action > 0) & ~y).sum() / (~y).sum()),
                "FNR": float(((action == 0) & y).sum() / y.sum()),
                "net_original_utility": float(du.mean()),
                "net_hard_iou": float(dj[valid].mean()),
                "gIoU": float(data["iou"][ii, action][valid].mean()),
                "harm": float(((action > 0) & (b > 0) & (dj < -0.02) & valid).sum()
                              / valid.sum()),
                "extra_decode": float(trial.mean()),
                "newFP": 0.0, "newMiss": 0.0,
            }
            equal_numbers(row, saved["metrics"], (arm, seed))
            rows.append(row)
            fields_checked.update(row)
            metric_rows += 1
        mean = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
        equal_numbers(mean, stats["mean"], (arm, "mean"))
    return {"arms": 7, "seed_metric_rows": metric_rows, "fields_checked": sorted(fields_checked),
            "not_recomputed_from_compact_arrays": [
                "cIoU (integer intersections/unions absent)",
                "proposal selection (full InternVL candidate features absent)",
                "joint opportunity counts", "bootstrap intervals", "native online timing",
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON audit output")
    args = parser.parse_args()
    manifest = read(HERE / "manifest.json")
    for entry in manifest["files"]:
        path = HERE / entry["path"]
        assert path.resolve().is_relative_to(HERE.resolve())
        assert sha(path) == entry["sha256"], entry["path"]
    report = {
        "status": "PASSED", "verified_export_files": len(manifest["files"]),
        "qwen": verify_qwen(HERE), "internvl": verify_internvl(HERE),
        "new_fits": 0, "neural_forwards": 0, "source_results_modified": False,
        "scope": "Frozen policy/metric verification; not independent model confirmation",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
