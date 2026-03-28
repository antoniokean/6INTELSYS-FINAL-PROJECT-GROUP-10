import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from train import (
    AUDIO_DATA_DIR,
    BASE_TRAINING_CONFIG,
    DEFAULT_CACHE_DIR,
    MANIFEST_FILE,
    OUTPUT_DIR,
    PRESET_DESCRIPTIONS,
    TRAINING_PRESETS,
    audio_to_log_mel,
    build_cache_path,
    build_legacy_cache_path,
    build_label_map,
    compute_macro_f1,
    is_valid_cached_feature,
    limit_rows,
    load_audio,
    load_manifest,
    resolve_preset_output_dirs,
    save_confusion_matrix,
    serialize_args,
    split_rows,
)


BASELINE_NAME = "nearest_centroid"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a simple non-DL speech-command baseline using nearest-centroid classification."
    )
    parser.add_argument("--preset", default=None)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--save-feature-cache", action="store_true")
    return parser.parse_args()


def apply_baseline_preset(args):
    if args.preset is None:
        args.preset = "medium_end"

    resolved = dict(BASE_TRAINING_CONFIG)
    resolved.update(TRAINING_PRESETS[args.preset])

    for key in ("seed", "max_train_samples", "max_val_samples", "max_test_samples"):
        value = getattr(args, key)
        if value is not None:
            resolved[key] = value

    args.seed = resolved["seed"]
    args.max_train_samples = resolved["max_train_samples"]
    args.max_val_samples = resolved["max_val_samples"]
    args.max_test_samples = resolved["max_test_samples"]
    return args


def resolve_baseline_output_dirs(base_output_dir, preset_name):
    preset_dirs = resolve_preset_output_dirs(base_output_dir, preset_name, "none")
    return {
        "configs": preset_dirs["configs"] / "baselines" / BASELINE_NAME,
        "results": preset_dirs["results"] / "baselines" / BASELINE_NAME,
    }


def resolve_cache_dir(args, preset_name):
    if args.cache_dir != DEFAULT_CACHE_DIR:
        return args.cache_dir
    preset_dirs = resolve_preset_output_dirs(args.output_dir, preset_name, "none")
    return preset_dirs["logs"] / "feature_cache"


def filter_rows_with_available_inputs(rows, cache_dir):
    available_rows = []
    for row in rows:
        cache_path = None
        legacy_cache_path = None
        if cache_dir is not None:
            cache_path = build_cache_path(cache_dir, row["path"])
            legacy_cache_path = cache_dir / Path(row["path"]).with_suffix(".npy")

        has_cached_feature = (
            cache_path is not None
            and (cache_path.exists() or (legacy_cache_path is not None and legacy_cache_path.exists()))
        )
        has_audio = (AUDIO_DATA_DIR / row["path"]).exists()
        if has_cached_feature or has_audio:
            available_rows.append(row)
    return available_rows


def load_feature_vector(relative_path, cache_dir):
    candidate_paths = []
    if cache_dir is not None:
        candidate_paths.extend(
            [
                build_cache_path(cache_dir, relative_path),
                build_legacy_cache_path(cache_dir, relative_path),
            ]
        )

    for candidate_path in candidate_paths:
        if not candidate_path.exists():
            continue
        try:
            features = np.load(candidate_path).astype(np.float32)
        except Exception:
            continue
        if is_valid_cached_feature(features):
            return features.reshape(-1)

    audio_path = AUDIO_DATA_DIR / relative_path
    if audio_path.exists():
        features = audio_to_log_mel(load_audio(audio_path))
        return features.reshape(-1)

    return None


def fit_nearest_centroids(rows, label_map, cache_dir):
    num_classes = len(label_map)
    centroid_sums = None
    counts = np.zeros(num_classes, dtype=np.int64)
    used_rows = 0

    for row in rows:
        feature_vector = load_feature_vector(row["path"], cache_dir)
        if feature_vector is None:
            continue
        label_index = label_map[row["label"]]

        if centroid_sums is None:
            centroid_sums = np.zeros((num_classes, feature_vector.size), dtype=np.float64)

        centroid_sums[label_index] += feature_vector
        counts[label_index] += 1
        used_rows += 1

    if centroid_sums is None:
        raise ValueError("Training dataset is empty.")

    missing_labels = np.where(counts == 0)[0]
    if len(missing_labels) > 0:
        raise ValueError(
            f"Nearest-centroid baseline cannot fit empty classes: {missing_labels.tolist()}"
        )

    centroids = centroid_sums / counts[:, None]
    return centroids.astype(np.float32), counts, used_rows


def predict_label(feature_vector, centroids):
    deltas = centroids - feature_vector[None, :]
    distances = np.einsum("ij,ij->i", deltas, deltas, optimize=True)
    return int(np.argmin(distances))


def evaluate_rows(rows, label_map, cache_dir, centroids, num_classes):
    predictions = []
    labels = []

    for row in rows:
        feature_vector = load_feature_vector(row["path"], cache_dir)
        if feature_vector is None:
            continue
        labels.append(label_map[row["label"]])
        predictions.append(predict_label(feature_vector, centroids))

    labels_array = np.asarray(labels, dtype=np.int64)
    predictions_array = np.asarray(predictions, dtype=np.int64)
    accuracy = float((labels_array == predictions_array).mean()) if len(labels_array) > 0 else None
    macro_f1 = (
        compute_macro_f1(labels_array, predictions_array, num_classes)
        if len(labels_array) > 0
        else None
    )

    return {
        "labels": labels_array,
        "predictions": predictions_array,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "num_samples": int(len(labels_array)),
    }


def build_fallback_cached_split(rows, seed, train_fraction=0.8):
    shuffled_rows = list(rows)
    random.Random(seed).shuffle(shuffled_rows)
    cutoff = max(1, min(len(shuffled_rows) - 1, int(round(len(shuffled_rows) * train_fraction))))
    return shuffled_rows[:cutoff], shuffled_rows[cutoff:]


def write_markdown_summary(path, summary):
    lines = [
        "# Non-DL Baseline Summary",
        "",
        f"- Method: {summary['method']}",
        f"- Preset: {summary['preset']}",
        f"- Seed: {summary['seed']}",
        f"- Fit source: {summary['fit_source']}",
        f"- Training samples: {summary['num_training_samples']}",
        f"- Validation samples: {summary['num_validation_samples']}",
        f"- Testing samples: {summary['num_testing_samples']}",
        f"- Validation accuracy: {summary['final_validation_accuracy']}",
        f"- Validation Macro-F1: {summary['final_validation_macro_f1']}",
        f"- Test accuracy: {summary['final_test_accuracy']}",
        f"- Test Macro-F1: {summary['final_test_macro_f1']}",
        "",
        "This baseline uses the same fixed split and the same 3-channel log-Mel input",
        "features as the CNN, but replaces the neural network with a nearest-centroid classifier.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    start_time = time.time()
    args = apply_baseline_preset(parse_args())

    rows = load_manifest(args.manifest)
    label_map = build_label_map(rows)
    split_to_rows = split_rows(rows)
    cache_dir = resolve_cache_dir(args, args.preset)

    split_to_rows["training"] = filter_rows_with_available_inputs(
        split_to_rows["training"],
        cache_dir,
    )
    split_to_rows["validation"] = filter_rows_with_available_inputs(
        split_to_rows["validation"],
        cache_dir,
    )
    split_to_rows["testing"] = filter_rows_with_available_inputs(
        split_to_rows["testing"],
        cache_dir,
    )

    split_to_rows["training"] = limit_rows(
        split_to_rows["training"],
        args.max_train_samples,
        args.seed,
        "training",
    )
    split_to_rows["validation"] = limit_rows(
        split_to_rows["validation"],
        args.max_val_samples,
        args.seed,
        "validation",
    )
    split_to_rows["testing"] = limit_rows(
        split_to_rows["testing"],
        args.max_test_samples,
        args.seed,
        "testing",
    )

    fit_source = "training"
    if len(split_to_rows["training"]) == 0 and len(split_to_rows["validation"]) >= 2:
        fallback_train, fallback_val = build_fallback_cached_split(
            split_to_rows["validation"],
            args.seed,
        )
        split_to_rows["training"] = fallback_train
        split_to_rows["validation"] = fallback_val
        fit_source = "validation_cache_fallback"

    output_dirs = resolve_baseline_output_dirs(args.output_dir, args.preset)

    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Preset: {args.preset} - {PRESET_DESCRIPTIONS[args.preset]}")
    print("Method: nearest centroid over flattened 3-channel log-Mel features")
    print(f"Manifest: {args.manifest}")
    print(f"Cache dir: {cache_dir}")
    print(f"Fit source: {fit_source}")
    print(
        "Samples used | "
        f"train={len(split_to_rows['training'])} "
        f"val={len(split_to_rows['validation'])} "
        f"test={len(split_to_rows['testing'])}"
    )

    centroids, train_counts, used_train_rows = fit_nearest_centroids(
        split_to_rows["training"],
        label_map,
        cache_dir,
    )

    val_metrics = evaluate_rows(
        split_to_rows["validation"],
        label_map,
        cache_dir,
        centroids,
        len(label_map),
    )

    test_metrics = None
    if len(split_to_rows["testing"]) > 0:
        test_metrics = evaluate_rows(
            split_to_rows["testing"],
            label_map,
            cache_dir,
            centroids,
            len(label_map),
        )

    summary = {
        "method": "nearest_centroid_logmel",
        "preset": args.preset,
        "seed": args.seed,
        "fit_source": fit_source,
        "num_classes": len(label_map),
        "num_training_samples": used_train_rows,
        "num_validation_samples": val_metrics["num_samples"],
        "num_testing_samples": 0 if test_metrics is None else test_metrics["num_samples"],
        "train_label_counts": {
            label: int(train_counts[index])
            for label, index in label_map.items()
        },
        "final_validation_accuracy": val_metrics["accuracy"],
        "final_validation_macro_f1": val_metrics["macro_f1"],
        "final_test_accuracy": None if test_metrics is None else test_metrics["accuracy"],
        "final_test_macro_f1": None if test_metrics is None else test_metrics["macro_f1"],
        "elapsed_seconds": round(time.time() - start_time, 3),
    }

    config_payload = serialize_args(args)
    config_payload["baseline_name"] = BASELINE_NAME
    config_payload["method"] = "nearest centroid over flattened 3-channel log-Mel features"
    config_payload["cache_dir"] = str(cache_dir)

    confusion_source = test_metrics if test_metrics is not None else val_metrics
    confusion_metadata = save_confusion_matrix(
        confusion_source["labels"],
        confusion_source["predictions"],
        label_map,
        output_dirs["results"] / "confusion_matrix.png",
        normalized_output_path=output_dirs["results"] / "confusion_matrix_normalized.png",
    )
    if confusion_metadata is not None:
        confusion_metadata["source_split"] = "test" if test_metrics is not None else "validation"
        with (output_dirs["results"] / "confusion_matrix.json").open("w", encoding="utf-8") as file:
            json.dump(confusion_metadata, file, indent=2)

    with (output_dirs["configs"] / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(config_payload, file, indent=2)
    with (output_dirs["results"] / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    write_markdown_summary(output_dirs["results"] / "summary.md", summary)

    print(f"Validation accuracy: {summary['final_validation_accuracy']}")
    print(f"Validation Macro-F1: {summary['final_validation_macro_f1']}")
    print(f"Test accuracy: {summary['final_test_accuracy']}")
    print(f"Test Macro-F1: {summary['final_test_macro_f1']}")
    print(f"Saved config: {output_dirs['configs'] / 'run_config.json'}")
    print(f"Saved summary: {output_dirs['results'] / 'summary.json'}")
    print(f"Saved confusion matrix: {output_dirs['results'] / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
