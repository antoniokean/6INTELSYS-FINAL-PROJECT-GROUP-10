from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import csv
import os
from pathlib import Path
import wave

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VALIDATION_FILE = DATA_DIR / "validation_list.txt"
TESTING_FILE = DATA_DIR / "testing_list.txt"
REMOVED_FILE = DATA_DIR / "removed_clips.txt"
MANIFEST_FILE = DATA_DIR / "dataset_manifest.csv"
INSPECTION_CACHE_FILE = DATA_DIR / "dataset_inspection_cache.csv"
SKIP_DIRS = {"_background_noise_"}
INSPECTION_RULESET_VERSION = "strict_v2"

EXPECTED_SAMPLE_RATE = 16000
MIN_DURATION_SECONDS = 0.80
TAIL_WINDOW_SECONDS = 0.04
SILENCE_RMS_THRESHOLD = 0.0025
TAIL_ACTIVITY_RMS_THRESHOLD = 0.08
TAIL_ACTIVITY_PEAK_THRESHOLD = 0.25
TAIL_TO_OVERALL_ENERGY_RATIO = 1.20
END_SCAN_SECONDS = 0.12
END_FRAME_SECONDS = 0.01
END_VOICED_FRAME_RMS_THRESHOLD = 0.02
MIN_TRAILING_SILENCE_FRAMES = 2
CLIPPED_SAMPLE_RATIO_THRESHOLD = 0.003
CLIPPED_SAMPLE_ABS_THRESHOLD = 0.995
DEFAULT_INSPECTION_WORKERS = min(16, max(4, os.cpu_count() or 4))


def normalize_path(path_str):
    normalized = path_str.strip().replace("\\", "/")

    if normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized.startswith("data/"):
        normalized = normalized[5:]

    return normalized


def load_path_set(file_path):
    with file_path.open("r", encoding="utf-8") as file:
        return {
            normalize_path(line)
            for line in file
            if normalize_path(line)
        }


def load_inspection_cache(cache_path):
    if not cache_path.exists():
        return {}

    cache = {}
    with cache_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rel_path = normalize_path(row.get("path", ""))
            if not rel_path:
                continue

            cache[rel_path] = {
                "size": row.get("size", ""),
                "mtime_ns": row.get("mtime_ns", ""),
                "ruleset_version": row.get("ruleset_version", ""),
                "status": row.get("status", "remove"),
                "removal_reason": row.get("removal_reason", "cache_miss"),
            }

    return cache


def save_inspection_cache(cache, cache_path):
    rows = [
        {
            "path": rel_path,
            "size": values["size"],
            "mtime_ns": values["mtime_ns"],
            "ruleset_version": values["ruleset_version"],
            "status": values["status"],
            "removal_reason": values["removal_reason"],
        }
        for rel_path, values in sorted(cache.items())
    ]

    with cache_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "path",
                "size",
                "mtime_ns",
                "ruleset_version",
                "status",
                "removal_reason",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def ensure_required_files_exist():
    required_files = [VALIDATION_FILE, TESTING_FILE, REMOVED_FILE]
    missing_files = [str(path) for path in required_files if not path.exists()]

    if missing_files:
        joined = "\n".join(missing_files)
        raise FileNotFoundError(f"Missing required input file(s):\n{joined}")


def get_inspection_workers():
    override = os.environ.get("DATA_PIPELINE_WORKERS", "").strip()
    if not override:
        return DEFAULT_INSPECTION_WORKERS

    try:
        return max(1, int(override))
    except ValueError:
        return DEFAULT_INSPECTION_WORKERS


def get_split(rel_path, validation_paths, testing_paths):
    in_validation = rel_path in validation_paths
    in_testing = rel_path in testing_paths

    if in_validation and in_testing:
        raise ValueError(f"Path appears in both validation and testing lists: {rel_path}")
    if in_validation:
        return "validation"
    if in_testing:
        return "testing"
    return "training"


def decode_audio_samples(raw_bytes, sample_width, channels):
    if sample_width == 1:
        samples = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
        samples -= 128.0
        scale = 128.0
    elif sample_width == 2:
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        scale = 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32)
        scale = 2147483648.0
    else:
        raise ValueError(f"unsupported_sample_width_{sample_width}")

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    return samples / scale


def inspect_wav_file(wav_path):
    try:
        with closing(wave.open(str(wav_path), "rb")) as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
    except Exception as error:
        error_name = type(error).__name__.lower()
        return "remove", f"unreadable_{error_name}"

    if frame_count <= 0:
        return "remove", "empty_clip"

    if sample_rate <= 0:
        return "remove", "invalid_sample_rate"

    duration_seconds = frame_count / sample_rate
    if duration_seconds < MIN_DURATION_SECONDS:
        return "remove", "short_clip"

    if sample_rate != EXPECTED_SAMPLE_RATE:
        return "remove", "unexpected_sample_rate"

    if sample_width not in {1, 2, 4}:
        return "remove", f"unsupported_sample_width_{sample_width}"

    try:
        with closing(wave.open(str(wav_path), "rb")) as wav_file:
            raw_audio = wav_file.readframes(frame_count)
    except Exception as error:
        error_name = type(error).__name__.lower()
        return "remove", f"unreadable_{error_name}"

    try:
        samples = decode_audio_samples(raw_audio, sample_width, channels)
    except ValueError as error:
        return "remove", str(error)

    if samples.size == 0:
        return "remove", "empty_clip"

    if not np.isfinite(samples).all():
        return "remove", "non_finite_samples"

    rms = float(np.sqrt(np.mean(np.square(samples))))
    peak = float(np.max(np.abs(samples)))
    if rms < SILENCE_RMS_THRESHOLD and peak < TAIL_ACTIVITY_PEAK_THRESHOLD:
        return "remove", "silent_clip"

    clipped_ratio = float(np.mean(np.abs(samples) >= CLIPPED_SAMPLE_ABS_THRESHOLD))
    if clipped_ratio >= CLIPPED_SAMPLE_RATIO_THRESHOLD:
        return "remove", "clipped_waveform"

    tail_window = max(1, int(sample_rate * TAIL_WINDOW_SECONDS))
    tail_samples = samples[-tail_window:]
    tail_rms = float(np.sqrt(np.mean(np.square(tail_samples))))
    tail_peak = float(np.max(np.abs(tail_samples)))

    if (
        tail_rms >= TAIL_ACTIVITY_RMS_THRESHOLD
        and tail_peak >= TAIL_ACTIVITY_PEAK_THRESHOLD
        and tail_rms >= max(rms, 1e-6) * TAIL_TO_OVERALL_ENERGY_RATIO
    ):
        return "remove", "speech_active_at_end"

    end_scan_window = max(1, int(sample_rate * END_SCAN_SECONDS))
    end_frame_window = max(1, int(sample_rate * END_FRAME_SECONDS))
    end_scan_samples = samples[-end_scan_window:]
    frame_count = max(1, len(end_scan_samples) // end_frame_window)
    trimmed_samples = end_scan_samples[-frame_count * end_frame_window :]
    frame_matrix = trimmed_samples.reshape(frame_count, end_frame_window)
    frame_rms = np.sqrt(np.mean(np.square(frame_matrix), axis=1))
    voiced_frames = frame_rms >= END_VOICED_FRAME_RMS_THRESHOLD

    if np.any(voiced_frames):
        last_voiced_index = int(np.max(np.flatnonzero(voiced_frames)))
        trailing_silence_frames = frame_count - last_voiced_index - 1
        if trailing_silence_frames < MIN_TRAILING_SILENCE_FRAMES:
            return "remove", "abrupt_end_after_speech"

    return "keep", "usable"


def inspect_wav_task(task):
    rel_path, wav_path = task
    status, removal_reason = inspect_wav_file(wav_path)
    return rel_path, status, removal_reason


def build_manifest(data_dir, validation_paths, testing_paths, removed_paths, inspection_cache):
    rows = []
    observed_paths = set()
    updated_cache = {}
    pending_tasks = []

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name in SKIP_DIRS:
            continue

        for wav_path in sorted(class_dir.glob("*.wav")):
            rel_path = normalize_path(str(wav_path.relative_to(data_dir)))
            observed_paths.add(rel_path)

            if rel_path in removed_paths:
                status = "remove"
                removal_reason = "listed_in_removed_clips"
            else:
                stat_info = wav_path.stat()
                size = str(stat_info.st_size)
                mtime_ns = str(stat_info.st_mtime_ns)
                cached_result = inspection_cache.get(rel_path)

                if (
                    cached_result
                    and cached_result["size"] == size
                    and cached_result["mtime_ns"] == mtime_ns
                    and cached_result.get("ruleset_version") == INSPECTION_RULESET_VERSION
                ):
                    status = cached_result["status"]
                    removal_reason = cached_result["removal_reason"]
                else:
                    status = None
                    removal_reason = None
                    pending_tasks.append((rel_path, wav_path))

                cache_entry = {
                    "size": size,
                    "mtime_ns": mtime_ns,
                    "ruleset_version": INSPECTION_RULESET_VERSION,
                    "status": status or "",
                    "removal_reason": removal_reason or "",
                }
                if status is not None:
                    updated_cache[rel_path] = cache_entry
            row = {
                "path": rel_path,
                "label": class_dir.name,
                "split": get_split(rel_path, validation_paths, testing_paths),
                "status": status,
                "removal_reason": removal_reason,
            }
            if rel_path not in removed_paths:
                row["cache_entry"] = cache_entry

            rows.append(row)

    if pending_tasks:
        worker_count = min(get_inspection_workers(), len(pending_tasks))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            inspected_results = {
                rel_path: (status, removal_reason)
                for rel_path, status, removal_reason in executor.map(
                    inspect_wav_task,
                    pending_tasks,
                )
            }

        for row in rows:
            if row["status"] is not None:
                continue

            status, removal_reason = inspected_results[row["path"]]
            row["status"] = status
            row["removal_reason"] = removal_reason
            row["cache_entry"]["status"] = status
            row["cache_entry"]["removal_reason"] = removal_reason
            updated_cache[row["path"]] = row["cache_entry"]

    for row in rows:
        row.pop("cache_entry", None)

    return rows, observed_paths, updated_cache


def save_manifest(rows, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["path", "label", "split", "status", "removal_reason"],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows, observed_paths, validation_paths, testing_paths, removed_paths):
    split_counts = Counter(row["split"] for row in rows)
    status_counts = Counter(row["status"] for row in rows)
    removal_reasons = Counter(
        row["removal_reason"]
        for row in rows
        if row["status"] == "remove"
    )

    missing_validation = validation_paths - observed_paths
    missing_testing = testing_paths - observed_paths
    missing_removed = removed_paths - observed_paths

    print(f"Manifest written to: {MANIFEST_FILE}")
    print(f"Total samples: {len(rows)}")
    print(
        "Split counts: "
        f"training={split_counts['training']}, "
        f"validation={split_counts['validation']}, "
        f"testing={split_counts['testing']}"
    )
    print(f"Status counts: keep={status_counts['keep']}, remove={status_counts['remove']}")
    if removal_reasons:
        reason_summary = ", ".join(
            f"{reason}={count}"
            for reason, count in sorted(removal_reasons.items())
        )
        print(f"Removal reasons: {reason_summary}")
    print(
        "Unmatched list entries: "
        f"validation={len(missing_validation)}, "
        f"testing={len(missing_testing)}, "
        f"removed={len(missing_removed)}"
    )


def main():
    ensure_required_files_exist()

    validation_paths = load_path_set(VALIDATION_FILE)
    testing_paths = load_path_set(TESTING_FILE)
    removed_paths = load_path_set(REMOVED_FILE)
    inspection_cache = load_inspection_cache(INSPECTION_CACHE_FILE)

    rows, observed_paths, updated_cache = build_manifest(
        DATA_DIR,
        validation_paths,
        testing_paths,
        removed_paths,
        inspection_cache,
    )
    save_manifest(rows, MANIFEST_FILE)
    save_inspection_cache(updated_cache, INSPECTION_CACHE_FILE)
    print_summary(rows, observed_paths, validation_paths, testing_paths, removed_paths)


if __name__ == "__main__":
    main()
