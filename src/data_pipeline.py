from collections import Counter
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VALIDATION_FILE = DATA_DIR / "validation_list.txt"
TESTING_FILE = DATA_DIR / "testing_list.txt"
REMOVED_FILE = DATA_DIR / "removed_clips.txt"
MANIFEST_FILE = DATA_DIR / "dataset_manifest.csv"
SKIP_DIRS = {"_background_noise_"}


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


def ensure_required_files_exist():
    required_files = [VALIDATION_FILE, TESTING_FILE, REMOVED_FILE]
    missing_files = [str(path) for path in required_files if not path.exists()]

    if missing_files:
        joined = "\n".join(missing_files)
        raise FileNotFoundError(f"Missing required input file(s):\n{joined}")


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


def build_manifest(data_dir, validation_paths, testing_paths, removed_paths):
    rows = []
    observed_paths = set()

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name in SKIP_DIRS:
            continue

        for wav_path in sorted(class_dir.glob("*.wav")):
            rel_path = normalize_path(str(wav_path.relative_to(data_dir)))
            observed_paths.add(rel_path)

            rows.append(
                {
                    "path": rel_path,
                    "label": class_dir.name,
                    "split": get_split(rel_path, validation_paths, testing_paths),
                    "status": "remove" if rel_path in removed_paths else "keep",
                }
            )

    return rows, observed_paths


def save_manifest(rows, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "label", "split", "status"])
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows, observed_paths, validation_paths, testing_paths, removed_paths):
    split_counts = Counter(row["split"] for row in rows)
    status_counts = Counter(row["status"] for row in rows)

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

    rows, observed_paths = build_manifest(
        DATA_DIR,
        validation_paths,
        testing_paths,
        removed_paths,
    )
    save_manifest(rows, MANIFEST_FILE)
    print_summary(rows, observed_paths, validation_paths, testing_paths, removed_paths)


if __name__ == "__main__":
    main()
