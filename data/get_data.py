import argparse
import shutil
from pathlib import Path

try:
    import kagglehub
except ImportError:
    kagglehub = None


DATA_DIR = Path(__file__).resolve().parent
README_PATH = DATA_DIR / "README.md"
DATASET_HANDLE = "mielvit16/cleaned-speech-commands-data-set-v0-02"
REQUIRED_FILES = [
    DATA_DIR / "validation_list.txt",
    DATA_DIR / "testing_list.txt",
]
REQUIRED_DIRS = [
    DATA_DIR / "_background_noise_",
]
MIN_LABEL_DIRS = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check, download, and extract the Speech Commands dataset."
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--dataset", default=DATASET_HANDLE)
    return parser.parse_args()


def get_label_dirs():
    return sorted(
        path.name
        for path in DATA_DIR.iterdir()
        if (
            path.is_dir()
            and not path.name.startswith(".")
            and path.name not in {"_background_noise_", "__pycache__"}
        )
    )


def get_missing_items():
    missing_items = []
    for path in REQUIRED_FILES + REQUIRED_DIRS:
        if not path.exists():
            missing_items.append(path.name)
    return missing_items


def dataset_is_ready():
    missing_items = get_missing_items()
    label_dirs = get_label_dirs()
    return len(missing_items) == 0 and len(label_dirs) >= MIN_LABEL_DIRS


def print_status():
    missing_items = get_missing_items()
    label_dirs = get_label_dirs()

    print("Speech Commands dataset helper")
    print(f"Data directory: {DATA_DIR}")
    print(f"Data guide: {README_PATH}")
    print(f"Detected label folders: {len(label_dirs)}")

    if missing_items:
        print("Missing required dataset items:")
        for item in missing_items:
            print(f" - {item}")

    if label_dirs:
        print("Detected labels:")
        for label in label_dirs:
            print(f" - {label}")

    if dataset_is_ready():
        print("Dataset appears ready for preprocessing and training.")
    else:
        print("Dataset appears incomplete.")


def download_dataset_with_kagglehub(dataset_handle):
    if kagglehub is None:
        raise RuntimeError(
            "kagglehub is not installed. Install it with 'python -m pip install kagglehub'."
        )

    print(f"Downloading dataset from KaggleHub: {dataset_handle}")
    return Path(kagglehub.dataset_download(dataset_handle))


def copy_downloaded_dataset(source_dir, target_dir):
    print(f"Restoring dataset files from: {source_dir}")
    for source_path in source_dir.iterdir():
        if source_path.name == "__pycache__":
            continue
        if source_path.name == "get_data.py":
            continue

        target_path = target_dir / source_path.name
        if source_path.is_dir():
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.copytree(source_path, target_path)
        else:
            shutil.copy2(source_path, target_path)


def main():
    args = parse_args()
    print_status()

    if dataset_is_ready() and not args.force_download:
        print("Nothing to download.")
        return

    if args.force_download:
        print("Force download requested.")
    else:
        print("Attempting to restore missing dataset files.")

    try:
        downloaded_path = download_dataset_with_kagglehub(args.dataset)
        copy_downloaded_dataset(downloaded_path, DATA_DIR)
    except Exception as error:
        print(f"Dataset download failed: {error}")
        print("Tip: authenticate KaggleHub first if your environment is not logged in to Kaggle.")
        print("Open data/README.md for the dataset source and setup details.")
        raise

    print("Download and extraction complete.")
    print_status()


if __name__ == "__main__":
    main()
