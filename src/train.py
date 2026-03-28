import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import librosa
import numpy as np
import torch
try:
    import soundfile as sf
except ImportError:
    sf = None
from models.cnn import SimpleCNN
from rl_agent import ThresholdAgentConfig, ThresholdTuningAgent, evaluate_threshold
from torch.cuda.amp import GradScaler, autocast
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_FILE = DATA_DIR / "dataset_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments"
DEFAULT_CONFIG_DIR = OUTPUT_DIR / "configs"
DEFAULT_LOG_DIR = OUTPUT_DIR / "logs"
DEFAULT_RESULTS_DIR = OUTPUT_DIR / "results"
DEFAULT_CACHE_DIR = DEFAULT_LOG_DIR / "feature_cache"
DEFAULT_NUM_WORKERS = min(4, max(1, (os.cpu_count() or 2) - 1))

SAMPLE_RATE = 16000
TARGET_DURATION_SECONDS = 1.0
TARGET_NUM_SAMPLES = int(SAMPLE_RATE * TARGET_DURATION_SECONDS)
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
FEATURE_CACHE_VERSION = "multi_logmel_v2"
BACKGROUND_NOISE_DIR = DATA_DIR / "_background_noise_"
DEFAULT_BACKGROUND_NOISE_PROB = 0.45
DEFAULT_GAIN_DB = 6.0
DEFAULT_SPEC_TIME_MASKS = 2
DEFAULT_SPEC_FREQ_MASKS = 2
DEFAULT_SPEC_TIME_MASK_RATIO = 0.12
DEFAULT_SPEC_FREQ_MASK_BINS = 8
BACKGROUND_NOISE_CACHE = None

BASE_TRAINING_CONFIG = {
    "epochs": 12,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "label_smoothing": 0.05,
    "seed": 2518392709,
    "device": "auto",
    "num_workers": DEFAULT_NUM_WORKERS,
    "log_every": 50,
    "no_feature_cache": False,
    "no_augment": False,
    "no_balanced_sampling": False,
    "time_shift_ms": 100.0,
    "noise_scale": 0.003,
    "background_noise_prob": DEFAULT_BACKGROUND_NOISE_PROB,
    "gain_db": DEFAULT_GAIN_DB,
    "spec_time_masks": DEFAULT_SPEC_TIME_MASKS,
    "spec_freq_masks": DEFAULT_SPEC_FREQ_MASKS,
    "spec_time_mask_ratio": DEFAULT_SPEC_TIME_MASK_RATIO,
    "spec_freq_mask_bins": DEFAULT_SPEC_FREQ_MASK_BINS,
    "class_weight_power": 0.35,
    "sampler_power": 0.5,
    "early_stop_patience": 5,
    "threshold_min": 0.35,
    "threshold_max": 0.95,
    "threshold_step": 0.05,
    "rl_episodes": 80,
    "abstain_cost": 1.0,
    "misclassification_cost": 5.0,
    "max_train_samples": None,
    "max_val_samples": None,
    "max_test_samples": None,
}

TRAINING_PRESETS = {
    "test": {
        "epochs": 1,
        "batch_size": 8,
        "num_workers": 0,
        "log_every": 5,
        "no_augment": True,
        "max_train_samples": 256,
        "max_val_samples": 64,
        "max_test_samples": 64,
    },
    "low_end": {
        "epochs": 5,
        "batch_size": 8,
        "num_workers": 0,
        "log_every": 10,
        "max_train_samples": 1000,
        "max_val_samples": 200,
        "max_test_samples": 200,
    },
    "medium_end": {
        "epochs": 15,
        "batch_size": 16,
        "num_workers": min(2, DEFAULT_NUM_WORKERS),
        "log_every": 20,
        "max_train_samples": 10000,
        "max_val_samples": 1000,
        "max_test_samples": 1000,
    },
    "high_end": {
        "epochs": 20,
        "batch_size": 32,
        "num_workers": DEFAULT_NUM_WORKERS,
        "log_every": 50,
        "max_train_samples": None,
        "max_val_samples": None,
        "max_test_samples": None,
    },
}

PRESET_DESCRIPTIONS = {
    "test": "Quick sanity check on a tiny subset.",
    "low_end": "Small run for weaker CPU or RAM limits.",
    "medium_end": "Balanced default for normal laptops or desktops.",
    "high_end": "Longer run using the full dataset when possible.",
}

PRESET_OUTPUT_DIR_NAMES = {
    "test": "test",
    "low_end": "low-end",
    "medium_end": "medium-end",
    "high_end": "high-end",
}
ABLATION_CHOICES = ("none", "no_augment", "no_balance")

# Edit the settings here
def parse_args():
    parser = argparse.ArgumentParser(description="Train a speech command CNN baseline.")
    parser.add_argument("--preset", choices=list(TRAINING_PRESETS), default=None)
    parser.add_argument("--ablation", choices=ABLATION_CHOICES, default="none")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--randomize-run", action="store_true")
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--no-open-files", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--no-feature-cache", action="store_true", default=None)
    parser.add_argument("--no-augment", action="store_true", default=None)
    parser.add_argument("--no-balanced-sampling", action="store_true", default=None)
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        help="Enable extra per-sample and per-batch finite-value checks for debugging.",
    )
    parser.add_argument("--time-shift-ms", type=float, default=None)
    parser.add_argument("--noise-scale", type=float, default=None)
    parser.add_argument("--background-noise-prob", type=float, default=None)
    parser.add_argument("--gain-db", type=float, default=None)
    parser.add_argument("--spec-time-masks", type=int, default=None)
    parser.add_argument("--spec-freq-masks", type=int, default=None)
    parser.add_argument("--spec-time-mask-ratio", type=float, default=None)
    parser.add_argument("--spec-freq-mask-bins", type=int, default=None)
    parser.add_argument("--class-weight-power", type=float, default=None)
    parser.add_argument("--sampler-power", type=float, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--threshold-min", type=float, default=None)
    parser.add_argument("--threshold-max", type=float, default=None)
    parser.add_argument("--threshold-step", type=float, default=None)
    parser.add_argument("--rl-episodes", type=int, default=None)
    parser.add_argument("--abstain-cost", type=float, default=None)
    parser.add_argument("--misclassification-cost", type=float, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser.parse_args()


def prompt_for_preset():
    default_preset = "medium_end"

    if not sys.stdin.isatty():
        print(f"No preset provided. Using default preset: {default_preset}")
        return default_preset

    preset_names = list(TRAINING_PRESETS)
    print("Choose a training preset:")
    for index, preset_name in enumerate(preset_names, start=1):
        print(f"  {index}. {preset_name} - {PRESET_DESCRIPTIONS[preset_name]}")

    while True:
        choice = input(f"Preset [default: {default_preset}]: ").strip().lower()

        if not choice:
            return default_preset

        if choice.isdigit():
            preset_index = int(choice) - 1
            if 0 <= preset_index < len(preset_names):
                return preset_names[preset_index]
        elif choice in TRAINING_PRESETS:
            return choice

        print("Invalid preset. Enter 1-4 or a preset name.")


def apply_preset(args):
    if args.preset is None:
        args.preset = prompt_for_preset()

    resolved = dict(BASE_TRAINING_CONFIG)
    resolved.update(TRAINING_PRESETS[args.preset])

    for key in resolved:
        value = getattr(args, key)
        if value is not None:
            resolved[key] = value

    for key, value in resolved.items():
        setattr(args, key, value)

    return args


def apply_ablation(args):
    if args.ablation == "no_augment":
        args.no_augment = True
    elif args.ablation == "no_balance":
        args.no_balanced_sampling = True
        args.class_weight_power = 0.0
        args.sampler_power = 0.0

    return args


def resolve_preset_output_dirs(base_output_dir, preset_name, ablation_name="none"):
    preset_dir_name = PRESET_OUTPUT_DIR_NAMES[preset_name]
    config_path = base_output_dir / "configs" / preset_dir_name
    log_path = base_output_dir / "logs" / preset_dir_name
    results_path = base_output_dir / "results" / preset_dir_name
    if ablation_name != "none":
        config_path = config_path / "ablations" / ablation_name
        log_path = log_path / "ablations" / ablation_name
        results_path = results_path / "ablations" / ablation_name

    return {
        "configs": config_path,
        "logs": log_path,
        "results": results_path,
    }


def resolve_cache_dir(args, output_dirs):
    if args.cache_dir == DEFAULT_CACHE_DIR:
        return output_dirs["logs"] / "feature_cache"
    return args.cache_dir


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False


def load_manifest(manifest_path):
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    rows = []
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["status"] != "keep":
                continue
            rows.append(row)

    if not rows:
        raise ValueError("Manifest contains no usable rows with status='keep'.")

    return rows


def build_label_map(rows):
    labels = sorted({row["label"] for row in rows})
    return {label: index for index, label in enumerate(labels)}


def split_rows(rows):
    splits = {"training": [], "validation": [], "testing": []}

    for row in rows:
        split = row["split"]
        if split not in splits:
            raise ValueError(f"Unexpected split value in manifest: {split}")
        splits[split].append(row)

    if not splits["training"] or not splits["validation"]:
        raise ValueError("Training and validation splits must both have at least one sample.")

    return splits


def build_label_counts(rows):
    counts = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    return counts


def build_loss_class_weights(label_map, rows, power):
    if power <= 0:
        return None

    label_counts = build_label_counts(rows)
    weights = np.ones(len(label_map), dtype=np.float32)
    max_count = max(label_counts.values())

    for label, index in label_map.items():
        count = label_counts.get(label, 1)
        weights[index] = (max_count / max(count, 1)) ** power

    weights = weights / max(float(weights.mean()), 1e-8)
    return torch.tensor(weights, dtype=torch.float32)


def build_weighted_train_sampler(rows, label_map, power):
    if power <= 0:
        return None

    label_counts = build_label_counts(rows)
    sample_weights = []

    for row in rows:
        count = label_counts.get(row["label"], 1)
        sample_weight = (1.0 / max(count, 1)) ** power
        sample_weights.append(sample_weight)

    weights = torch.tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=len(rows), replacement=True)


def limit_rows(rows, max_samples, seed, split_name):
    if max_samples is None or max_samples >= len(rows):
        return rows

    digest = hashlib.sha256(f"{seed}:{split_name}".encode("utf-8")).hexdigest()
    rng = random.Random(int(digest, 16))
    rows_by_label = {}

    for row in rows:
        rows_by_label.setdefault(row["label"], []).append(row)

    labels = list(rows_by_label)
    rng.shuffle(labels)

    for label in labels:
        rng.shuffle(rows_by_label[label])

    selected_rows = []
    while len(selected_rows) < max_samples:
        made_progress = False
        for label in labels:
            if rows_by_label[label]:
                selected_rows.append(rows_by_label[label].pop())
                made_progress = True
                if len(selected_rows) >= max_samples:
                    break
        if not made_progress:
            break
        rng.shuffle(labels)

    rng.shuffle(selected_rows)
    return selected_rows


def describe_device(device):
    if device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        return f"cuda ({torch.cuda.get_device_name(device_index)})"
    return str(device)


def resolve_device(device_choice):
    if device_choice == "cpu":
        return torch.device("cpu")

    if device_choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA-compatible GPU is available.")
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maybe_disable_amp_for_device(args, device):
    if device.type != "cuda" or args.no_amp:
        return

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_index)
    if "GTX 1650" in device_name.upper():
        args.no_amp = True
        print(f"AMP disabled automatically on {device_name} for stability.")


def maybe_reduce_num_workers_for_device(args, device):
    if args.num_workers <= 0:
        return

    if os.name != "nt" or device.type != "cuda":
        return

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_index)
    args.num_workers = 0
    print(
        "DataLoader workers reduced to 0 on "
        f"{device_name} under Windows to avoid worker crashes and paging-file errors."
    )


def build_loader_kwargs(device, num_workers):
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    return loader_kwargs


def serialize_args(args):
    serialized = {}
    for key, value in vars(args).items():
        serialized[key] = str(value) if isinstance(value, Path) else value
    return serialized


def build_threshold_agent_config(args):
    return ThresholdAgentConfig(
        min_threshold=args.threshold_min,
        max_threshold=args.threshold_max,
        step_size=args.threshold_step,
        episodes=args.rl_episodes,
        abstain_cost=args.abstain_cost,
        misclassification_cost=args.misclassification_cost,
        seed=args.seed,
    )


def compute_macro_f1(labels, predictions, num_classes):
    f1_scores = []

    for class_index in range(num_classes):
        true_positive = int(((labels == class_index) & (predictions == class_index)).sum())
        false_positive = int(((labels != class_index) & (predictions == class_index)).sum())
        false_negative = int(((labels == class_index) & (predictions != class_index)).sum())

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0

        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append((2 * precision * recall) / (precision + recall))

    return float(np.mean(f1_scores))


def load_audio(audio_path, target_num_samples=TARGET_NUM_SAMPLES):
    if sf is not None:
        audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    else:
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)

    if target_num_samples is not None:
        if len(audio) < target_num_samples:
            pad_width = target_num_samples - len(audio)
            audio = np.pad(audio, (0, pad_width))
        else:
            audio = audio[:target_num_samples]

    return np.asarray(audio, dtype=np.float32)


def normalize_feature_channel(channel):
    channel = channel.astype(np.float32)
    mean = float(channel.mean())
    std = float(channel.std())
    return (channel - mean) / max(std, 1e-6)


def summarize_numeric_array(array):
    return {
        "shape": tuple(array.shape),
        "min": float(np.nanmin(array)),
        "max": float(np.nanmax(array)),
        "mean": float(np.nanmean(array)),
        "std": float(np.nanstd(array)),
    }


def validate_numpy_array(name, array, sample_path):
    if not np.isfinite(array).all():
        stats = summarize_numeric_array(array)
        raise ValueError(
            f"Non-finite {name} for sample '{sample_path}'. "
            f"shape={stats['shape']} min={stats['min']:.6f} max={stats['max']:.6f} "
            f"mean={stats['mean']:.6f} std={stats['std']:.6f}"
        )


def is_valid_cached_feature(features):
    return (
        isinstance(features, np.ndarray)
        and features.ndim == 3
        and features.shape[0] == 3
        and features.shape[1] == N_MELS
        and np.isfinite(features).all()
    )


def summarize_tensor(tensor):
    tensor = tensor.detach().float()
    finite_mask = torch.isfinite(tensor)
    finite_values = tensor[finite_mask]
    if finite_values.numel() == 0:
        return {
            "shape": tuple(tensor.shape),
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
        }

    return {
        "shape": tuple(tensor.shape),
        "min": float(finite_values.min().item()),
        "max": float(finite_values.max().item()),
        "mean": float(finite_values.mean().item()),
        "std": float(finite_values.std(unbiased=False).item()) if finite_values.numel() > 1 else 0.0,
    }


def validate_tensor(name, tensor, phase_name, batch_index, sample_paths):
    if torch.isfinite(tensor).all():
        return

    stats = summarize_tensor(tensor)
    sample_paths = sample_paths or []
    sample_preview = ", ".join(sample_paths[:5]) if sample_paths else "<paths unavailable>"
    raise FloatingPointError(
        f"Non-finite {name} in phase '{phase_name}' batch {batch_index}. "
        f"samples=[{sample_preview}] shape={stats['shape']} min={stats['min']:.6f} "
        f"max={stats['max']:.6f} mean={stats['mean']:.6f} std={stats['std']:.6f}"
    )


def compute_logits_and_loss(
    model,
    features,
    labels,
    criterion,
    autocast_context,
    device,
    phase_name,
    batch_index,
    sample_paths,
):
    with autocast_context():
        logits = model(features)
        loss = criterion(logits, labels)

    if torch.isfinite(logits).all() and torch.isfinite(loss).all():
        return logits, loss

    if device.type == "cuda":
        retry_features = features.float().contiguous()
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            retry_autocast_context = torch.amp.autocast(device_type=device.type, enabled=False)
        else:
            retry_autocast_context = autocast(enabled=False)
        with retry_autocast_context:
            retry_logits = model(retry_features)
            retry_loss = criterion(retry_logits, labels)

        if torch.isfinite(retry_logits).all() and torch.isfinite(retry_loss).all():
            print(
                f"Recovered non-finite forward pass in phase '{phase_name}' batch {batch_index} "
                "by retrying in float32."
            )
            return retry_logits, retry_loss

        logits = retry_logits
        loss = retry_loss

    validate_tensor("logits", logits, phase_name, batch_index, sample_paths)
    validate_tensor("loss", loss, phase_name, batch_index, sample_paths)
    return logits, loss


def load_background_noises():
    global BACKGROUND_NOISE_CACHE

    if BACKGROUND_NOISE_CACHE is not None:
        return BACKGROUND_NOISE_CACHE

    background_clips = []
    if BACKGROUND_NOISE_DIR.exists():
        for wav_path in sorted(BACKGROUND_NOISE_DIR.glob("*.wav")):
            noise_audio = load_audio(wav_path, target_num_samples=None)
            if len(noise_audio) > 0:
                background_clips.append(noise_audio)

    BACKGROUND_NOISE_CACHE = background_clips
    return BACKGROUND_NOISE_CACHE


def mix_background_noise(audio, background_noises):
    if not background_noises:
        return audio

    noise_audio = random.choice(background_noises)
    if len(noise_audio) < TARGET_NUM_SAMPLES:
        repeats = int(np.ceil(TARGET_NUM_SAMPLES / max(len(noise_audio), 1)))
        noise_audio = np.tile(noise_audio, repeats)

    max_offset = max(0, len(noise_audio) - TARGET_NUM_SAMPLES)
    start_index = 0 if max_offset == 0 else random.randint(0, max_offset)
    noise_segment = noise_audio[start_index:start_index + TARGET_NUM_SAMPLES]

    signal_rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-7))
    noise_rms = float(np.sqrt(np.mean(np.square(noise_segment)) + 1e-7))
    target_snr_db = random.uniform(8.0, 24.0)
    target_noise_rms = signal_rms / (10 ** (target_snr_db / 20.0))
    scaled_noise = noise_segment * (target_noise_rms / max(noise_rms, 1e-7))
    return audio + scaled_noise.astype(np.float32)


def maybe_augment_audio(audio, max_shift_samples, noise_scale, background_noises, background_noise_prob, gain_db):
    if max_shift_samples > 0:
        shift = np.random.randint(-max_shift_samples, max_shift_samples + 1)
        if shift != 0:
            audio = np.roll(audio, shift)
            if shift > 0:
                audio[:shift] = 0.0
            else:
                audio[shift:] = 0.0

    if gain_db > 0:
        gain = 10 ** (random.uniform(-gain_db, gain_db) / 20.0)
        audio = audio * gain

    if background_noises and random.random() < background_noise_prob:
        audio = mix_background_noise(audio, background_noises)

    if noise_scale > 0:
        gaussian_scale = random.uniform(0.0, noise_scale)
        audio = audio + np.random.normal(0.0, gaussian_scale, size=audio.shape).astype(np.float32)

    return np.clip(audio, -1.0, 1.0)


def audio_to_log_mel(audio):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    log_mel = librosa.power_to_db(mel + 1e-10, ref=np.max).astype(np.float32)
    delta = librosa.feature.delta(log_mel).astype(np.float32)
    delta_delta = librosa.feature.delta(log_mel, order=2).astype(np.float32)
    return np.stack(
        [
            normalize_feature_channel(log_mel),
            normalize_feature_channel(delta),
            normalize_feature_channel(delta_delta),
        ],
        axis=0,
    )


def apply_spec_augment(features, time_masks, freq_masks, time_mask_ratio, freq_mask_bins):
    augmented = np.array(features, copy=True)
    _, num_mels, num_frames = augmented.shape

    max_time_mask = max(1, int(round(num_frames * max(0.0, time_mask_ratio))))
    max_freq_mask = max(1, int(freq_mask_bins))

    for _ in range(max(0, time_masks)):
        width = random.randint(0, min(max_time_mask, num_frames))
        if width == 0 or width >= num_frames:
            continue
        start_index = random.randint(0, num_frames - width)
        augmented[:, :, start_index:start_index + width] = 0.0

    for _ in range(max(0, freq_masks)):
        width = random.randint(0, min(max_freq_mask, num_mels))
        if width == 0 or width >= num_mels:
            continue
        start_index = random.randint(0, num_mels - width)
        augmented[:, start_index:start_index + width, :] = 0.0

    return augmented


def get_plotting_backend():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("Plot export skipped: matplotlib is not installed.")
        return None


def save_training_curves(history, output_path):
    plt = get_plotting_backend()
    if plt is None or not history:
        return

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]
    train_accuracy = [entry["train_accuracy"] for entry in history]
    val_accuracy = [entry["val_accuracy"] for entry in history]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss, label="Train Loss")
    axes[0].plot(epochs, val_loss, label="Val Loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_accuracy, label="Train Accuracy")
    axes[1].plot(epochs, val_accuracy, label="Val Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_sample_spectrogram(sample_rows, output_path):
    if not sample_rows:
        return

    plt = get_plotting_backend()
    if plt is None:
        return

    sample_count = len(sample_rows)
    num_columns = 5
    num_rows = 3

    figure, axes = plt.subplots(num_rows, num_columns, figsize=(4.2 * num_columns, 3.6 * num_rows))
    axes = np.atleast_1d(axes).ravel()
    active_axes = axes[:sample_count].tolist()
    last_image = None

    for index, (axis, sample_row) in enumerate(zip(axes, sample_rows)):
        audio = load_audio(DATA_DIR / sample_row["path"])
        features = audio_to_log_mel(audio)
        last_image = axis.imshow(features[0], aspect="auto", origin="lower", cmap="magma")
        file_name = Path(sample_row["path"]).name
        axis.set_title(f"{sample_row['label']} | {file_name}", fontsize=9)
        axis.set_xlabel("Time", fontsize=8)
        if index % num_columns == 0:
            axis.set_ylabel(f"{sample_row['label']}\nPitch", fontsize=8)
        else:
            axis.set_ylabel("Pitch", fontsize=8)

    for axis in axes[sample_count:]:
        axis.axis("off")

    figure.suptitle("Example Log-Mel Spectrograms from Data Used in This Run", fontsize=14)
    figure.text(
        0.5,
        0.02,
        "Each row shows five same-label examples selected after training. A strong representative clip is chosen first, then the row is filled with the closest matching spectrogram patterns from that label, using training clips only when the evaluated splits do not have enough matches.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.06, 0.9, 0.95))
    if last_image is not None:
        colorbar = figure.colorbar(
            last_image,
            ax=active_axes,
            format="%.1f",
            fraction=0.03,
            pad=0.02,
        )
        colorbar.set_label("Sound strength")
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def build_confusion_matrix(labels, predictions, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_index, predicted_index in zip(labels.astype(np.int64), predictions.astype(np.int64)):
        if 0 <= true_index < num_classes and 0 <= predicted_index < num_classes:
            matrix[true_index, predicted_index] += 1
    return matrix


def save_confusion_matrix(labels, predictions, label_map, output_path, normalized_output_path=None):
    if labels is None or predictions is None or len(labels) == 0 or len(predictions) == 0:
        return None

    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must contain the same number of samples.")

    plt = get_plotting_backend()
    if plt is None:
        return None

    index_to_label = {index: label for label, index in label_map.items()}
    ordered_labels = [index_to_label[index] for index in range(len(index_to_label))]
    matrix = build_confusion_matrix(labels, predictions, len(ordered_labels))

    def render_matrix(values, path, title, value_format):
        figure_size = max(10, min(18, len(ordered_labels) * 0.5))
        figure, axis = plt.subplots(figsize=(figure_size, figure_size))
        image = axis.imshow(values, interpolation="nearest", cmap="Blues", aspect="auto")
        axis.set_title(title)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks(np.arange(len(ordered_labels)))
        axis.set_yticks(np.arange(len(ordered_labels)))
        axis.set_xticklabels(ordered_labels, rotation=90, fontsize=8)
        axis.set_yticklabels(ordered_labels, fontsize=8)

        threshold = float(np.nanmax(values)) / 2.0 if np.size(values) else 0.0
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                display_value = "-" if np.isnan(value) else format(float(value), value_format)
                axis.text(
                    column_index,
                    row_index,
                    display_value,
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                    fontsize=6,
                )

        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
        figure.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(figure)

    render_matrix(matrix.astype(np.float32), output_path, "Confusion Matrix", ".0f")

    if normalized_output_path is not None:
        row_totals = matrix.sum(axis=1, keepdims=True)
        normalized_matrix = np.divide(
            matrix.astype(np.float32),
            np.maximum(row_totals, 1),
            out=np.zeros_like(matrix, dtype=np.float32),
            where=row_totals > 0,
        )
        render_matrix(normalized_matrix, normalized_output_path, "Confusion Matrix (Normalized)", ".2f")

    return {
        "labels": ordered_labels,
        "matrix": matrix.tolist(),
    }


def build_cache_path(cache_dir, relative_audio_path):
    return cache_dir / FEATURE_CACHE_VERSION / Path(relative_audio_path).with_suffix(".npy")


def save_cached_feature(cache_path, features):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.tmp.npy")
    np.save(temp_path, features)
    Path(temp_path).replace(cache_path)


class SpeechCommandsDataset(Dataset):
    def __init__(
        self,
        rows,
        label_map,
        cache_dir=None,
        save_feature_cache=False,
        augment=False,
        time_shift_ms=0.0,
        noise_scale=0.0,
        background_noise_prob=0.0,
        gain_db=0.0,
        spec_time_masks=0,
        spec_freq_masks=0,
        spec_time_mask_ratio=0.0,
        spec_freq_mask_bins=0,
        validate_samples=False,
        return_paths=False,
    ):
        self.rows = rows
        self.label_map = label_map
        self.cache_dir = cache_dir
        self.save_feature_cache = save_feature_cache and cache_dir is not None
        self.augment = augment
        self.max_shift_samples = int(SAMPLE_RATE * (time_shift_ms / 1000.0))
        self.noise_scale = noise_scale
        self.background_noise_prob = max(0.0, min(1.0, background_noise_prob))
        self.gain_db = max(0.0, gain_db)
        self.spec_time_masks = max(0, spec_time_masks)
        self.spec_freq_masks = max(0, spec_freq_masks)
        self.spec_time_mask_ratio = max(0.0, spec_time_mask_ratio)
        self.spec_freq_mask_bins = max(0, spec_freq_mask_bins)
        self.validate_samples = validate_samples
        self.return_paths = return_paths
        self.background_noises = load_background_noises() if augment and self.background_noise_prob > 0 else []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        audio_path = DATA_DIR / row["path"]
        cache_path = None

        if self.cache_dir is not None and not self.augment:
            cache_path = build_cache_path(self.cache_dir, row["path"])
            if cache_path.exists():
                features = np.load(cache_path).astype(np.float32)
                if not is_valid_cached_feature(features):
                    print(f"Rebuilding invalid feature cache entry: {cache_path}")
                    audio = load_audio(audio_path)
                    if self.validate_samples:
                        validate_numpy_array("audio", audio, row["path"])
                    features = audio_to_log_mel(audio)
                    if self.validate_samples:
                        validate_numpy_array("features", features, row["path"])
                    if self.save_feature_cache:
                        save_cached_feature(cache_path, features)
            else:
                audio = load_audio(audio_path)
                if self.validate_samples:
                    validate_numpy_array("audio", audio, row["path"])
                features = audio_to_log_mel(audio)
                if self.validate_samples:
                    validate_numpy_array("features", features, row["path"])
                if self.save_feature_cache:
                    save_cached_feature(cache_path, features)
        else:
            audio = load_audio(audio_path)
            if self.validate_samples:
                validate_numpy_array("audio", audio, row["path"])
            if self.augment:
                audio = maybe_augment_audio(
                    audio,
                    self.max_shift_samples,
                    self.noise_scale,
                    self.background_noises,
                    self.background_noise_prob,
                    self.gain_db,
                )
                if self.validate_samples:
                    validate_numpy_array("augmented audio", audio, row["path"])
            features = audio_to_log_mel(audio)
            if self.validate_samples:
                validate_numpy_array("features", features, row["path"])
            if self.augment:
                features = apply_spec_augment(
                    features,
                    self.spec_time_masks,
                    self.spec_freq_masks,
                    self.spec_time_mask_ratio,
                    self.spec_freq_mask_bins,
                )
                if self.validate_samples:
                    validate_numpy_array("augmented features", features, row["path"])

        features = torch.from_numpy(features)
        label = torch.tensor(self.label_map[row["label"]], dtype=torch.long)
        if self.return_paths:
            return features, label, row["path"]
        return features, label


def make_loader(rows, label_map, batch_size, shuffle, loader_kwargs, dataset_kwargs, sampler=None):
    dataset = SpeechCommandsDataset(rows, label_map, **dataset_kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        **loader_kwargs,
    )


def unpack_batch(batch):
    if len(batch) == 3:
        features, labels, sample_paths = batch
        return features, labels, list(sample_paths)
    features, labels = batch
    return features, labels, None


def move_features_to_device(features, device):
    features = features.to(device, non_blocking=True)
    if device.type == "cuda":
        features = features.contiguous(memory_format=torch.channels_last)
    return features


def get_autocast_context_factory(device, enabled):
    if not enabled:
        return nullcontext
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return lambda: torch.amp.autocast(device_type=device.type)
    return autocast


def run_epoch(model, loader, criterion, optimizer, scaler, device, phase_name, log_every, strict_validation=False):
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    if len(loader) == 0:
        raise ValueError(f"No batches available for phase '{phase_name}'.")

    use_amp = device.type == "cuda" and scaler is not None and scaler.is_enabled()

    for batch_index, batch in enumerate(loader, start=1):
        features, labels, sample_paths = unpack_batch(batch)
        features = move_features_to_device(features, device)
        labels = labels.to(device, non_blocking=True)
        if strict_validation:
            validate_tensor("input features", features, phase_name, batch_index, sample_paths)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        autocast_context = get_autocast_context_factory(device, use_amp)

        with torch.set_grad_enabled(is_training):
            logits, loss = compute_logits_and_loss(
                model,
                features,
                labels,
                criterion,
                autocast_context,
                device,
                phase_name,
                batch_index,
                sample_paths,
            )

            if is_training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

        predictions = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

        if log_every and (batch_index % log_every == 0 or batch_index == len(loader)):
            print(
                f"{phase_name}: batch {batch_index}/{len(loader)} | "
                f"loss={loss.item():.4f}"
            )

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def evaluate_loader_with_outputs(model, loader, criterion, device, phase_name, log_every, strict_validation=False):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    probability_batches = []
    label_batches = []

    if len(loader) == 0:
        raise ValueError(f"No batches available for phase '{phase_name}'.")

    use_amp = device.type == "cuda"

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            features, labels, sample_paths = unpack_batch(batch)
            features = move_features_to_device(features, device)
            labels = labels.to(device, non_blocking=True)
            if strict_validation:
                validate_tensor("input features", features, phase_name, batch_index, sample_paths)
            autocast_context = get_autocast_context_factory(device, use_amp)
            logits, loss = compute_logits_and_loss(
                model,
                features,
                labels,
                criterion,
                autocast_context,
                device,
                phase_name,
                batch_index,
                sample_paths,
            )
            probabilities = torch.softmax(logits, dim=1)
            if strict_validation:
                validate_tensor("probabilities", probabilities, phase_name, batch_index, sample_paths)
            predictions = probabilities.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            probability_batches.append(probabilities.cpu())
            label_batches.append(labels.cpu())

            if log_every and (batch_index % log_every == 0 or batch_index == len(loader)):
                print(
                    f"{phase_name}: batch {batch_index}/{len(loader)} | "
                    f"loss={loss.item():.4f}"
                )

    probabilities = torch.cat(probability_batches).numpy()
    labels = torch.cat(label_batches).numpy()
    predictions = probabilities.argmax(axis=1)

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "macro_f1": compute_macro_f1(labels, predictions, probabilities.shape[1]),
        "probabilities": probabilities,
        "labels": labels,
        "predictions": predictions,
    }


def load_previous_preview_metadata(results_dir):
    preview_metadata_path = results_dir / "preview_samples.json"
    if preview_metadata_path.exists():
        with preview_metadata_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    legacy_preview_metadata_path = results_dir / "preview_sample.json"
    if not legacy_preview_metadata_path.exists():
        return []

    with legacy_preview_metadata_path.open("r", encoding="utf-8") as file:
        preview = json.load(file)
    return [preview] if preview else []


def build_preview_entries(rows, eval_result, label_map):
    if not rows or eval_result is None:
        return []

    probabilities = eval_result["probabilities"]
    predictions = eval_result["predictions"]
    labels = eval_result["labels"]

    if len(rows) != len(predictions):
        raise ValueError("Preview row count does not match evaluation outputs.")

    index_to_label = {index: label for label, index in label_map.items()}
    entries = []

    for row, predicted_index, true_index, probability_vector in zip(rows, predictions, labels, probabilities):
        confidence = float(probability_vector[predicted_index])
        entries.append(
            {
                "path": row["path"],
                "label": row["label"],
                "split": row["split"],
                "confidence": confidence,
                "predicted_label": index_to_label[int(predicted_index)],
                "correct": bool(int(predicted_index) == int(true_index)),
            }
        )

    return entries


def preview_entry_quality(entry, split_priority):
    confidence = float(entry.get("confidence") or 0.0)
    correctness_bonus = 1.0 if entry.get("correct") is True else 0.0
    evaluated_bonus = 0.15 if entry.get("correct") is not None else 0.0
    split_bonus = max(0.0, 0.1 - (0.03 * split_priority.get(entry.get("split"), 3)))
    return confidence + correctness_bonus + evaluated_bonus + split_bonus


def rank_entries_by_similarity(entries, target_count, split_priority, cache=None, max_candidates=36):
    if not entries:
        return []

    if cache is None:
        cache = {}

    candidate_entries = sorted(
        entries,
        key=lambda entry: (
            not bool(entry.get("correct")),
            -preview_entry_quality(entry, split_priority),
            entry["path"],
        ),
    )[:max_candidates]
    feature_vectors = []

    for entry in candidate_entries:
        feature_key = entry["path"]
        if feature_key not in cache:
            audio = load_audio(DATA_DIR / entry["path"])
            cache[feature_key] = audio_to_log_mel(audio).reshape(-1).astype(np.float32)
        feature_vectors.append(cache[feature_key])

    feature_matrix = np.stack(feature_vectors, axis=0)
    row_norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
    normalized_features = feature_matrix / np.maximum(row_norms, 1e-8)
    pairwise_similarity = normalized_features @ normalized_features.T
    quality_scores = np.array(
        [preview_entry_quality(entry, split_priority) for entry in candidate_entries],
        dtype=np.float32,
    )

    if len(candidate_entries) <= target_count:
        selected_indices = list(range(len(candidate_entries)))
        anchor_index = int(np.argmax(quality_scores))
    else:
        anchor_scores = []
        for anchor_index in range(len(candidate_entries)):
            neighbor_order = np.argsort(-pairwise_similarity[anchor_index])
            neighbor_order = [index for index in neighbor_order if index != anchor_index]
            best_neighbors = neighbor_order[: max(0, target_count - 1)]
            cohesion_score = float(np.mean(pairwise_similarity[anchor_index, best_neighbors])) if best_neighbors else 0.0
            anchor_scores.append(cohesion_score + (0.05 * float(quality_scores[anchor_index])))

        anchor_index = int(np.argmax(anchor_scores))
        neighbor_order = np.argsort(-pairwise_similarity[anchor_index])
        neighbor_order = [index for index in neighbor_order if index != anchor_index]
        selected_indices = [anchor_index] + neighbor_order[: max(0, target_count - 1)]

    representative_path = candidate_entries[anchor_index]["path"]
    selected_entries = []
    for selected_index in selected_indices:
        selected_entry = dict(candidate_entries[selected_index])
        selected_entry["pattern_similarity"] = float(pairwise_similarity[anchor_index, selected_index])
        selected_entry["representative_path"] = representative_path
        selected_entries.append(selected_entry)

    selected_entries.sort(
        key=lambda entry: (
            -float(entry.get("pattern_similarity") or 0.0),
            not bool(entry.get("correct")),
            -preview_entry_quality(entry, split_priority),
            entry["path"],
        )
    )

    return selected_entries[:target_count]


def choose_preview_rows(preview_sources, fallback_rows, label_map, randomize_run, previous_preview=None, max_preview_samples=15):
    previous_paths = {item["path"] for item in (previous_preview or []) if "path" in item}
    previous_labels = {item["label"] for item in (previous_preview or []) if "label" in item}
    split_priority = {"testing": 0, "validation": 1, "training": 2}
    samples_per_label = 5
    labels_per_grid = min(3, max(1, max_preview_samples // samples_per_label))

    preview_entries = []
    for rows, eval_result in preview_sources:
        preview_entries.extend(build_preview_entries(rows, eval_result, label_map))

    if not preview_entries:
        return []

    entries_by_label = {}
    for entry in preview_entries:
        entries_by_label.setdefault(entry["label"], []).append(entry)

    for label_entries in entries_by_label.values():
        label_entries.sort(
            key=lambda entry: (
                not entry["correct"],
                -entry["confidence"],
                split_priority.get(entry["split"], 9),
                entry["path"],
            )
        )

    candidate_labels = list(entries_by_label)
    if randomize_run and previous_labels:
        fresh_labels = [label for label in candidate_labels if label not in previous_labels]
        if len(fresh_labels) >= labels_per_grid:
            candidate_labels = fresh_labels

    candidate_labels.sort(
        key=lambda label: (
            -sum(entry["correct"] for entry in entries_by_label[label][:samples_per_label]),
            -float(np.mean([entry["confidence"] for entry in entries_by_label[label][:samples_per_label]])),
            float(
                np.mean(
                    [
                        split_priority.get(entry["split"], 9)
                        for entry in entries_by_label[label][:samples_per_label]
                    ]
                )
            ),
            label,
        )
    )

    chosen_labels = candidate_labels[:labels_per_grid]
    preview_rows = []
    preview_feature_cache = {}

    for label in chosen_labels:
        label_entries = list(entries_by_label[label])
        if randomize_run and previous_paths:
            fresh_entries = [entry for entry in label_entries if entry["path"] not in previous_paths]
            if len(fresh_entries) >= samples_per_label:
                label_entries = fresh_entries

        fallback_candidates = [
            {
                "path": row["path"],
                "label": row["label"],
                "split": row["split"],
                "confidence": None,
                "predicted_label": None,
                "correct": None,
            }
            for row in fallback_rows
            if row["label"] == label
        ]

        combined_candidates = label_entries + [
            candidate for candidate in fallback_candidates if candidate["path"] not in {entry["path"] for entry in label_entries}
        ]

        if randomize_run and previous_paths:
            fresh_candidates = [entry for entry in combined_candidates if entry["path"] not in previous_paths]
            if len(fresh_candidates) >= samples_per_label:
                combined_candidates = fresh_candidates

        selected_entries = rank_entries_by_similarity(
            combined_candidates,
            samples_per_label,
            split_priority,
            cache=preview_feature_cache,
        )

        preview_rows.extend(selected_entries)

    return preview_rows[:max_preview_samples]


def open_image_file(path):
    resolved_path = Path(path).resolve()

    for _ in range(10):
        if resolved_path.exists():
            break
        time.sleep(0.2)
    else:
        return False

    try:
        if hasattr(os, "startfile"):
            os.startfile(str(resolved_path))  # type: ignore[attr-defined]
            return True

        if sys.platform == "darwin":
            subprocess.Popen(["open", str(resolved_path)])
            return True

        if shutil.which("xdg-open") is not None:
            subprocess.Popen(["xdg-open", str(resolved_path)])
            return True
    except OSError:
        pass

    if os.name == "nt":
        try:
            subprocess.Popen(["cmd", "/c", "start", "", str(resolved_path)], shell=False)
            return True
        except OSError:
            return False

    return False


def open_generated_images(*paths):
    opened_any = False

    for path in paths:
        if path is None:
            continue
        if open_image_file(path):
            print(f"Opened image: {path}")
            opened_any = True
            time.sleep(0.75)
        else:
            print(f"Could not open image automatically: {path}")

    return opened_any


def open_generated_files(paths, file_label="file"):
    opened_any = False

    for path in paths:
        if open_image_file(path):
            print(f"Opened {file_label}: {path}")
            opened_any = True
        else:
            print(f"Could not open {file_label} automatically: {path}")

    return opened_any


def maybe_run_eval_after_training(args, output_dirs):
    results_dir = output_dirs["results"]
    summary_path = results_dir / "summary.json"
    final_metrics_json = results_dir / "final_metrics.json"
    final_metrics_md = results_dir / "final_metrics.md"

    print(f"Summary file: {summary_path}")
    print(
        "Final metrics files are created by eval.py: "
        f"{final_metrics_json} | {final_metrics_md}"
    )
    command = [
        sys.executable,
        "src/eval.py",
        "--preset",
        args.preset,
        "--ablation",
        args.ablation,
        "--output-dir",
        str(args.output_dir),
    ]
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    if not args.no_open_files:
        open_generated_files([summary_path, final_metrics_md, final_metrics_json], "result file")


def save_outputs(output_dirs, model, label_map, history, summary, threshold_tuning, args, preview_rows, val_eval, test_eval):
    config_dir = output_dirs["configs"]
    log_dir = output_dirs["logs"]
    results_dir = output_dirs["results"]

    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), results_dir / "baseline_cnn.pt")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_map": label_map,
            "history": history,
            "summary": summary,
            "args": serialize_args(args),
        },
        results_dir / "baseline_cnn_checkpoint.pt",
    )

    with (results_dir / "label_map.json").open("w", encoding="utf-8") as file:
        json.dump(label_map, file, indent=2)

    with (log_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    with (results_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    confusion_matrix_path = results_dir / "confusion_matrix.png"
    confusion_matrix_normalized_path = results_dir / "confusion_matrix_normalized.png"
    confusion_matrix_json_path = results_dir / "confusion_matrix.json"
    confusion_source = test_eval if len(test_eval.get("labels", [])) > 0 else val_eval
    confusion_metadata = save_confusion_matrix(
        confusion_source.get("labels"),
        confusion_source.get("predictions"),
        label_map,
        confusion_matrix_path,
        normalized_output_path=confusion_matrix_normalized_path,
    )
    if confusion_metadata is not None:
        confusion_metadata["source_split"] = (
            "test" if confusion_source is test_eval and len(test_eval.get("labels", [])) > 0 else "validation"
        )
        with confusion_matrix_json_path.open("w", encoding="utf-8") as file:
            json.dump(confusion_metadata, file, indent=2)

    if preview_rows:
        preview_metadata = [
            {
                "path": preview_row["path"],
                "label": preview_row["label"],
                "split": preview_row["split"],
                "seed": args.seed,
                "confidence": preview_row.get("confidence"),
                "predicted_label": preview_row.get("predicted_label"),
                "correct": preview_row.get("correct"),
                "pattern_similarity": preview_row.get("pattern_similarity"),
                "representative_path": preview_row.get("representative_path"),
            }
            for preview_row in preview_rows
        ]

        with (results_dir / "preview_samples.json").open("w", encoding="utf-8") as file:
            json.dump(preview_metadata, file, indent=2)

        with (results_dir / "preview_sample.json").open("w", encoding="utf-8") as file:
            json.dump(preview_metadata[0], file, indent=2)

    if threshold_tuning is not None:
        with (results_dir / "threshold_tuning.json").open("w", encoding="utf-8") as file:
            json.dump(threshold_tuning, file, indent=2)

        with (log_dir / "rl_threshold_history.json").open("w", encoding="utf-8") as file:
            json.dump(threshold_tuning["history"], file, indent=2)

    with (config_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(serialize_args(args), file, indent=2)

    curves_path = log_dir / "training_curves.png"
    spectrogram_path = results_dir / "sample_log_mel.png"
    save_training_curves(history, curves_path)
    save_sample_spectrogram(preview_rows, spectrogram_path)

    return {
        "training_curves": curves_path,
        "sample_log_mel": spectrogram_path,
        "confusion_matrix": confusion_matrix_path if confusion_metadata is not None else None,
        "confusion_matrix_normalized": confusion_matrix_normalized_path if confusion_metadata is not None else None,
    }


def main():
    args = apply_ablation(apply_preset(parse_args()))
    seed_was_randomized = args.randomize_run
    if seed_was_randomized:
        args.seed = random.SystemRandom().randrange(0, (2**32) - 1)
    set_seed(args.seed)
    device = resolve_device(args.device)
    maybe_disable_amp_for_device(args, device)
    maybe_reduce_num_workers_for_device(args, device)
    output_dirs = resolve_preset_output_dirs(args.output_dir, args.preset, args.ablation)

    rows = load_manifest(args.manifest)
    label_map = build_label_map(rows)
    split_to_rows = split_rows(rows)

    split_to_rows["training"] = limit_rows(split_to_rows["training"], args.max_train_samples, args.seed, "training")
    split_to_rows["validation"] = limit_rows(split_to_rows["validation"], args.max_val_samples, args.seed, "validation")
    split_to_rows["testing"] = limit_rows(split_to_rows["testing"], args.max_test_samples, args.seed, "testing")
    train_label_counts = build_label_counts(split_to_rows["training"])
    use_balanced_sampling = not args.no_balanced_sampling
    train_sampler = (
        build_weighted_train_sampler(split_to_rows["training"], label_map, args.sampler_power)
        if use_balanced_sampling
        else None
    )
    loss_class_weights = build_loss_class_weights(label_map, split_to_rows["training"], args.class_weight_power)
    previous_preview = load_previous_preview_metadata(output_dirs["results"]) if seed_was_randomized else None
    preview_rows = []

    loader_kwargs = build_loader_kwargs(device, args.num_workers)
    cache_dir = resolve_cache_dir(args, output_dirs)
    use_feature_cache = not args.no_feature_cache
    use_train_augmentation = not args.no_augment

    if args.clear_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"Cleared feature cache for preset run: {cache_dir}")
    elif args.clear_cache:
        print(f"Feature cache already empty: {cache_dir}")
    else:
        print(f"Feature cache preserved for preset run: {cache_dir}")

    print(f"Loaded manifest: {args.manifest}")
    print(f"Preset: {args.preset} - {PRESET_DESCRIPTIONS[args.preset]}")
    if args.ablation != "none":
        print(f"Ablation: {args.ablation}")
    print(f"Config dir: {output_dirs['configs']}")
    print(f"Log dir: {output_dirs['logs']}")
    print(f"Results dir: {output_dirs['results']}")
    print(
        "Samples | "
        f"train={len(split_to_rows['training'])} "
        f"val={len(split_to_rows['validation'])} "
        f"test={len(split_to_rows['testing'])}"
    )
    print(f"Classes: {len(label_map)}")
    print(
        "Settings | "
        f"epochs={args.epochs} "
        f"batch_size={args.batch_size} "
        f"device={args.device} "
        f"workers={args.num_workers}"
    )
    print(f"Seed: {args.seed}{' (randomized)' if seed_was_randomized else ''}")
    print(
        "Threshold tuning | "
        f"range=({args.threshold_min:.2f}, {args.threshold_max:.2f}) "
        f"step={args.threshold_step:.2f} "
        f"episodes={args.rl_episodes} "
        f"abstain_cost={args.abstain_cost:.2f} "
        f"misclassification_cost={args.misclassification_cost:.2f}"
    )
    print(
        "Augmentation | "
        f"time_shift_ms={args.time_shift_ms:.1f} "
        f"noise_scale={args.noise_scale:.4f} "
        f"background_noise_prob={args.background_noise_prob:.2f} "
        f"gain_db={args.gain_db:.1f}"
    )
    print(
        "Balancing | "
        f"sampler={'on' if train_sampler is not None else 'off'} "
        f"class_weight_power={args.class_weight_power:.2f} "
        f"sampler_power={args.sampler_power:.2f} "
        f"min_class={min(train_label_counts.values())} "
        f"max_class={max(train_label_counts.values())}"
    )

    train_loader = make_loader(
        split_to_rows["training"],
        label_map,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=train_sampler,
        loader_kwargs=loader_kwargs,
        dataset_kwargs={
            "cache_dir": None if use_train_augmentation else cache_dir,
            "save_feature_cache": use_feature_cache and not use_train_augmentation,
            "augment": use_train_augmentation,
            "validate_samples": args.strict_validation,
            "time_shift_ms": args.time_shift_ms,
            "noise_scale": args.noise_scale,
            "background_noise_prob": args.background_noise_prob,
            "gain_db": args.gain_db,
            "spec_time_masks": args.spec_time_masks,
            "spec_freq_masks": args.spec_freq_masks,
            "spec_time_mask_ratio": args.spec_time_mask_ratio,
            "spec_freq_mask_bins": args.spec_freq_mask_bins,
            "return_paths": args.strict_validation,
        },
    )
    val_loader = make_loader(
        split_to_rows["validation"],
        label_map,
        batch_size=args.batch_size,
        shuffle=False,
        loader_kwargs=loader_kwargs,
        dataset_kwargs={
            "cache_dir": cache_dir,
            "save_feature_cache": use_feature_cache,
            "augment": False,
            "validate_samples": args.strict_validation,
            "return_paths": args.strict_validation,
        },
    )
    test_loader = make_loader(
        split_to_rows["testing"],
        label_map,
        batch_size=args.batch_size,
        shuffle=False,
        loader_kwargs=loader_kwargs,
        dataset_kwargs={
            "cache_dir": cache_dir,
            "save_feature_cache": use_feature_cache,
            "augment": False,
            "validate_samples": args.strict_validation,
            "return_paths": args.strict_validation,
        },
    )

    model = SimpleCNN(num_classes=len(label_map)).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing,
        weight=None if loss_class_weights is None else loss_class_weights.to(device),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not args.no_amp)
    else:
        scaler = GradScaler(enabled=device.type == "cuda" and not args.no_amp)

    print(f"Device: {describe_device(device)}")
    print(f"DataLoader workers: {args.num_workers}")
    print(f"Feature cache: {'disabled' if not use_feature_cache else cache_dir}")
    print(f"AMP: {'disabled' if args.no_amp or device.type != 'cuda' else 'enabled'}")
    print(f"Strict validation: {'enabled' if args.strict_validation else 'disabled'}")
    print("Starting training...")

    history = []
    best_val_accuracy = -1.0
    best_val_macro_f1 = -1.0
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            phase_name="train",
            log_every=args.log_every,
            strict_validation=args.strict_validation,
        )
        val_metrics = evaluate_loader_with_outputs(
            model,
            val_loader,
            criterion,
            device=device,
            phase_name="val",
            log_every=args.log_every,
            strict_validation=args.strict_validation,
        )
        scheduler.step(val_metrics["accuracy"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        is_better_model = (
            val_metrics["accuracy"] > best_val_accuracy
            or (
                np.isclose(val_metrics["accuracy"], best_val_accuracy)
                and val_metrics["macro_f1"] > best_val_macro_f1
            )
            or (
                np.isclose(val_metrics["accuracy"], best_val_accuracy)
                and np.isclose(val_metrics["macro_f1"], best_val_macro_f1)
                and val_metrics["loss"] < best_val_loss
            )
        )

        if is_better_model:
            best_val_accuracy = val_metrics["accuracy"]
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_val_loss = val_metrics["loss"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.early_stop_patience:
            print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_eval = evaluate_loader_with_outputs(
        model,
        val_loader,
        criterion,
        device=device,
        phase_name="val-final",
        log_every=args.log_every,
        strict_validation=args.strict_validation,
    )
    threshold_agent = ThresholdTuningAgent(build_threshold_agent_config(args))
    threshold_tuning = threshold_agent.fit(val_eval["probabilities"], val_eval["labels"])

    if len(split_to_rows["testing"]) > 0:
        test_eval = evaluate_loader_with_outputs(
            model,
            test_loader,
            criterion,
            device=device,
            phase_name="test",
            log_every=args.log_every,
            strict_validation=args.strict_validation,
        )
        tuned_test_metrics = evaluate_threshold(
            test_eval["probabilities"],
            test_eval["labels"],
            threshold_tuning["best_threshold"],
            args.abstain_cost,
            args.misclassification_cost,
        )
        baseline_test_metrics = evaluate_threshold(
            test_eval["probabilities"],
            test_eval["labels"],
            threshold=0.0,
            abstain_cost=args.abstain_cost,
            misclassification_cost=args.misclassification_cost,
        )
    else:
        test_eval = {
            "loss": None,
            "accuracy": None,
            "macro_f1": None,
            "probabilities": np.empty((0, len(label_map))),
            "labels": np.empty((0,), dtype=np.int64),
            "predictions": np.empty((0,), dtype=np.int64),
        }
        tuned_test_metrics = None
        baseline_test_metrics = None

    threshold_tuning["validation_loss"] = val_eval["loss"]
    threshold_tuning["validation_accuracy"] = val_eval["accuracy"]
    threshold_tuning["validation_macro_f1"] = val_eval["macro_f1"]
    threshold_tuning["test_baseline_metrics"] = baseline_test_metrics
    threshold_tuning["test_tuned_metrics"] = tuned_test_metrics
    threshold_tuning["test_cost_reduction"] = (
        None
        if tuned_test_metrics is None or baseline_test_metrics is None
        else baseline_test_metrics["expected_cost"] - tuned_test_metrics["expected_cost"]
    )
    threshold_tuning["test_macro_f1"] = None if len(split_to_rows["testing"]) == 0 else test_eval["macro_f1"]

    preview_sources = []
    if len(split_to_rows["testing"]) > 0:
        preview_sources.append((split_to_rows["testing"], test_eval))
    if len(split_to_rows["validation"]) > 0:
        preview_sources.append((split_to_rows["validation"], val_eval))

    preview_rows = choose_preview_rows(
        preview_sources,
        split_to_rows["training"],
        label_map,
        seed_was_randomized,
        previous_preview=previous_preview,
    )
    if preview_rows:
        preview_labels = ", ".join(sorted({row["label"] for row in preview_rows}))
        print(f"Preview comparison labels: {preview_labels}")

    summary = {
        "best_val_accuracy": best_val_accuracy,
        "final_validation_loss": val_eval["loss"],
        "final_validation_accuracy": val_eval["accuracy"],
        "final_validation_macro_f1": val_eval["macro_f1"],
        "final_test_loss": test_eval["loss"],
        "final_test_accuracy": test_eval["accuracy"],
        "final_test_macro_f1": None if len(split_to_rows["testing"]) == 0 else test_eval["macro_f1"],
        "num_classes": len(label_map),
        "num_training_samples": len(split_to_rows["training"]),
        "num_validation_samples": len(split_to_rows["validation"]),
        "num_testing_samples": len(split_to_rows["testing"]),
        "rl_threshold_tuning": {
            "best_threshold": threshold_tuning["best_threshold"],
            "validation_expected_cost_baseline": threshold_tuning["baseline_metrics"]["expected_cost"],
            "validation_expected_cost_tuned": threshold_tuning["best_metrics"]["expected_cost"],
            "validation_cost_reduction": threshold_tuning["cost_reduction"],
            "test_expected_cost_baseline": None if baseline_test_metrics is None else baseline_test_metrics["expected_cost"],
            "test_expected_cost_tuned": None if tuned_test_metrics is None else tuned_test_metrics["expected_cost"],
            "test_cost_reduction": threshold_tuning["test_cost_reduction"],
            "abstain_cost": args.abstain_cost,
            "misclassification_cost": args.misclassification_cost,
        },
    }

    image_paths = save_outputs(
        output_dirs,
        model,
        label_map,
        history,
        summary,
        threshold_tuning,
        args,
        preview_rows,
        val_eval,
        test_eval,
    )
    print(
        f"Best val_acc={best_val_accuracy:.4f} | "
        f"threshold={threshold_tuning['best_threshold']:.2f} "
        f"val_cost={threshold_tuning['best_metrics']['expected_cost']:.4f} "
        f"test_cost={tuned_test_metrics['expected_cost'] if tuned_test_metrics is not None else 'n/a'}"
    )
    print(f"Training complete. Outputs saved under: {output_dirs['results']}")
    maybe_run_eval_after_training(args, output_dirs)
    if not args.no_open_files:
        open_generated_images(
            image_paths["training_curves"],
            image_paths["sample_log_mel"],
            image_paths["confusion_matrix"],
            image_paths["confusion_matrix_normalized"],
        )


if __name__ == "__main__":
    main()
