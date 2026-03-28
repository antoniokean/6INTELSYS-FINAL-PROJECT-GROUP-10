import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PRESETS = ("test", "low_end", "medium_end", "high_end")
ABLATION_CHOICES = ("none", "no_augment", "no_balance", "all")
PRESET_DESCRIPTIONS = {
    "test": "Quick sanity check on a tiny subset.",
    "low_end": "Small run for weaker CPU or RAM limits.",
    "medium_end": "Balanced default for normal laptops or desktops.",
    "high_end": "Longest run using the full dataset when available.",
}
PRESET_SAMPLE_COUNTS = {
    "test": "train=256, val=64, test=64",
    "low_end": "train=1000, val=200, test=200",
    "medium_end": "train=10000, val=1000, test=1000",
    "high_end": "train=full, val=full, test=full",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run data preprocessing, training, and evaluation in sequence."
    )
    parser.add_argument("--preset", choices=PRESETS, default=None)
    parser.add_argument("--ablation", choices=ABLATION_CHOICES, default="none")
    parser.add_argument("--skip-data-pipeline", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--no-open-files", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--strict-validation", action="store_true")
    return parser.parse_args()


def prompt_for_preset():
    default_preset = "medium_end"

    if not sys.stdin.isatty():
        print(f"No preset provided. Using default preset: {default_preset}")
        return default_preset

    print("Choose a run preset:")
    for index, preset_name in enumerate(PRESETS, start=1):
        print(
            f"  {index}. {preset_name} - {PRESET_DESCRIPTIONS[preset_name]} "
            f"[{PRESET_SAMPLE_COUNTS[preset_name]}]"
        )

    while True:
        choice = input(f"Preset [default: {default_preset}]: ").strip().lower()

        if not choice:
            return default_preset

        if choice.isdigit():
            preset_index = int(choice) - 1
            if 0 <= preset_index < len(PRESETS):
                return PRESETS[preset_index]
        elif choice in PRESETS:
            return choice

        print("Invalid preset. Enter 1-4 or a preset name.")


def run_command(command):
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def resolve_ablation_targets(ablation_name):
    if ablation_name == "all":
        return ["no_augment", "no_balance"]
    return [ablation_name]


def main():
    args = parse_args()
    preset = args.preset or prompt_for_preset()
    python_executable = sys.executable
    ablation_targets = resolve_ablation_targets(args.ablation)

    if not args.skip_data_pipeline:
        run_command([python_executable, "src/data_pipeline.py"])

    for ablation_name in ablation_targets:
        train_command = [
            python_executable,
            "src/train.py",
            "--preset",
            preset,
            "--ablation",
            ablation_name,
        ]
        if args.no_open_files:
            train_command.append("--no-open-files")
        if args.no_amp:
            train_command.append("--no-amp")
        if args.strict_validation:
            train_command.append("--strict-validation")
        run_command(train_command)

        if not args.skip_eval:
            eval_command = [
                python_executable,
                "src/eval.py",
                "--preset",
                preset,
                "--ablation",
                ablation_name,
            ]
            run_command(eval_command)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
