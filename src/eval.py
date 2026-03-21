import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.reporting import (
    build_evaluation_report,
    build_final_metrics_summary,
    load_json,
    save_json,
    save_text,
)


OUTPUT_DIR = PROJECT_ROOT / "experiments"

PRESET_DIR_NAMES = {
    "test": "test",
    "low_end": "low-end",
    "medium_end": "medium-end",
    "high_end": "high-end",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Read and summarize saved experiment results.")
    parser.add_argument("--preset", default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def normalize_preset_name(preset_name):
    normalized = preset_name.strip().lower().replace("-", "_")
    if normalized not in PRESET_DIR_NAMES:
        valid_names = ", ".join(sorted(set(PRESET_DIR_NAMES) | set(PRESET_DIR_NAMES.values())))
        raise ValueError(f"Unknown preset '{preset_name}'. Use one of: {valid_names}")
    return normalized


def prompt_for_preset():
    default_preset = "medium_end"

    if not sys.stdin.isatty():
        print(f"No preset provided. Using default preset: {default_preset}")
        return default_preset

    preset_names = list(PRESET_DIR_NAMES)
    print("Choose an evaluation preset:")
    for index, preset_name in enumerate(preset_names, start=1):
        print(f"  {index}. {preset_name}")

    while True:
        choice = input(f"Preset [default: {default_preset}]: ").strip().lower()

        if not choice:
            return default_preset

        if choice.isdigit():
            preset_index = int(choice) - 1
            if 0 <= preset_index < len(preset_names):
                return preset_names[preset_index]
        else:
            normalized = choice.replace("-", "_")
            if normalized in PRESET_DIR_NAMES:
                return normalized

        print("Invalid preset. Enter 1-4 or a preset name.")


def main():
    args = parse_args()
    if args.preset is None:
        args.preset = prompt_for_preset()
    args.preset = normalize_preset_name(args.preset)
    preset_dir = PRESET_DIR_NAMES[args.preset]
    summary_path = args.output_dir / "results" / preset_dir / "summary.json"
    history_path = args.output_dir / "logs" / preset_dir / "history.json"
    config_path = args.output_dir / "configs" / preset_dir / "run_config.json"
    results_dir = args.output_dir / "results" / preset_dir

    summary = load_json(summary_path)
    history = load_json(history_path)
    config = load_json(config_path)
    report, report_markdown = build_evaluation_report(args.preset, summary, history, config)
    final_metrics, final_metrics_markdown = build_final_metrics_summary(args.preset, summary)

    print(f"Preset: {report['preset']}")
    print(f"Summary file: {summary_path}")
    print(f"Config file: {config_path}")
    print(f"Epochs completed: {report['epochs_completed']}")
    print(f"Best epoch: {report['best_epoch']}")
    print(f"Best validation accuracy: {report['best_validation_accuracy']}")
    print(f"Final validation accuracy: {report['final_validation_accuracy']}")
    print(f"Final validation macro-F1: {report['final_validation_macro_f1']}")
    print(f"Final test accuracy: {report['final_test_accuracy']}")
    print(f"Final test macro-F1: {report['final_test_macro_f1']}")
    print(f"Generalization gap: {report['generalization_gap']}")
    print(f"Batch size: {report['batch_size']}")
    print(f"Epoch target: {report['epochs_target']}")

    threshold_tuning = report.get("rl_threshold_tuning") or {}
    if threshold_tuning.get("best_threshold") is not None:
        print(f"Best tuned threshold: {threshold_tuning.get('best_threshold')}")
        print(f"Validation expected cost: {threshold_tuning.get('validation_expected_cost_baseline')} -> {threshold_tuning.get('validation_expected_cost_tuned')}")
        print(f"Test expected cost: {threshold_tuning.get('test_expected_cost_baseline')} -> {threshold_tuning.get('test_expected_cost_tuned')}")
        print(f"Validation relative reduction: {threshold_tuning.get('validation_relative_reduction')}")
        print(f"Test relative reduction: {threshold_tuning.get('test_relative_reduction')}")

    save_json(results_dir / "evaluation_report.json", report)
    save_text(results_dir / "evaluation_report.md", report_markdown)
    save_json(results_dir / "final_metrics.json", final_metrics)
    save_text(results_dir / "final_metrics.md", final_metrics_markdown)
    print(f"Saved evaluation report: {results_dir / 'evaluation_report.json'}")
    print(f"Saved evaluation summary: {results_dir / 'evaluation_report.md'}")
    print(f"Saved final metrics: {results_dir / 'final_metrics.json'}")
    print(f"Saved final metrics summary: {results_dir / 'final_metrics.md'}")


if __name__ == "__main__":
    main()
