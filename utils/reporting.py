from utils.metrics import relative_cost_reduction, summarize_training_history
import json


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def save_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write(text)


def build_evaluation_report(preset, summary, history, config):
    history_summary = summarize_training_history(history)
    threshold_tuning = summary.get("rl_threshold_tuning") or {}

    validation_baseline_cost = threshold_tuning.get("validation_expected_cost_baseline")
    validation_tuned_cost = threshold_tuning.get("validation_expected_cost_tuned")
    test_baseline_cost = threshold_tuning.get("test_expected_cost_baseline")
    test_tuned_cost = threshold_tuning.get("test_expected_cost_tuned")

    report = {
        "preset": preset,
        "epochs_completed": history_summary["epochs_completed"],
        "best_epoch": history_summary["best_epoch"],
        "best_validation_accuracy": summary.get("best_val_accuracy"),
        "final_validation_accuracy": summary.get("final_validation_accuracy"),
        "final_validation_macro_f1": summary.get("final_validation_macro_f1"),
        "final_test_accuracy": summary.get("final_test_accuracy"),
        "final_test_macro_f1": summary.get("final_test_macro_f1"),
        "generalization_gap": history_summary["generalization_gap"],
        "batch_size": config.get("batch_size"),
        "epochs_target": config.get("epochs"),
        "rl_threshold_tuning": {
            "best_threshold": threshold_tuning.get("best_threshold"),
            "validation_expected_cost_baseline": validation_baseline_cost,
            "validation_expected_cost_tuned": validation_tuned_cost,
            "validation_relative_reduction": relative_cost_reduction(validation_baseline_cost, validation_tuned_cost),
            "test_expected_cost_baseline": test_baseline_cost,
            "test_expected_cost_tuned": test_tuned_cost,
            "test_relative_reduction": relative_cost_reduction(test_baseline_cost, test_tuned_cost),
        },
    }

    lines = [
        f"# Evaluation Report: {preset}",
        "",
        "## Core Metrics",
        f"- Best validation accuracy: {summary.get('best_val_accuracy')}",
        f"- Final validation accuracy: {summary.get('final_validation_accuracy')}",
        f"- Final validation macro-F1: {summary.get('final_validation_macro_f1')}",
        f"- Final test accuracy: {summary.get('final_test_accuracy')}",
        f"- Final test macro-F1: {summary.get('final_test_macro_f1')}",
        "",
        "## Training Summary",
        f"- Epochs completed: {history_summary['epochs_completed']}",
        f"- Best epoch: {history_summary['best_epoch']}",
        f"- Generalization gap: {history_summary['generalization_gap']}",
        f"- Batch size: {config.get('batch_size')}",
        f"- Epoch target: {config.get('epochs')}",
    ]

    if threshold_tuning:
        lines.extend(
            [
                "",
                "## RL Threshold Tuning",
                f"- Best tuned threshold: {threshold_tuning.get('best_threshold')}",
                f"- Validation expected cost: {validation_baseline_cost} -> {validation_tuned_cost}",
                f"- Test expected cost: {test_baseline_cost} -> {test_tuned_cost}",
                f"- Validation relative reduction: {relative_cost_reduction(validation_baseline_cost, validation_tuned_cost)}",
                f"- Test relative reduction: {relative_cost_reduction(test_baseline_cost, test_tuned_cost)}",
            ]
        )

    return report, "\n".join(lines) + "\n"


def build_final_metrics_summary(preset, summary):
    threshold_tuning = summary.get("rl_threshold_tuning") or {}

    final_metrics = {
        "preset": preset,
        "final_validation_accuracy": summary.get("final_validation_accuracy"),
        "final_validation_macro_f1": summary.get("final_validation_macro_f1"),
        "final_test_accuracy": summary.get("final_test_accuracy"),
        "final_test_macro_f1": summary.get("final_test_macro_f1"),
        "num_classes": summary.get("num_classes"),
        "num_training_samples": summary.get("num_training_samples"),
        "num_validation_samples": summary.get("num_validation_samples"),
        "num_testing_samples": summary.get("num_testing_samples"),
        "best_threshold": threshold_tuning.get("best_threshold"),
        "validation_expected_cost_baseline": threshold_tuning.get("validation_expected_cost_baseline"),
        "validation_expected_cost_tuned": threshold_tuning.get("validation_expected_cost_tuned"),
        "validation_cost_reduction": threshold_tuning.get("validation_cost_reduction"),
        "test_expected_cost_baseline": threshold_tuning.get("test_expected_cost_baseline"),
        "test_expected_cost_tuned": threshold_tuning.get("test_expected_cost_tuned"),
        "test_cost_reduction": threshold_tuning.get("test_cost_reduction"),
    }

    lines = [
        f"# Final Metrics: {preset}",
        "",
        "## Core Results",
        f"- Final validation accuracy: {final_metrics['final_validation_accuracy']}",
        f"- Final validation macro-F1: {final_metrics['final_validation_macro_f1']}",
        f"- Final test accuracy: {final_metrics['final_test_accuracy']}",
        f"- Final test macro-F1: {final_metrics['final_test_macro_f1']}",
        "",
        "## Dataset Used",
        f"- Training samples: {final_metrics['num_training_samples']}",
        f"- Validation samples: {final_metrics['num_validation_samples']}",
        f"- Testing samples: {final_metrics['num_testing_samples']}",
        f"- Number of classes: {final_metrics['num_classes']}",
    ]

    if final_metrics["best_threshold"] is not None:
        lines.extend(
            [
                "",
                "## RL Threshold Tuning",
                f"- Best threshold: {final_metrics['best_threshold']}",
                f"- Validation expected cost: {final_metrics['validation_expected_cost_baseline']} -> {final_metrics['validation_expected_cost_tuned']}",
                f"- Validation cost reduction: {final_metrics['validation_cost_reduction']}",
                f"- Test expected cost: {final_metrics['test_expected_cost_baseline']} -> {final_metrics['test_expected_cost_tuned']}",
                f"- Test cost reduction: {final_metrics['test_cost_reduction']}",
            ]
        )

    return final_metrics, "\n".join(lines) + "\n"
