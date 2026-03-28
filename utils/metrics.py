import numpy as np


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


def summarize_training_history(history):
    if not history:
        return {
            "epochs_completed": 0,
            "best_epoch": None,
            "last_epoch": None,
            "best_val_accuracy": None,
            "best_val_loss": None,
            "last_train_accuracy": None,
            "last_val_accuracy": None,
            "generalization_gap": None,
        }

    best_epoch_entry = max(history, key=lambda entry: entry["val_accuracy"])
    last_epoch_entry = history[-1]

    return {
        "epochs_completed": len(history),
        "best_epoch": best_epoch_entry["epoch"],
        "last_epoch": last_epoch_entry["epoch"],
        "best_val_accuracy": best_epoch_entry["val_accuracy"],
        "best_val_loss": best_epoch_entry["val_loss"],
        "last_train_accuracy": last_epoch_entry["train_accuracy"],
        "last_val_accuracy": last_epoch_entry["val_accuracy"],
        "generalization_gap": last_epoch_entry["train_accuracy"] - last_epoch_entry["val_accuracy"],
    }


def relative_cost_reduction(baseline_cost, tuned_cost):
    if baseline_cost in (None, 0):
        return None
    if tuned_cost is None:
        return None
    return float((baseline_cost - tuned_cost) / baseline_cost)
