from dataclasses import asdict, dataclass
import random

import numpy as np


@dataclass
class ThresholdAgentConfig:
    min_threshold: float = 0.35
    max_threshold: float = 0.95
    step_size: float = 0.05
    episodes: int = 80
    steps_per_episode: int = 12
    learning_rate: float = 0.3
    discount_factor: float = 0.9
    epsilon: float = 0.25
    epsilon_decay: float = 0.97
    min_epsilon: float = 0.02
    abstain_cost: float = 1.0
    misclassification_cost: float = 5.0
    seed: int = 42


def build_threshold_grid(config):
    steps = int(round((config.max_threshold - config.min_threshold) / config.step_size))
    grid = [
        round(config.min_threshold + (index * config.step_size), 6)
        for index in range(steps + 1)
    ]
    if not grid:
        raise ValueError("Threshold grid is empty.")
    return grid


def apply_threshold(probabilities, threshold):
    max_probabilities = probabilities.max(axis=1)
    predicted_labels = probabilities.argmax(axis=1)
    thresholded_predictions = predicted_labels.copy()
    thresholded_predictions[max_probabilities < threshold] = -1
    return max_probabilities, predicted_labels, thresholded_predictions


def evaluate_threshold(probabilities, labels, threshold, abstain_cost, misclassification_cost):
    max_probabilities, predicted_labels, thresholded_predictions = apply_threshold(
        probabilities,
        threshold,
    )

    abstain_mask = thresholded_predictions == -1
    accepted_mask = ~abstain_mask
    correct_mask = thresholded_predictions == labels
    wrong_accepted_mask = accepted_mask & ~correct_mask

    total_samples = int(labels.shape[0])
    abstain_count = int(abstain_mask.sum())
    accepted_count = int(accepted_mask.sum())
    correct_count = int(correct_mask.sum())
    wrong_accepted_count = int(wrong_accepted_mask.sum())

    expected_cost = (
        (abstain_count * abstain_cost) +
        (wrong_accepted_count * misclassification_cost)
    ) / max(total_samples, 1)

    accepted_accuracy = 0.0
    if accepted_count > 0:
        accepted_accuracy = float((predicted_labels[accepted_mask] == labels[accepted_mask]).mean())

    return {
        "threshold": float(threshold),
        "expected_cost": float(expected_cost),
        "coverage": accepted_count / max(total_samples, 1),
        "abstain_rate": abstain_count / max(total_samples, 1),
        "accepted_accuracy": float(accepted_accuracy),
        "thresholded_accuracy": correct_count / max(total_samples, 1),
        "average_confidence": float(max_probabilities.mean()),
        "num_samples": total_samples,
        "num_abstained": abstain_count,
        "num_accepted": accepted_count,
        "num_wrong_accepted": wrong_accepted_count,
    }


class ThresholdEnvironment:
    ACTIONS = (-1, 0, 1)
    ACTION_NAMES = {
        -1: "decrease_threshold",
        0: "keep_threshold",
        1: "increase_threshold",
    }

    def __init__(self, probabilities, labels, config):
        if probabilities.ndim != 2:
            raise ValueError("probabilities must be a 2D array of shape [num_samples, num_classes].")
        if labels.ndim != 1:
            raise ValueError("labels must be a 1D array of shape [num_samples].")
        if probabilities.shape[0] != labels.shape[0]:
            raise ValueError("probabilities and labels must contain the same number of samples.")
        if probabilities.shape[0] == 0:
            raise ValueError("Cannot tune a threshold with zero samples.")

        self.probabilities = probabilities
        self.labels = labels
        self.config = config
        self.threshold_grid = build_threshold_grid(config)
        self._metric_cache = {}

    def num_states(self):
        return len(self.threshold_grid)

    def threshold_for_state(self, state_index):
        return float(self.threshold_grid[state_index])

    def metrics_for_state(self, state_index):
        if state_index not in self._metric_cache:
            self._metric_cache[state_index] = evaluate_threshold(
                self.probabilities,
                self.labels,
                self.threshold_for_state(state_index),
                self.config.abstain_cost,
                self.config.misclassification_cost,
            )
        return dict(self._metric_cache[state_index])

    def transition(self, state_index, action_index):
        next_state_index = state_index + self.ACTIONS[action_index]
        return int(np.clip(next_state_index, 0, self.num_states() - 1))

    def reward(self, current_state_index, next_state_index):
        current_metrics = self.metrics_for_state(current_state_index)
        next_metrics = self.metrics_for_state(next_state_index)
        return current_metrics["expected_cost"] - next_metrics["expected_cost"]


class ThresholdTuningAgent:
    def __init__(self, config=None):
        self.config = config or ThresholdAgentConfig()
        if self.config.step_size <= 0:
            raise ValueError("step_size must be greater than 0.")
        if self.config.max_threshold < self.config.min_threshold:
            raise ValueError("max_threshold must be greater than or equal to min_threshold.")
        if self.config.steps_per_episode <= 0:
            raise ValueError("steps_per_episode must be greater than 0.")

        self.rng = random.Random(self.config.seed)
        self.q_values = None
        self.threshold_grid = build_threshold_grid(self.config)

    def _choose_action(self, state_index, epsilon):
        if self.rng.random() < epsilon:
            return self.rng.randrange(len(ThresholdEnvironment.ACTIONS))
        return int(np.argmax(self.q_values[state_index]))

    def _serialize_q_values(self):
        return [[float(value) for value in row] for row in self.q_values]

    def fit(self, probabilities, labels):
        environment = ThresholdEnvironment(probabilities, labels, self.config)
        self.threshold_grid = environment.threshold_grid
        self.q_values = np.zeros(
            (environment.num_states(), len(ThresholdEnvironment.ACTIONS)),
            dtype=np.float32,
        )

        epsilon = self.config.epsilon
        best_state_index = environment.num_states() // 2
        best_metrics = environment.metrics_for_state(best_state_index)
        history = []

        for episode in range(1, self.config.episodes + 1):
            state_index = self.rng.randrange(environment.num_states())
            start_state_index = state_index
            episode_reward = 0.0
            episode_best_state_index = state_index
            episode_best_metrics = environment.metrics_for_state(state_index)

            for _ in range(self.config.steps_per_episode):
                action_index = self._choose_action(state_index, epsilon)
                next_state_index = environment.transition(state_index, action_index)
                reward = environment.reward(state_index, next_state_index)
                next_metrics = environment.metrics_for_state(next_state_index)

                best_future_q = float(np.max(self.q_values[next_state_index]))
                current_q = float(self.q_values[state_index, action_index])
                updated_q = current_q + self.config.learning_rate * (
                    reward + (self.config.discount_factor * best_future_q) - current_q
                )
                self.q_values[state_index, action_index] = updated_q

                episode_reward += reward

                if next_metrics["expected_cost"] < episode_best_metrics["expected_cost"]:
                    episode_best_state_index = next_state_index
                    episode_best_metrics = next_metrics

                if next_metrics["expected_cost"] < best_metrics["expected_cost"]:
                    best_state_index = next_state_index
                    best_metrics = next_metrics

                state_index = next_state_index

            history.append(
                {
                    "episode": episode,
                    "start_threshold": environment.threshold_for_state(start_state_index),
                    "end_threshold": environment.threshold_for_state(state_index),
                    "best_threshold_seen": environment.threshold_for_state(episode_best_state_index),
                    "best_expected_cost_seen": float(episode_best_metrics["expected_cost"]),
                    "episode_reward": float(episode_reward),
                    "epsilon": float(epsilon),
                }
            )

            epsilon = max(self.config.min_epsilon, epsilon * self.config.epsilon_decay)

        baseline_metrics = evaluate_threshold(
            probabilities,
            labels,
            threshold=0.0,
            abstain_cost=self.config.abstain_cost,
            misclassification_cost=self.config.misclassification_cost,
        )

        threshold_metrics = [
            environment.metrics_for_state(state_index)
            for state_index in range(environment.num_states())
        ]

        return {
            "config": asdict(self.config),
            "threshold_grid": [float(threshold) for threshold in self.threshold_grid],
            "baseline_metrics": baseline_metrics,
            "best_threshold": environment.threshold_for_state(best_state_index),
            "best_metrics": best_metrics,
            "cost_reduction": float(baseline_metrics["expected_cost"] - best_metrics["expected_cost"]),
            "threshold_metrics": threshold_metrics,
            "q_values": self._serialize_q_values(),
            "action_names": ThresholdEnvironment.ACTION_NAMES,
            "history": history,
        }
