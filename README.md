# GitHub Release v0.1

# Speech Command Recognition + RL Threshold Tuning (Audio + RL)

- **Task:**
  - Classify simple commands from audio spectrograms
  - Use RL to optimize the operating threshold under asymmetric errors
- **MVP:** Spectrogram + CNN; RL agent tuning decision threshold.
- **Metrics:** Accuracy, macro-F1, expected cost.
- **Stretch:** Noise robustness; on-device inference.
- **Ethics:** Accessibility, accent/dialect fairness, consent for audio.

## Requirements
![Python](https://img.shields.io/badge/python-3.10-blue)
- **Libraries:** TensorFlow / PyTorch, NumPy, Librosa, Matplotlib, Scikit-learn
