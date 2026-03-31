# Model Card: Speech Command Recognition with CNN + RL Threshold Tuning

## Model Details

- Residual squeeze-excitation CNN classifier with RL-based threshold tuning
- Uses 1-second, 16 kHz audio clips
- Input features are normalized log-Mel, delta, and delta-delta spectrograms
- Trained using AdamW with a ReduceLROnPlateau scheduler
- Official final preset: `medium_end`

## Intended Use

- Academic research and low-risk Voice User Interface tasks
- Not intended for surveillance, speaker identification, or high-stakes deployment

## Dataset

- Google Speech Commands Dataset v0.02 (Warden, 2018)
- Contains 105,829 total clips, 35 classes, and about 2,600 speakers

## Evaluation Data

- Evaluated on Google Speech Commands Dataset v0.02
- Uses the dataset's provided validation and testing lists
- Current manifest contains 105,829 clips across 35 classes
- 89,080 clips were kept and 16,749 were removed during filtering

## Training Data

- Uses the same cleaned Speech Commands dataset as the evaluation data
- Official `medium_end` run uses 10,000 training, 1,000 validation, and 1,000 testing samples
- Audio is converted into spectrogram-based features before training
- Main setup also uses augmentation, balanced sampling, and class-weighted loss

## License

- Creative Commons BY 4.0

## Input

- 1-second, 16 kHz audio converted into normalized log-Mel, delta, and delta-delta features

## Metrics

- Evaluation metrics include accuracy >= 90%, Macro-F1 >= 0.88, and RL cost reduction >= 20%

## Performance

- The model achieved 93.00% test accuracy and 92.91% test Macro-F1
- RL threshold tuning reduced expected test cost from 0.350 to 0.220

## Factors

- Performance can be affected by background noise
- Accent variation and speaking style can change results
- Microphone quality and recording conditions also matter

## Limitation

- Performance may drop under noise, accent variation, non-English speech, and more diverse real-world speaking conditions

## Ethical Considerations

- Accent and dialect bias may be present, reducing the accuracy for non-native speakers
- Voice data may be personal and sensitive by nature
- Formal demographic fairness auditing was not possible
- Fairness evaluation is limited to class-level metrics and confusion-matrix analysis

## Caveats and Recommendations

- Performance may drop in noisy or more varied real-world settings
- The model was trained only on short English command clips
- It should not be used for high-stakes deployment
- Low-confidence prediction handling should be kept in any prototype use
- Future work can focus on stronger robustness and better handling of similar-sounding commands
