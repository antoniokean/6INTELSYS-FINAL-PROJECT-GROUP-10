# Model Card

## Model Summary

This project implements a speech command recognition system for short spoken command words. The system uses a residual squeeze-excitation CNN to classify spectrogram-based audio features, followed by a Q-learning RL threshold-tuning step to improve cost-sensitive decision-making.

The official final reporting preset for this repository is `medium_end`.

## Model Details

- **Model type:** CNN classifier with residual and squeeze-excitation components, plus an RL threshold agent
- **Input:** 1-second, 16 kHz audio clips
- **Features:** normalized log-Mel, delta, and delta-delta spectrogram channels with 64 Mel bins
- **Output:** one of 34 speech command classes
- **Optimizer:** AdamW
- **Scheduler:** ReduceLROnPlateau
- **Official preset:** `medium_end`

## Dataset

- **Dataset:** Google Speech Commands Dataset v0.02
- **License:** CC BY 4.0
- **Size:** 105,829 audio clips, 34 classes, approximately 2,600 speakers
- **Official run subset:** 10,000 training samples, 1,000 validation samples, and 1,000 testing samples
- **Cleaning:** 252 invalid clips were excluded through the dataset manifest
- **Splits:** dataset-provided validation and testing lists, with remaining usable clips assigned to training

## Intended Use

This model is intended for limited-vocabulary speech command recognition in low-risk Voice User Interface tasks, academic research, and controlled demonstrations. Example use cases include simple command recognition such as device navigation or lightweight command triggering in offline prototypes.

## Out-of-Scope Use

This model should not be used for:

- speaker identification
- surveillance
- medical, legal, or other high-stakes decision-making
- open-ended speech understanding
- safety-critical deployment without additional validation

## Training Setup

The official `medium_end` run used the following setup:

- 15 epochs
- batch size of 16
- learning rate of 0.001
- weight decay of 0.0001
- label smoothing of 0.05
- fixed seed `2518392709`
- feature caching enabled
- AMP disabled
- automatic device selection

The pipeline also supports time shifting, background-noise mixing, Gaussian noise, SpecAugment masking, balanced sampling, and class-weighted loss.

## Performance

The current official `medium_end` run achieved:

- **Validation Accuracy:** 92.70%
- **Validation Macro-F1:** 92.70%
- **Test Accuracy:** 91.60%
- **Test Macro-F1:** 91.59%
- **Best Threshold:** 0.80
- **Validation Expected Cost:** 0.365 -> 0.209
- **Test Expected Cost:** 0.420 -> 0.257
- **Test Cost Reduction:** 0.163

These values are based on:

- `experiments/results/medium-end/final_metrics.json`
- `experiments/results/medium-end/evaluation_report.json`

## Error Analysis

The main class-level error analysis in this project is the final test-set confusion matrix. One of the more visible confusion patterns is between similar-sounding commands such as `forward` and `four`. The project does not include a separate demographic, speaker-based, or accent-based slice analysis.

## Ethical Considerations

Important ethical considerations for this model include:

- reduced performance under accent and dialect variation
- weaker performance in noisy environments
- fairness limitations due to the lack of demographic metadata in the dataset
- privacy concerns that naturally come with voice data, even though this project uses a public dataset and does not collect new personal audio

The dataset notes indicate that participant age, gender, and location were not retained, and speakers are represented by anonymized identifiers.

## Limitations

This model has several practical limitations:

- it was trained only on short English command clips
- performance may decrease under heavy noise, accent variation, and different speaking styles
- it does not perform general speech recognition or language understanding
- it is best suited for controlled or low-risk settings
- hardware limits may affect training speed and reproducibility across devices

## Deployment Guidance

If this model is deployed in a prototype or low-risk application, it should:

- validate performance in the target acoustic environment
- use a fallback or abstain mechanism for low-confidence predictions
- avoid high-stakes use
- be retested if the input device, microphone quality, or environment changes

## Reproduction

Recommended quick-start commands:

```powershell
cd .\data
python .\get_data.py
cd ..
python .\src\data_pipeline.py
python .\run.py --preset medium_end --no-open-files
python .\src\eval.py --preset medium_end
```
