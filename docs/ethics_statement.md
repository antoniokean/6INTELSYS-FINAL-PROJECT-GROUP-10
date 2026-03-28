# Ethics Statement

## Overview

This project focuses on limited-vocabulary speech command recognition for low-risk Voice User Interface tasks. It is an academic and research-oriented system, not a production system for high-stakes deployment.

## Main Ethical Risks

### 1. Accent and speaking-style fairness

The model may perform unevenly across accents, dialects, and speaking styles. Since the system is trained on a public speech-command dataset, its performance may be stronger for some speech patterns than others. This can lead to misrecognition for users whose speech is less represented in the dataset.

### 2. Noise and environment sensitivity

Speech command systems may perform worse in noisy or uncontrolled environments. Background noise, microphone quality, and room conditions can affect prediction quality and may increase the likelihood of incorrect command recognition.

### 3. Privacy concerns with voice data

Voice data can be sensitive because speech may reveal identifying or personal information. Although this project uses a public dataset and does not collect new personal audio, privacy remains an important concern whenever voice data is involved.

### 4. Misinterpretation of commands

The system may confuse similar-sounding words such as `forward` and `four`. In a real application, this could lead to unintended actions or unreliable user experiences if predictions are accepted without additional safeguards.

## Mitigations

The project includes several design choices to reduce these risks:

- data augmentation to improve robustness under variation and noise
- balanced sampling and class-weighted loss in the main setup
- RL-based threshold tuning to reduce costly wrong decisions
- fallback or abstain behavior for low-confidence predictions
- class-level error analysis using the final test-set confusion matrix

## Privacy

This project uses the Google Speech Commands Dataset v0.02, a public dataset licensed under CC BY 4.0. The dataset notes indicate that participant age, gender, and location were not retained, and speakers are represented by anonymized identifiers. The project does not collect any new personal audio and does not add new participant recordings to the repository.

## Fairness and Slice Analysis

The project includes class-level error analysis through the final test-set confusion matrix. However, it does not include a separate demographic, speaker-based, accent-based, or noise-specific slice analysis. A formal fairness audit was not possible because the available dataset metadata does not provide those subgroup labels.

## Intended Use

This system is intended for:

- academic research
- low-risk speech command recognition experiments
- controlled demonstrations of Voice User Interface tasks

## Not Intended For

This system is not intended for:

- speaker identification
- surveillance
- legal, medical, or safety-critical use
- high-stakes deployment without additional validation

## Final Position

This project is acceptable as an academic speech-command recognition system because it uses a public dataset, documents its limitations, and includes safeguards such as threshold tuning and low-confidence rejection. However, its fairness, privacy, and robustness limits should be acknowledged clearly, and it should not be treated as a ready-for-deployment high-stakes system.
