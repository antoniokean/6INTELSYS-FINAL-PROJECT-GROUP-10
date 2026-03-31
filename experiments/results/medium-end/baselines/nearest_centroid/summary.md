# Non-DL Baseline Summary

- Method: nearest_centroid_logmel
- Preset: medium_end
- Seed: 2518392709
- Fit source: training
- Training samples: 10000
- Validation samples: 1000
- Testing samples: 1000
- Validation accuracy: 0.16
- Validation Macro-F1: 0.14682095097635564
- Test accuracy: 0.133
- Test Macro-F1: 0.1269598012364739

This baseline uses the same fixed split and the same 3-channel log-Mel input
features as the CNN, but replaces the neural network with a nearest-centroid classifier.
