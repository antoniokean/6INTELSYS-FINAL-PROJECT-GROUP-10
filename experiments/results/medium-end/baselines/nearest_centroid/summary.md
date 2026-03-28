# Non-DL Baseline Summary

- Method: nearest_centroid_logmel
- Preset: medium_end
- Seed: 2518392709
- Fit source: training
- Training samples: 9995
- Validation samples: 1000
- Testing samples: 1000
- Validation accuracy: 0.111
- Validation Macro-F1: 0.10087187074705095
- Test accuracy: 0.131
- Test Macro-F1: 0.11306162708300536

This baseline uses the same fixed split and the same 3-channel log-Mel input
features as the CNN, but replaces the neural network with a nearest-centroid classifier.
