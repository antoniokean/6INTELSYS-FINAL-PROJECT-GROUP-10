# `medium_end` Ablations

This folder contains ablation runs for the `medium_end` preset.

Commands:

```powershell
python .\run.py --preset medium_end --ablation no_augment --no-open-files
python .\run.py --preset medium_end --ablation no_balance --no-open-files
```

Subfolders:

- `no_augment/` stores the run with augmentation disabled
- `no_balance/` stores the run with balancing disabled

The main `medium_end` preset output stays in `experiments/results/medium-end/`.
