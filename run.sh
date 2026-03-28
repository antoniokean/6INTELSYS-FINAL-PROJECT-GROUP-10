#!/usr/bin/env bash
set -e

python src/data_pipeline.py
python src/train.py --preset medium_end
python src/eval.py --preset medium_end

# Optional ablation runs:
# python src/train.py --preset medium_end --ablation no_augment
# python src/eval.py --preset medium_end --ablation no_augment
# python src/train.py --preset medium_end --ablation no_balance
# python src/eval.py --preset medium_end --ablation no_balance
