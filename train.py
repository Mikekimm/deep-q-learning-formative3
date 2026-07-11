"""
Final training script -- produces the submission artifact `dqn_model.zip`.

Run this once the group has picked a winning hyperparameter config from
the pooled results in results/experiments_log.csv. Defaults are the
shared baseline; override any value via CLI flags.

Usage:
    python train.py --learning_rate 1e-4 --gamma 0.99 --batch_size 32 \
        --exploration_initial_eps 1.0 --exploration_final_eps 0.05 \
        --exploration_fraction 0.1
"""

import argparse
import shutil

from shared_train import BASELINE_CONFIG, train_one_run


def parse_args():
    parser = argparse.ArgumentParser()
    for key, default in BASELINE_CONFIG.items():
        parser.add_argument(f"--{key}", type=type(default), default=default)
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = vars(args)
    row = train_one_run(overrides=overrides, run_name="final_model", member="final")
    shutil.copy(row["model_path"], "dqn_model.zip")
    print("Saved final model to dqn_model.zip")
    print(row)


if __name__ == "__main__":
    main()
