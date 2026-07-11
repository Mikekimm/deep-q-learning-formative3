# DQN Atari Group Project

Repo: https://github.com/Mikekimm/deep-q-learning-formative3

3-person group assignment: train a DQN agent (Stable Baselines3 + Gymnasium)
on an Atari game, run 10 hyperparameter experiments per member (30 total),
and evaluate the best model with `play.py`.

## Role split

Each member owns one hyperparameter axis, holds everything else at the
shared baseline, and runs 10 experiments varying only their axis. This
keeps results comparable (no confounded variables) and makes the
"noted behavior" write-up a causal explanation instead of a guess.

| Member | Axis under test | Notebook |
|---|---|---|
| A | `learning_rate` (10 values) | `notebooks/experiments_memberA_lr.ipynb` |
| B | `gamma` + `batch_size` (5 + 5) | `notebooks/experiments_memberB_gamma_batch.ipynb` |
| C | exploration params (`exploration_initial_eps`, `exploration_final_eps`, `exploration_fraction`, 10 combos) | `notebooks/experiments_memberC_epsilon.ipynb` |

## Locked decisions

Do not change these mid-experiment -- see `GAME_ID`, `TOTAL_TIMESTEPS`,
`SEED`, `BASELINE_CONFIG` in `shared_train.py`. If a run's config differs
in more than the assigned axis, it can't be fairly compared to the others.

- **Game:** `ALE/Pong-v5`
- **Timesteps per run:** 200,000 (same for all 30 runs)
- **Seed:** 42
- **Baseline config:** see `BASELINE_CONFIG` in `shared_train.py`

If your compute is too slow for 200k steps x 10 runs, raise it with the
group before changing the number solo -- it has to stay identical across
everyone's runs.

## How to run your experiments (Colab)

1. Open your notebook.
2. Upload `shared_train.py` to the Colab session (or `git clone` this repo
   and `%cd` into it) so the import works.
3. Run the install cell, then run your experiment loop.
4. Each run automatically:
   - saves a model to `results/models/<run_name>.zip`
   - appends a row to `results/experiments_log.csv`
5. After each run, jot down what you observed (reward trend, stability,
   divergence, etc.) -- you'll need this for the report's "noted behavior"
   column.

## After all 30 runs are in

1. Pool `results/experiments_log.csv` from all three notebooks into one
   file (same schema, so this is just concatenating rows).
2. Pick the config with the best `mean_reward`.
3. Retrain it as the official submission artifact:
   ```
   python train.py --learning_rate <val> --gamma <val> --batch_size <val> \
       --exploration_initial_eps <val> --exploration_final_eps <val> \
       --exploration_fraction <val>
   ```
   This produces `dqn_model.zip` in the repo root.
4. Watch it play:
   ```
   python play.py --model dqn_model.zip --episodes 5
   ```

## Report

Each member writes the "why" for their axis's results; combine into one
report with a shared intro/conclusion, the pooled 30-row hyperparameter
table, and an individual-contributions section (who ran experiment IDs
1-10 on which axis, who wrote which section).
