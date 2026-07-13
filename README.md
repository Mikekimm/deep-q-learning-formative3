# deep-q-learning-formative3

Our group's DQN Atari assignment -- training a DQN agent (Stable
Baselines3 + Gymnasium) to play Pong, running 30 hyperparameter
experiments (10 each), and evaluating the best model with `play.py`.

## Who's doing what

Each of us owns one hyperparameter axis and holds everything else at
the shared baseline. That way our 30 results are actually comparable --
no confounded variables -- and the "noted behavior" write-up is an
explanation, not a guess.

| Member | Axis | Notebook |
|---|---|---|
| A | `learning_rate` (10 values) | `notebooks/experiments_memberA_lr.ipynb` |
| B | `gamma` + `batch_size` (5 + 5) | `notebooks/experiments_memberB_gamma_batch.ipynb` |
| C | exploration params (`exploration_initial_eps`, `exploration_final_eps`, `exploration_fraction`, 10 combos) | `notebooks/experiments_memberC_epsilon.ipynb` |

Also on us: a one-off MlpPolicy vs CnnPolicy comparison the assignment
requires (`notebooks/policy_comparison_mlp_vs_cnn.ipynb`) -- not part of
anyone's 10, whoever picks it up just runs it once.

## What we locked in

We agreed on these up front so our 30 runs stay comparable -- don't
change them mid-sweep without checking with the group first (actual
values live in `shared_train.py`):

- **Game:** `ALE/Pong-v5`
- **Timesteps per run:** 200,000, same for every run
- **Seed:** 42
- **Baseline hyperparameters:** `BASELINE_CONFIG` in `shared_train.py`

If your Colab is too slow for 200k steps x 10 runs, flag it to the
group rather than quietly changing the number -- it has to stay
identical across everyone's runs or the comparison breaks.

## Running your experiments (Colab)

1. Open your notebook fresh from GitHub each session (not a cached tab)
   -- File → Open notebook → GitHub.
2. **Enable a GPU runtime**: Runtime → Change runtime type → T4 GPU.
   Massively faster than CPU, worth doing before you start.
3. Run the install cell, then the repo cell -- it clones fresh or resets
   to match GitHub exactly, so you're always on the latest code.
4. Run the smoke test cell by itself first -- confirms your session's
   actually working before committing hours to the real sweep. Don't
   Run All the whole notebook.
5. Run your real experiment loop. Each run automatically saves a model
   to `results/models/<run_name>.zip` and appends a row to
   `results/experiments_log.csv`.
6. After each run, jot down what we saw -- reward trend, stability,
   divergence -- we need this for the report's "noted behavior" column.
7. Optional: the live TensorBoard cell shows reward curves updating in
   real time while a run trains, if you'd rather watch than wait.

## Syncing results back

We're all appending to the same `results/experiments_log.csv`, so pull
before you push or we'll clobber each other's rows:
```
git add results/experiments_log.csv
git commit -m "Add <name> <axis> results"
git pull --no-rebase origin main
git push origin main
```

## Once all 30 runs (+ the MLP/CNN comparison) are in

1. Check `results/experiments_log.csv` has everyone's rows.
2. Pick the config with the best `mean_reward`.
3. Retrain it as our official submission artifact:
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
   Needs a real display, so it won't work in Colab (headless) -- run it
   locally instead.

## Report

Each of us writes the "why" for our own axis's results; we combine
these into one report with a shared intro/conclusion, the pooled
30-row hyperparameter table, and a contributions section (who ran which
experiment IDs, who wrote what).
