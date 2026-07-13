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
4. Watch it play (run locally, not in Colab -- Colab's headless and can't
   open a real display):
   ```
   python play.py --model dqn_model.zip --episodes 5
   ```
5. **Screen-record step 4** -- the submission requires a video of
   `play.py` actually running with the agent playing. Save it and either
   embed it below or link it (e.g. upload to the repo if small enough,
   or link a Drive/YouTube unlisted upload).

We don't need a separate report doc for this one -- the table, the
discussion, and the video all just go straight into this README.

## Hyperparameter Results

A's `learning_rate` sweep is done, filled in below. B and C's rows are
still TODO until their sweeps finish.

| Member | Hyperparameter Set | Noted Behavior |
|---|---|---|
| A | lr=0.01, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.05, eps_decay=0.1 | Reward pinned at -21.0 the whole run, zero variance across eval episodes -- learning rate too high, updates never stabilized. |
| A | lr=0.005, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.05, eps_decay=0.1 | Same as above -- flat -21.0, no learning. Still too high. |
| A | lr=0.001, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.05, eps_decay=0.1 | Still flat -21.0. Even an order of magnitude down from the top value, learning rate is still too high for this to converge in 200k steps. |
| A | lr=0.0007, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.05, eps_decay=0.1 | Flat -21.0 again -- no sign of the pattern breaking yet. |
| A | lr=0.0005, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.05, eps_decay=0.1 | Same flat -21.0 result. |
| A | lr=0.0003, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.05, eps_decay=0.1 | Last of the flat -21.0 runs -- everything above ~1e-4 failed to learn at all in our budget. |
| A | lr=0.0001, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.05, eps_decay=0.1 | First real break from -21: reward -17.8 +/- 2.14. Non-zero variance means the agent's actually behaving differently across episodes now, not stuck in one degenerate policy. This is our baseline lr. |
| A | lr=7e-05, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.05, eps_decay=0.1 | Similar to the run above: -18.0 +/- 1.79. Confirms the 5e-5 to 1e-4 range is roughly where this converges within 200k steps. |
| A | lr=5e-05, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.05, eps_decay=0.1 | Best result of the sweep: -17.2 +/- 3.25. Highest reward and highest variance -- looks like the most active learning of any run. |
| A | lr=1e-05, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.05, eps_decay=0.1 | Reward dropped back to -20.6 +/- 0.49. Learning rate this low is probably too slow to make much progress in only 200k steps. |
| B | *(TODO)* | *(TODO)* |
| C | *(TODO)* | *(TODO)* |

Overall pattern for `learning_rate`: anything from 0.01 down to 0.0003
failed to learn at all (flat -21.0, no variance). Real learning only
showed up once lr dropped to the 5e-5 to 1e-4 range, peaking around
5e-05. Going lower still (1e-05) started losing ground again -- too
slow to converge in the timestep budget we gave it. Classic too-high
(unstable) vs too-low (too slow) tradeoff.

### MlpPolicy vs CnnPolicy comparison

Same baseline hyperparameters (lr=0.0001, gamma=0.99, batch=32),
200,000 steps, only the policy/observation type differs:

| Policy | Observation | mean_reward |
|---|---|---|
| CnnPolicy | pixel frames (84x84, 4-stacked) | -17.40 +/- 2.06 |
| MlpPolicy | RAM vector (128,) | -21.00 +/- 0.00 |

CnnPolicy clearly wins here. MlpPolicy never broke away from -21 at
all (zero variance, same as our worst learning-rate runs) despite
running the exact config that worked well for CnnPolicy. Makes sense
given what the two are working with -- pixel frames have spatial
structure (ball position, paddle position, motion) that convolutions
are built to pick up on, while raw RAM bytes are just unstructured
memory values with no spatial relationship between neighboring
numbers. A plain MLP has a much harder time pulling anything useful
out of that in the same training budget.

## Discussion of Hyperparameter Tuning Results

*(TODO once the table above is filled in -- write up the patterns we
saw and why, not just what happened. Which axis moved the needle most?
Any surprises?)*

## Demo Video

*(TODO -- link or embed the `play.py` recording here once the final
model is trained.)*

## Contributions

*(TODO -- who ran which experiment IDs on which axis, who wrote which
part of this README.)*
