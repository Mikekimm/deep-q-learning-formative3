# deep-q-learning-formative3

Repo: https://github.com/Mikekimm/deep-q-learning-formative3

Our group's DQN Atari assignment -- we trained a DQN agent (Stable
Baselines3 + Gymnasium) to play Pong, ran 30 hyperparameter experiments
(10 each), and evaluated the best model with `play.py`.

## Who did what

Each of us owned one hyperparameter axis and held everything else at
the shared baseline. That way our 30 results were actually comparable --
no confounded variables -- and the "noted behavior" write-up is an
explanation, not a guess.

| Member | Axis | Notebook |
|---|---|---|
| A | `learning_rate` (10 values) | `notebooks/experiments_memberA_lr.ipynb` |
| B | `gamma` + `batch_size` (5 + 5) | `notebooks/experiments_memberB_gamma_batch.ipynb` |
| C | exploration params (`exploration_initial_eps`, `exploration_final_eps`, `exploration_fraction`, 10 combos) | `notebooks/experiments_memberC_epsilon.ipynb` |

We also ran a one-off MlpPolicy vs CnnPolicy comparison the assignment
required (`notebooks/policy_comparison_mlp_vs_cnn.ipynb`) -- not part of
anyone's 10.

## What we locked in

We agreed on these up front so our 30 runs would stay comparable --
nobody changed them mid-sweep without checking with the group first
(actual values live in `shared_train.py`):

- **Game:** `ALE/Pong-v5`
- **Timesteps per run:** 200,000, same for every run
- **Seed:** 42
- **Baseline hyperparameters:** `BASELINE_CONFIG` in `shared_train.py`

We'd agreed that if anyone's Colab was too slow for the full 200k
steps x 10 runs, they'd flag it to the group rather than quietly
changing the number -- it had to stay identical across everyone's runs
or the comparison would break.

## How we ran our experiments (Colab)

Each of us opened our notebook fresh from GitHub every session (never
a cached tab) via File → Open notebook → GitHub, and enabled a GPU
runtime (Runtime → Change runtime type → T4 GPU) since it was
massively faster than CPU. After the install cell and the repo cell
(which cloned fresh or reset to match GitHub exactly, so we were
always on the latest code), we ran a smoke test cell by itself first
to confirm the session was actually working before committing hours
to a real sweep -- never Run All on the whole notebook. Then we ran
the real experiment loop; each run automatically saved a model to
`results/models/<run_name>.zip` and appended a row to
`results/experiments_log.csv`. After each run, we jotted down what we
saw -- reward trend, stability, divergence -- for the "noted behavior"
column. A live TensorBoard cell let us watch reward curves update in
real time while a run trained, for whoever preferred watching over
waiting.

## Syncing results back

We were all appending to the same `results/experiments_log.csv`, so we
pulled before pushing to avoid clobbering each other's rows:
```
git add results/experiments_log.csv
git commit -m "Add <name> <axis> results"
git pull --no-rebase origin main
git push origin main
```

## Once all 30 runs (+ the MLP/CNN comparison) were in

1. We checked `results/experiments_log.csv` had everyone's rows.
2. We picked the config with the best `mean_reward` -- gamma=0.95
   (Member B's sweep), which beat every other config across all 30 runs.
3. Retrained it as our official submission artifact:
   ```
   python train.py --learning_rate 0.0001 --gamma 0.95 --batch_size 32 \
       --exploration_initial_eps 1.0 --exploration_final_eps 0.05 \
       --exploration_fraction 0.1
   ```
   This produced `dqn_model.zip` in the repo root.
4. Watched it play (run locally, not Colab -- Colab's headless and
   can't open a real display):
   ```
   python play.py --model dqn_model.zip --episodes 5
   ```
5. Screen-recorded that run -- the submission requires a video of
   `play.py` actually running with the agent playing.

We didn't need a separate report doc for this one -- the table, the
discussion, and the video all went straight into this README.

## Hyperparameter Results

All 30 experiments are done -- filled in below.

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
| B | lr=0.0001, gamma=0.999, batch=32, eps_start=1.0, eps_end=0.05, eps_decay=0.1 | Reward -18.0 +/- 2.37. Discount factor this close to 1 weighs distant future rewards almost as much as immediate ones -- harder credit assignment over that long a horizon, landing below baseline. |
| B | lr=0.0001, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.05, eps_decay=0.1 | Reward -17.8 +/- 2.14. This is the shared baseline config -- our reference point for the rest of the gamma sweep. |
| B | lr=0.0001, gamma=0.98, batch=32, eps_start=1.0, eps_end=0.05, eps_decay=0.1 | Reward -18.4 +/- 2.06. Slightly below baseline -- a small step down from 0.99 didn't help, if anything it shortened the effective planning horizon a bit too much. |
| B | lr=0.0001, gamma=0.95, batch=32, eps_start=1.0, eps_end=0.05, eps_decay=0.1 | Reward -15.4 +/- 2.42. Best result of the entire 30-experiment sweep. A shorter effective horizon than baseline seems to make credit assignment easier in Pong -- reward arrives within a few hundred steps of the action that caused it, so discounting more aggressively doesn't lose much signal while making learning more stable. |
| B | lr=0.0001, gamma=0.90, batch=32, eps_start=1.0, eps_end=0.05, eps_decay=0.1 | Reward -17.8 +/- 1.17. Pushing the discount factor down further stopped helping -- back to baseline-level reward, though with noticeably lower variance than any other gamma value we tried. |
| B | lr=0.0001, gamma=0.99, batch=8, eps_start=1.0, eps_end=0.05, eps_decay=0.1 | Reward -20.6 +/- 0.49. Worst result of our sweep. A batch this small gives very noisy gradient estimates each update -- barely better than a flat -21 run, though the low variance suggests it's consistently bad rather than occasionally getting lucky. |
| B | lr=0.0001, gamma=0.99, batch=16, eps_start=1.0, eps_end=0.05, eps_decay=0.1 | Reward -18.2 +/- 1.33. Doubling the batch size from the row above helps noticeably -- less noisy gradients start showing up in the reward. |
| B | lr=0.0001, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.05, eps_decay=0.1 | Reward -17.8 +/- 2.14. Shared baseline batch size -- matches the gamma sweep's baseline row exactly, as expected since it's the identical config. |
| B | lr=0.0001, gamma=0.99, batch=64, eps_start=1.0, eps_end=0.05, eps_decay=0.1 | Reward -17.4 +/- 1.63. Slightly better than baseline -- larger batches keep smoothing out the gradient noise. |
| B | lr=0.0001, gamma=0.99, batch=128, eps_start=1.0, eps_end=0.05, eps_decay=0.1 | Reward -16.4 +/- 1.63. Best of the batch_size sweep. Larger batches consistently helped throughout this axis -- more stable gradient estimates per update seem to matter more than the extra compute per step. |
| C | eps_start=1.0, eps_end=0.20, eps_decay=0.05 | Reward -17.2 +/- 1.17. Keeping a relatively high floor on epsilon (20%) means the agent never really commits to exploiting what it's learned -- one-fifth of actions stay random for the rest of training after only 5% of the run. Still learns something, but that much permanent randomness caps how consistent it gets. |
| C | eps_start=1.0, eps_end=0.10, eps_decay=0.05 | Reward -16.2 +/- 1.60. Best result of the sweep. Same fast 5%-of-training decay as the row above, but a lower floor afterward -- looks like the sweet spot between exploring early and actually exploiting later. |
| C | eps_start=1.0, eps_end=0.05, eps_decay=0.05 | Reward -18.0 +/- 1.67. Baseline eps_end but decaying five times faster than baseline. Locking into a near-greedy policy this early (only 10k of 200k steps) seems to hurt -- the agent hasn't explored enough of the state space yet before it stops exploring. |
| C | eps_start=1.0, eps_end=0.05, eps_decay=0.10 | Reward -17.8 +/- 2.14. This is the shared baseline config -- included here as our internal reference point for the rest of the sweep. |
| C | eps_start=1.0, eps_end=0.05, eps_decay=0.20 | Reward -16.2 +/- 2.32. Tied for the best result in the sweep. Doubling the decay window versus baseline gives twice as long exploring before settling down -- more exploration time helped here, not hurt. |
| C | eps_start=1.0, eps_end=0.05, eps_decay=0.30 | Reward -17.2 +/- 2.23. Stretching decay even further doesn't keep improving on the row above -- reward drops back toward baseline. Looks like there's a point past which more exploration time stops paying off. |
| C | eps_start=1.0, eps_end=0.01, eps_decay=0.10 | Reward -17.0 +/- 1.27. Near-zero floor instead of baseline's 0.05, same decay speed. Small improvement and noticeably lower variance -- a near-fully-greedy final policy is more consistent episode to episode. |
| C | eps_start=0.8, eps_end=0.05, eps_decay=0.10 | Reward -17.4 +/- 1.02. Starting exploration at 80% instead of fully random didn't hurt -- close to baseline. Makes sense: even at 80% random, the agent still gets a huge amount of random experience early on. |
| C | eps_start=1.0, eps_end=0.30, eps_decay=0.10 | Reward -17.0 +/- 1.41. Big permanent floor (30% random forever), yet did about as well as the near-zero floor above. Reading these two together, the exact final epsilon value seems to matter less than we expected. |
| C | eps_start=1.0, eps_end=0.05, eps_decay=0.50 | Reward -18.6 +/- 1.02. Worst result of the sweep. Spending half the entire training budget still in heavy exploration leaves too little time actually exploiting what's been learned. |

Overall pattern for `learning_rate`: anything from 0.01 down to 0.0003
failed to learn at all (flat -21.0, no variance). Real learning only
showed up once lr dropped to the 5e-5 to 1e-4 range, peaking around
5e-05. Going lower still (1e-05) started losing ground again -- too
slow to converge in the timestep budget we gave it. Classic too-high
(unstable) vs too-low (too slow) tradeoff.

Overall pattern for `gamma`: shorter effective horizons (0.90-0.95)
outperformed both the baseline and values closer to 1 (0.98-0.999) --
gamma=0.95 gave the single best result of the whole 30-experiment
sweep. Pong's reward arrives quickly relative to the actions that
cause it, so discounting more aggressively doesn't lose much useful
signal while apparently making the value estimates easier to learn.

Overall pattern for `batch_size`: reward increased almost monotonically
from batch=8 up through batch=128 -- bigger batches gave less noisy
gradient estimates and better results across the board, with no sign
of diminishing returns yet at the largest size we tested.

Overall pattern for exploration params: decay speed mattered more than
where epsilon actually ended up. The 10-20%-of-training decay window
(rows 2 and 5) gave the best results, while decaying too fast (5%) or
too slow (50%) both hurt -- there's a real middle ground. The final
epsilon floor itself barely moved the result (0.01 vs 0.30 landed close
together), which surprised us going in -- we expected the floor to
matter as much as the decay speed, but it didn't.

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

Across all three axes, `gamma` had the single biggest impact on final
performance -- gamma=0.95 produced the best result of any of the 30
experiments (-15.4), beating the best `learning_rate` result (-17.2)
and the best `epsilon` configs (-16.2). But `learning_rate` had the
most dramatic effect in a different sense: it was the only axis where
results were essentially binary. Either the agent learned nothing at
all (flat -21.0, six of the ten runs) or it learned something real
(-17 to -18 range) -- there was no middle ground. `gamma` and
`epsilon`, by contrast, never produced a complete failure; every
config on those two axes learned at least something, and the
differences between configs were about degree, not kind.

The exploration-exploitation tradeoff showed up most clearly in the
epsilon sweep: decay speed (how much of training passed before
settling into the final epsilon) mattered more than where epsilon
actually ended up. Too fast a decay (5% of training) locked the agent
into greedy behavior before it had explored enough of the state space;
too slow (50%) left too little time to actually exploit what it had
learned. The final epsilon floor itself barely moved the needle, which
surprised us going in.

Biggest surprise overall: shorter discount horizons (gamma=0.90-0.95)
consistently beat both the baseline (0.99) and values closer to 1
(0.98-0.999). We expected valuing future rewards more (higher gamma)
to help a game like Pong, but Pong's reward signal arrives quickly
relative to the actions that cause it, so discounting more aggressively
didn't lose much useful signal while apparently making the value
function easier to learn -- and gave us the best single result of the
whole sweep.

## Demo Video

We ran `play.py` locally with our final trained model (`dqn_model.zip`,
gamma=0.95) and screen-recorded it -- 5 episodes, orange paddle is our
agent, green is Pong's built-in opponent.

<video src="demo_video.mp4" controls width="600"></video>

(If the video doesn't render inline, it's also available directly at
[`demo_video.mp4`](demo_video.mp4) in the repo root.)

(If the video doesn't play inline, it's also downloadable directly at
[`demo_video.mp4`](demo_video.mp4) in the repo root.)

## Contributions

- Member A: ran the learning_rate sweep (10 runs, IDs memberA_lr_01 to memberA_lr_10), wrote the corresponding results table rows and pattern summary. Also ran the MlpPolicy vs CnnPolicy comparison and wrote that section.
- Member C: ran the exploration params (epsilon) sweep (10 runs, IDs memberC_eps_01 to memberC_eps_10), wrote the corresponding results table rows and pattern summary, pushed the executed notebook to notebooks/executed/.
- Member B: gamma + batch_size sweep (10 runs, IDs memberB_gamma_01 to memberB_gamma_05 and memberB_batch_01 to memberB_batch_05). Member A reran this sweep to completion after the original attempt didn't finish, and wrote the corresponding results table rows and pattern summary.
