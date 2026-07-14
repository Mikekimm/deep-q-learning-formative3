deep-q-learning-formative3

Our group's DQN Atari assignment -- training a DQN agent (Stable
Baselines3 + Gymnasium) to play Pong, running 30 hyperparameter
experiments (10 each), and evaluating the best model with play.py.

Who's doing what

Each of us owns one hyperparameter axis and holds everything else at
the shared baseline. That way our 30 results are actually comparable --
no confounded variables -- and the "noted behavior" write-up is an
explanation, not a guess.

MemberAxisNotebookAlearning_rate (10 values)notebooks/experiments_memberA_lr.ipynbBgamma + batch_size (5 + 5)notebooks/experiments_memberB_gamma_batch.ipynbCexploration params (exploration_initial_eps, exploration_final_eps, exploration_fraction, 10 combos)notebooks/experiments_memberC_epsilon.ipynb

Also on us: a one-off MlpPolicy vs CnnPolicy comparison the assignment
requires (notebooks/policy_comparison_mlp_vs_cnn.ipynb) -- not part of
anyone's 10, whoever picks it up just runs it once.

What we locked in

We agreed on these up front so our 30 runs stay comparable -- don't
change them mid-sweep without checking with the group first (actual
values live in shared_train.py):


Game: ALE/Pong-v5
Timesteps per run: 200,000, same for every run
Seed: 42
Baseline hyperparameters: BASELINE_CONFIG in shared_train.py


If your Colab is too slow for 200k steps x 10 runs, flag it to the
group rather than quietly changing the number -- it has to stay
identical across everyone's runs or the comparison breaks.

Running your experiments (Colab)


Open your notebook fresh from GitHub each session (not a cached tab)
-- File → Open notebook → GitHub.
Enable a GPU runtime: Runtime → Change runtime type → T4 GPU.
Massively faster than CPU, worth doing before you start.
Run the install cell, then the repo cell -- it clones fresh or resets
to match GitHub exactly, so you're always on the latest code.
Run the smoke test cell by itself first -- confirms your session's
actually working before committing hours to the real sweep. Don't
Run All the whole notebook.
Run your real experiment loop. Each run automatically saves a model
to results/models/<run_name>.zip and appends a row to
results/experiments_log.csv.
After each run, jot down what we saw -- reward trend, stability,
divergence -- we need this for the report's "noted behavior" column.
Optional: the live TensorBoard cell shows reward curves updating in
real time while a run trains, if you'd rather watch than wait.


Syncing results back

We're all appending to the same results/experiments_log.csv, so pull
before you push or we'll clobber each other's rows:

git add results/experiments_log.csv
git commit -m "Add <name> <axis> results"
git pull --no-rebase origin main
git push origin main

Once all 30 runs (+ the MLP/CNN comparison) are in


Check results/experiments_log.csv has everyone's rows.
Pick the config with the best mean_reward.
Retrain it as our official submission artifact:


   python train.py --learning_rate <val> --gamma <val> --batch_size <val> \
       --exploration_initial_eps <val> --exploration_final_eps <val> \
       --exploration_fraction <val>

This produces dqn_model.zip in the repo root.
4. Watch it play (run locally, not in Colab -- Colab's headless and can't
open a real display):

   python play.py --model dqn_model.zip --episodes 5


Screen-record step 4 -- the submission requires a video of
play.py actually running with the agent playing. Save it and either
embed it below or link it (e.g. upload to the repo if small enough,
or link a Drive/YouTube unlisted upload).


We don't need a separate report doc for this one -- the table, the
discussion, and the video all just go straight into this README.

Hyperparameter Results

A's learning_rate sweep and C's exploration sweep are done, filled in
below. B's rows are still TODO until their sweep finishes.

MemberHyperparameter SetNoted BehaviorAlr=0.01, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.05, eps_decay=0.1Reward pinned at -21.0 the whole run, zero variance across eval episodes -- learning rate too high, updates never stabilized.Alr=0.005, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.05, eps_decay=0.1Same as above -- flat -21.0, no learning. Still too high.Alr=0.001, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.05, eps_decay=0.1Still flat -21.0. Even an order of magnitude down from the top value, learning rate is still too high for this to converge in 200k steps.Alr=0.0007, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.05, eps_decay=0.1Flat -21.0 again -- no sign of the pattern breaking yet.Alr=0.0005, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.05, eps_decay=0.1Same flat -21.0 result.Alr=0.0003, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.05, eps_decay=0.1Last of the flat -21.0 runs -- everything above ~1e-4 failed to learn at all in our budget.Alr=0.0001, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.05, eps_decay=0.1First real break from -21: reward -17.8 +/- 2.14. Non-zero variance means the agent's actually behaving differently across episodes now, not stuck in one degenerate policy. This is our baseline lr.Alr=7e-05, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.05, eps_decay=0.1Similar to the run above: -18.0 +/- 1.79. Confirms the 5e-5 to 1e-4 range is roughly where this converges within 200k steps.Alr=5e-05, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.05, eps_decay=0.1Best result of the sweep: -17.2 +/- 3.25. Highest reward and highest variance -- looks like the most active learning of any run.Alr=1e-05, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.05, eps_decay=0.1Reward dropped back to -20.6 +/- 0.49. Learning rate this low is probably too slow to make much progress in only 200k steps.B(TODO)(TODO)Ceps_start=1.0, eps_end=0.20, eps_decay=0.05Reward -17.2 +/- 1.17. Keeping a relatively high floor on epsilon (20%) means the agent never really commits to exploiting what it's learned -- one-fifth of actions stay random for the rest of training after only 5% of the run. Still learns something, but that much permanent randomness caps how consistent it gets.Ceps_start=1.0, eps_end=0.10, eps_decay=0.05Reward -16.2 +/- 1.60. Best result of the sweep. Same fast 5%-of-training decay as the row above, but a lower floor afterward -- looks like the sweet spot between exploring early and actually exploiting later.Ceps_start=1.0, eps_end=0.05, eps_decay=0.05Reward -18.0 +/- 1.67. Baseline eps_end but decaying five times faster than baseline. Locking into a near-greedy policy this early (only 10k of 200k steps) seems to hurt -- the agent hasn't explored enough of the state space yet before it stops exploring.Ceps_start=1.0, eps_end=0.05, eps_decay=0.10Reward -17.8 +/- 2.14. This is the shared baseline config -- included here as our internal reference point for the rest of the sweep.Ceps_start=1.0, eps_end=0.05, eps_decay=0.20Reward -16.2 +/- 2.32. Tied for the best result in the sweep. Doubling the decay window versus baseline gives twice as long exploring before settling down -- more exploration time helped here, not hurt.Ceps_start=1.0, eps_end=0.05, eps_decay=0.30Reward -17.2 +/- 2.23. Stretching decay even further doesn't keep improving on the row above -- reward drops back toward baseline. Looks like there's a point past which more exploration time stops paying off.Ceps_start=1.0, eps_end=0.01, eps_decay=0.10Reward -17.0 +/- 1.27. Near-zero floor instead of baseline's 0.05, same decay speed. Small improvement and noticeably lower variance -- a near-fully-greedy final policy is more consistent episode to episode.Ceps_start=0.8, eps_end=0.05, eps_decay=0.10Reward -17.4 +/- 1.02. Starting exploration at 80% instead of fully random didn't hurt -- close to baseline. Makes sense: even at 80% random, the agent still gets a huge amount of random experience early on.Ceps_start=1.0, eps_end=0.30, eps_decay=0.10Reward -17.0 +/- 1.41. Big permanent floor (30% random forever), yet did about as well as the near-zero floor above. Reading these two together, the exact final epsilon value seems to matter less than we expected.Ceps_start=1.0, eps_end=0.05, eps_decay=0.50Reward -18.6 +/- 1.02. Worst result of the sweep. Spending half the entire training budget still in heavy exploration leaves too little time actually exploiting what's been learned.

Overall pattern for learning_rate: anything from 0.01 down to 0.0003
failed to learn at all (flat -21.0, no variance). Real learning only
showed up once lr dropped to the 5e-5 to 1e-4 range, peaking around
5e-05. Going lower still (1e-05) started losing ground again -- too
slow to converge in the timestep budget we gave it. Classic too-high
(unstable) vs too-low (too slow) tradeoff.

Overall pattern for exploration params: decay speed mattered more than
where epsilon actually ended up. The 10-20%-of-training decay window
(rows 2 and 5) gave the best results, while decaying too fast (5%) or
too slow (50%) both hurt -- there's a real middle ground. The final
epsilon floor itself barely moved the result (0.01 vs 0.30 landed close
together), which surprised us going in -- we expected the floor to
matter as much as the decay speed, but it didn't.

MlpPolicy vs CnnPolicy comparison

Same baseline hyperparameters (lr=0.0001, gamma=0.99, batch=32),
200,000 steps, only the policy/observation type differs:

PolicyObservationmean_rewardCnnPolicypixel frames (84x84, 4-stacked)-17.40 +/- 2.06MlpPolicyRAM vector (128,)-21.00 +/- 0.00

CnnPolicy clearly wins here. MlpPolicy never broke away from -21 at
all (zero variance, same as our worst learning-rate runs) despite
running the exact config that worked well for CnnPolicy. Makes sense
given what the two are working with -- pixel frames have spatial
structure (ball position, paddle position, motion) that convolutions
are built to pick up on, while raw RAM bytes are just unstructured
memory values with no spatial relationship between neighboring
numbers. A plain MLP has a much harder time pulling anything useful
out of that in the same training budget.

Discussion of Hyperparameter Tuning Results

(TODO once the table above is filled in -- write up the patterns we
saw and why, not just what happened. Which axis moved the needle most?
Any surprises?)

Demo Video

(TODO -- link or embed the play.py recording here once the final
model is trained.)

Contributions

(TODO -- who ran which experiment IDs on which axis, who wrote which
part of this README.)