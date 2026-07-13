"""
Shared training utilities for the DQN Atari group assignment.

All three members import from this file so every experiment uses an
identical environment setup, model construction, and logging pipeline.
Only the hyperparameter(s) each member is testing should differ between
runs -- everything else comes from BASELINE_CONFIG.
"""

import csv
import gc
import os
from datetime import datetime

import ale_py
import gymnasium
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack

# ale_py doesn't always self-register the ALE/* environments via entry
# points on Colab's pre-installed gymnasium -- register explicitly so
# gym.make("ALE/Pong-v5") reliably works.
gymnasium.register_envs(ale_py)

# ---------------------------------------------------------------------------
# LOCKED TEAM DECISIONS -- agree on these once as a group, then do not touch
# them per-experiment. Only the hyperparameters under test should vary.
# ---------------------------------------------------------------------------
GAME_ID = "ALE/Pong-v5"
TOTAL_TIMESTEPS = 200_000  # same budget for every one of the 30 runs
SEED = 42
N_EVAL_EPISODES = 5

# DQN's default buffer_size (1,000,000) needs ~56GB for stacked Atari
# frames -- far more than Colab provides. Capped so every run actually
# fits in memory instead of OOM-crashing partway through training.
BUFFER_SIZE = 100_000

BASELINE_CONFIG = {
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "batch_size": 32,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "exploration_fraction": 0.1,
}

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(_HERE, "results", "experiments_log.csv")
MODELS_DIR = os.path.join(_HERE, "results", "models")
TB_LOG_DIR = os.path.join(_HERE, "results", "tb_logs")


def make_env(render_mode=None):
    """Single source of truth for env construction -- used by both
    train_one_run() and play.py, so training and playback preprocessing
    can never drift apart."""
    env_kwargs = {"render_mode": render_mode} if render_mode else None
    env = make_atari_env(GAME_ID, n_envs=1, seed=SEED, env_kwargs=env_kwargs)
    env = VecFrameStack(env, n_stack=4)
    return env


def train_one_run(overrides: dict, run_name: str, member: str, notes: str = ""):
    """Train one DQN run. `overrides` should contain ONLY the
    hyperparameter(s) this member is testing."""
    config = {**BASELINE_CONFIG, **overrides}

    env = make_env()
    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        batch_size=config["batch_size"],
        exploration_initial_eps=config["exploration_initial_eps"],
        exploration_final_eps=config["exploration_final_eps"],
        exploration_fraction=config["exploration_fraction"],
        buffer_size=BUFFER_SIZE,
        seed=SEED,
        verbose=1,
        tensorboard_log=TB_LOG_DIR,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=run_name)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{run_name}.zip")
    model.save(model_path)

    eval_env = make_env()
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=N_EVAL_EPISODES, deterministic=True
    )

    row = {
        "run_name": run_name,
        "member": member,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **config,
        "total_timesteps": TOTAL_TIMESTEPS,
        "mean_reward": round(mean_reward, 3),
        "std_reward": round(std_reward, 3),
        "model_path": model_path,
        "notes": notes,
    }
    _append_result(row)
    print(f"[{run_name}] mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    # Free the replay buffer (~5.6GB per run) and env before returning --
    # without this, repeated calls in a sweep loop accumulate memory across
    # runs instead of releasing it, eventually causing severe swap-induced
    # slowdowns or an OOM crash partway through a 10-run sweep.
    env.close()
    eval_env.close()
    del model, env, eval_env
    gc.collect()

    return row


def train_mlp_baseline(notes: str = ""):
    """One-off MlpPolicy comparison run required by the assignment. Uses
    the RAM observation variant of the same game (a flat 128-byte vector,
    unlike the pixel frames CnnPolicy needs) with the same baseline
    hyperparameters and timestep budget as the CnnPolicy runs, so the
    comparison is fair. Not part of the 30-run hyperparameter sweep, so
    it isn't logged to experiments_log.csv -- just returns the result.

    Newer ale-py versions don't register a separate "-ram-" environment
    id -- RAM observations come from the same id via obs_type="ram"."""
    env = make_vec_env(GAME_ID, n_envs=1, seed=SEED, env_kwargs={"obs_type": "ram"})
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=BASELINE_CONFIG["learning_rate"],
        gamma=BASELINE_CONFIG["gamma"],
        batch_size=BASELINE_CONFIG["batch_size"],
        exploration_initial_eps=BASELINE_CONFIG["exploration_initial_eps"],
        exploration_final_eps=BASELINE_CONFIG["exploration_final_eps"],
        exploration_fraction=BASELINE_CONFIG["exploration_fraction"],
        buffer_size=BUFFER_SIZE,
        seed=SEED,
        verbose=1,
        tensorboard_log=TB_LOG_DIR,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="mlp_baseline")

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "mlp_baseline.zip")
    model.save(model_path)

    eval_env = make_vec_env(GAME_ID, n_envs=1, seed=SEED, env_kwargs={"obs_type": "ram"})
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=N_EVAL_EPISODES, deterministic=True
    )
    result = {
        "policy": "MlpPolicy",
        "game_id": f"{GAME_ID} (obs_type=ram)",
        "total_timesteps": TOTAL_TIMESTEPS,
        "mean_reward": round(mean_reward, 3),
        "std_reward": round(std_reward, 3),
        "model_path": model_path,
        "notes": notes,
    }
    print(f"[mlp_baseline] mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()
    eval_env.close()
    del model, env, eval_env
    gc.collect()

    return result


def _append_result(row: dict):
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    file_exists = os.path.isfile(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
