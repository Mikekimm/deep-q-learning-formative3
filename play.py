"""
Loads the trained model and plays it in the Atari environment with
rendering. Actions are chosen greedily (deterministic=True is SB3's
equivalent of a GreedyQPolicy -- it always picks the highest-Q action
instead of the epsilon-greedy exploration used during training).
"""

import argparse

from stable_baselines3 import DQN

from shared_train import make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dqn_model.zip")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    model = DQN.load(args.model)
    env = make_env(render_mode="human")

    for episode in range(1, args.episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward[0]
            env.render()
        print(f"Episode {episode}: reward={total_reward}")

    env.close()


if __name__ == "__main__":
    main()
