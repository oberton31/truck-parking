#!/usr/bin/env python3
import argparse
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gym_env import TruckParkingGymEnv

from time import time
def make_env(max_episode_steps, use_cameras):
    """Create a non-vectorized env for deterministic evaluation."""
    env = TruckParkingGymEnv(
        max_episode_steps=max_episode_steps,
        phase=1,               # phase doesn't matter for evaluation unless you want curriculum
        decision_period=4,
        stack_size=5,
        use_cameras=use_cameras,
        npv_max=20,
        map_location=0,
    )
    return Monitor(env)


def evaluate(model_path, episodes=50, max_steps=1000):
    env = make_env(max_steps, use_cameras=False)

    # Load model
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")

    rewards = []
    lengths = []
    successes = 0

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        success = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            #time.sleep(0.2)  # To better visualize in CARLA

            total_reward += reward
            steps += 1

            done = terminated or truncated
            success = info.get("goal_reached") or info.get("is_success")

        rewards.append(total_reward)
        lengths.append(steps)
        if success:
            successes += 1

        print(f"Episode {ep+1}/{episodes} | Reward={total_reward:.3f} | Steps={steps} | Success={success}")

    env.close()

    print("\n===== Evaluation Summary =====")
    print(f"Episodes:          {episodes}")
    print(f"Success rate:      {successes/episodes:.3f}")
    print(f"Avg reward:        {np.mean(rewards):.3f}")
    print(f"Std reward:        {np.std(rewards):.3f}")
    print(f"Avg episode length:{np.mean(lengths):.1f}")
    print("==============================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO truck parking policy.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained PPO model .zip file.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to run.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode.")
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        episodes=args.episodes,
        max_steps=args.max_steps,
    )
