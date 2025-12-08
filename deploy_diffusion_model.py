#!/usr/bin/env python3
"""
Deploy a trained diffusion policy in the TruckEnv (state-only, no cameras).

Usage example:
  python3 deploy_diffusion_model.py \
      --checkpoint checkpoints/diffusion_best.pt \
      --stats_path data_no_image/.norm_stats.json \
      --max_steps 2000
"""
import argparse
import json
import math
import os
from typing import Dict, Any

import numpy as np
import torch

from diffusion_model import ActionConditionedDiffusion
from env import TruckEnv

WORKSPACE_BOUNDS = np.array([-69, 157, -30, 200], dtype=np.float32)


def _normalize_pose(arr: np.ndarray) -> np.ndarray:
    """Normalize (x, y, yaw) using workspace bounds and yaw scaling."""
    arr = arr.copy()
    arr[:2] = 2 * (arr[:2] - WORKSPACE_BOUNDS[[0, 2]]) / (WORKSPACE_BOUNDS[[1, 3]] - WORKSPACE_BOUNDS[[0, 2]]) - 1
    arr[2] = arr[2] / 180.0
    return arr


def build_condition(obs: Any, stats: Dict[str, Any], map_location: float = 0.0) -> torch.Tensor:
    """
    Convert env observation to diffusion conditioning vector.
    Obs (use_cameras=False): (pos, vel, accel, trailer_angle, reverse, goal)
    Returns tensor shape (1, 9): pos_rel(3) + vel(2) + accel(2) + trailer(1) + reverse(1)
    """
    pos_list, vel_list, accel_list, trailer_angle, reverse_flag, goal_list = obs

    # Align coordinates if multiple map slices were used during training
    shift = (11000.0 * map_location) / 100.0
    pos_arr = np.array(pos_list, dtype=np.float32)
    goal_arr = np.array(goal_list, dtype=np.float32)
    pos_arr[0] -= shift
    goal_arr[0] -= shift

    # Select pose fields (x, y, yaw)
    pos_sel = pos_arr[[0, 1, 4]]
    goal_sel = goal_arr[[0, 1, 4]]
    pos_norm = _normalize_pose(pos_sel)
    goal_norm = _normalize_pose(goal_sel)
    pos_rel = torch.from_numpy(goal_norm - pos_norm)

    vel = np.array(vel_list, dtype=np.float32)[[0, 1]]
    accel = np.array(accel_list, dtype=np.float32)[[0, 1]]
    vel_t = (torch.from_numpy(vel) - torch.tensor(stats["vel_mean"], dtype=torch.float32)) / torch.tensor(
        stats["vel_std"], dtype=torch.float32
    )
    accel_t = (torch.from_numpy(accel) - torch.tensor(stats["accel_mean"], dtype=torch.float32)) / torch.tensor(
        stats["accel_std"], dtype=torch.float32
    )

    trailer_t = torch.tensor([float(trailer_angle) / 180.0], dtype=torch.float32)
    rev_t = torch.tensor([1.0 if reverse_flag else 0.0], dtype=torch.float32)

    cond = torch.cat([pos_rel, vel_t, accel_t, trailer_t, rev_t], dim=0).unsqueeze(0)
    return cond


def clamp_action(action: torch.Tensor) -> np.ndarray:
    """Clamp action to env ranges and convert to numpy."""
    # action format: [throttle/brake (signed), steer, reverse_flag]
    a = action.clone()
    a[0] = torch.clamp(a[0], -1.0, 1.0)  # throttle/brake
    a[1] = torch.clamp(a[1], -2.0, 2.0)  # steer (env allows [-2,2])
    a[2] = 1.0 if a[2] > 0 else 0.0      # reverse toggle flag
    return a.cpu().numpy()


def load_stats(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Stats file not found: {path}")
    with open(path, "r") as fh:
        return json.load(fh)


def main():
    parser = argparse.ArgumentParser(description="Deploy a diffusion policy in TruckEnv.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to diffusion checkpoint (.pt)")
    parser.add_argument("--stats_path", type=str, default="data_no_image/.norm_stats.json", help="Normalization stats JSON")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (auto if not set)")
    parser.add_argument("--max_steps", type=int, default=2000, help="Max environment steps before reset")
    parser.add_argument("--phase", type=int, default=2, help="Env phase (penalties/reward shaping)")
    parser.add_argument("--map_location", type=float, default=0.0, help="Map slice index used during training")
    args = parser.parse_args()

    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    stats = load_stats(args.stats_path)

    # Instantiate and load model
    model = ActionConditionedDiffusion().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()

    env = TruckEnv(max_steps=args.max_steps, phase=args.phase, map_location=args.map_location, use_cameras=False)

    step = 0
    obs = env.reset()
    try:
        with torch.no_grad():
            while True:
                cond = build_condition(obs, stats, map_location=args.map_location).to(device)
                sampled = model.sample(cond, num_samples=1)[0]
                action = clamp_action(sampled)

                obs, _, terminated, truncated = env.step(action)
                step += 1

                if terminated or truncated or step >= args.max_steps:
                    step = 0
                    obs = env.reset()
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down.")
    finally:
        env.destroy()


if __name__ == "__main__":
    main()
