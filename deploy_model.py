#!/usr/bin/env python3
"""
deploy_model.py

Run a trained policy inside the TruckEnv (replaces the human controller in
`data_collection.py`). The script will:
- create the env
- load a checkpoint saved by `train.py`
- run a control loop where at each step it reads the env observation, builds
  model inputs, runs the model, and applies the predicted action to the env.

Behavior notes:
- The model predicts 5 dims: throttle, brake, steer, reverse_prob, handbrake_prob.
- To avoid toggling reverse every step, the script converts the model's
  reverse probability to a one-shot toggle pulse only when the desired
  reverse state differs from the env's current reverse flag.

Usage:
  python deploy_model.py --checkpoint checkpoints/best.pt --max_steps 2000 --save_dir collected_by_model

"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from model import TruckNet
from env import TruckEnv


# Workspace bounds - same as dataset preprocessing
WORKSPACE_BOUNDS = np.array([-69, 157, -30, 200])


def make_transform(img_size):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def frame_to_model_inputs(obs, img_size, stats_path=None, device='cpu'):
    """
    Convert env observation to model inputs.

    obs is (image_list, pos_list, vel_list, accel_list, trailer_angle, reverse, goal_list)
    Returns tensors: imgs(1,num_cams,3,H,W), pos_rel(1,3), vel(1,2), accel(1,2), trailer(1,1), rev(1,1)
    """
    images, pos_list, vel_list, accel_list, trailer_angle, reverse_flag, goal_list = obs

    transform = make_transform(img_size)
    imgs = []
    for i in range(len(images)):
        im = images[i]
        im_pil = T.functional.to_pil_image(im)
        imgs.append(transform(im_pil))
    imgs = torch.stack(imgs, dim=0).unsqueeze(0)  # (1, num_cams, 3, H, W)

    # select indices used in dataset: pos [0,1,4], vel [0,1], accel [0,1], goal [0,1,4]
    def sel_list(arr, indices):
        a = np.array(arr, dtype=np.float32).ravel()
        out = [float(a[i]) if i < a.size else 0.0 for i in indices]
        return np.array(out, dtype=np.float32)

    pos = sel_list(pos_list, [0, 1, 4])
    vel = sel_list(vel_list, [0, 1])
    accel = sel_list(accel_list, [0, 1])
    goal = sel_list(goal_list, [0, 1, 4])

    pos_t = torch.from_numpy(pos).unsqueeze(0)
    goal_t = torch.from_numpy(goal).unsqueeze(0)

    workspace_bounds = torch.tensor(WORKSPACE_BOUNDS, dtype=torch.float32)
    pos_t[:, :2] = 2 * (pos_t[:, :2] - workspace_bounds[[0, 2]]) / (workspace_bounds[[1, 3]] - workspace_bounds[[0, 2]]) - 1
    pos_t[:, 2] = pos_t[:, 2] / 180.0

    goal_t[:, :2] = 2 * (goal_t[:, :2] - workspace_bounds[[0, 2]]) / (workspace_bounds[[1, 3]] - workspace_bounds[[0, 2]]) - 1
    goal_t[:, 2] = goal_t[:, 2] / 180.0

    pos_rel = goal_t - pos_t

    # load normalization stats if provided; otherwise use zero/one fallback
    with open(stats_path, 'r') as fh:
        stats = json.load(fh)
    vel_mean = torch.tensor(stats.get('vel_mean', [0.0, 0.0]), dtype=torch.float32).unsqueeze(0)
    vel_std = torch.tensor(stats.get('vel_std', [1.0, 1.0]), dtype=torch.float32).unsqueeze(0)
    accel_mean = torch.tensor(stats.get('accel_mean', [0.0, 0.0]), dtype=torch.float32).unsqueeze(0)
    accel_std = torch.tensor(stats.get('accel_std', [1.0, 1.0]), dtype=torch.float32).unsqueeze(0)


    vel_t = (torch.from_numpy(vel).unsqueeze(0) - vel_mean) / vel_std
    accel_t = (torch.from_numpy(accel).unsqueeze(0) - accel_mean) / accel_std

    trailer_t = torch.tensor([[float(trailer_angle)]], dtype=torch.float32) / 180.0
    rev_t = torch.tensor([[1.0 if reverse_flag else 0.0]], dtype=torch.float32)

    return imgs, pos_rel, vel_t, accel_t, trailer_t, rev_t


class AsyncSaverSimple:
    """Simple synchronous saver used to write .npz chunks (no multiprocessing).
    Kept small to avoid extra dependencies; used only when --save_dir is provided.
    """
    def __init__(self, save_dir, chunk_size=15):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.buffer = []
        self.chunk_idx = 0

    def append(self, frame_dict):
        self.buffer.append(frame_dict)
        if len(self.buffer) >= self.chunk_size:
            self._flush()

    def _flush(self):
        if len(self.buffer) == 0:
            return
        fname = self.save_dir / f"chunk_{self.chunk_idx:04d}.npz"
        try:
            np.savez_compressed(str(fname), frames=self.buffer)
            print(f"Saved {fname}")
        except Exception as e:
            print(f"Failed to save {fname}: {e}")
        self.buffer = []
        self.chunk_idx += 1

    def close(self):
        self._flush()


def run_loop(args):
    device = torch.device(args.device if args.device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))

    env = TruckEnv(max_steps=args.max_steps, phase=args.phase)
    obs = env.reset()

    model = TruckNet(pretrained=args.pretrained)
    model.to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get('model_state', ckpt)
    model.load_state_dict(state)
    model.eval()

    saver = None
    if args.save_dir is not None:
        saver = AsyncSaverSimple(args.save_dir, chunk_size=args.save_chunk)

    step = 0
    try:
        while True:
            imgs, pos_rel, vel_t, accel_t, trailer_t, rev_t = frame_to_model_inputs(obs, args.img_size, args.stats, device=device)

            imgs = imgs.to(device)
            pos_rel = pos_rel.to(device)
            vel_t = vel_t.to(device)
            accel_t = accel_t.to(device)
            trailer_t = trailer_t.to(device)
            rev_t = rev_t.to(device)

            with torch.no_grad():
                mean, logvar, _ = model(imgs, pos_rel, vel_t, accel_t, trailer_t, rev_t)

            mean = mean.squeeze(0).cpu().numpy()  # (5,)
            # mean[0:2] in [0,1], mean[2] in [-2,2], mean[3:5] in [0,1]

            throttle = float(np.clip(mean[0], 0.0, 1.0))
            brake = float(np.clip(mean[1], 0.0, 1.0))
            steer = float(mean[2])  
            
            rev_toggle = bool(mean[3] >= 0.5)
            handbrake = bool(mean[4] >= 0.5)
            
            action = [throttle, brake, steer, rev_toggle, float(handbrake)]

            next_obs, reward, terminated, truncated = env.step(action)

            # Optionally save data collected by the model
            if saver is not None:
                saver.append(dict(
                    images=np.stack(obs[0], axis=0),
                    pos=np.array(obs[1], dtype=np.float32),
                    vel=np.array(obs[2], dtype=np.float32),
                    accel=np.array(obs[3], dtype=np.float32),
                    trailer_angle=np.array(obs[4], dtype=np.float32),
                    reverse=np.array(obs[5], dtype=np.float32),
                    goal=np.array(obs[6], dtype=np.float32),
                    actions=np.array(action, dtype=np.float32),
                    reward=np.array(reward, dtype=np.float32),
                    done=np.array(terminated or truncated, dtype=np.uint8),
                    timestamp=np.array(time.time(), dtype=np.float64),
                ))

            obs = next_obs
            step += 1

            if terminated or truncated:
                print("Episode finished, resetting env")
                obs = env.reset()

            if args.max_steps is not None and step >= args.max_steps:
                print("Reached max steps, exiting")
                break

    except KeyboardInterrupt:
        print("Keyboard interrupt, shutting down")
    finally:
        if saver is not None:
            saver.close()
        env.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--save_dir', type=str, default=None, help='Optional directory to save model-collected chunks (.npz)')
    parser.add_argument('--save_chunk', type=int, default=15)
    parser.add_argument('--stats', type=str, default=None, help='Optional path to .norm_stats.json used for vel/accel normalization')
    parser.add_argument('--phase', type=int, default=1, help='Env phase (collision penalties)')
    args = parser.parse_args()

    run_loop(args)


if __name__ == '__main__':
    main()
