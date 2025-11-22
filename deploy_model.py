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
from __future__ import print_function

import glob
import os
import sys
try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
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

# Optional OpenCV-based visualizer for actions (used when --gui is passed)
# We import cv2 lazily inside run_loop so missing dependency is handled gracefully.


# Workspace bounds - same as dataset preprocessing
WORKSPACE_BOUNDS = np.array([-69, 157, -30, 200])


def make_transform(img_size):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def frame_to_model_inputs(obs, img_size, stats_path=None, device='cpu', map_location=0):
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
    
    pos_list[0] -= map_location * 11000 / 100
    goal_list[0] -= map_location * 11000 / 100  # adjust x pos based on map location
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


def _update_visualizer_cv(vis, action):
    """Draw a simple HUD using OpenCV showing throttle, brake, steer angle and gear flags.

    action: [throttle, brake, steer, rev, handbrake]
    """
    cv2 = vis['cv2']
    win = vis['win_name']
    W, H = vis['W'], vis['H']

    throttle, brake, steer, rev_f, handbrake = action

    # background
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)

    # steering wheel: center-left
    cx, cy = 110, (H // 2) - 30
    radius = 60
    cv2.circle(img, (cx, cy), radius, (200, 200, 200), 6)

    # steering indicator line
    max_steer_deg = 45.0
    steer_norm = max(-1.0, min(1.0, steer / 2.0)) if abs(steer) > 0 else steer / 2.0
    angle_rad = -steer_norm * max_steer_deg * np.pi / 180.0
    lx = int(cx + (radius - 10) * np.cos(angle_rad))
    ly = int(cy + (radius - 10) * np.sin(angle_rad)) - 30
    cv2.line(img, (cx, cy), (lx, ly), (255, 100, 100), 6)

    # center marker
    cv2.circle(img, (cx, cy), 6, (255, 255, 255), -1)

    # throttle/brake bars on right
    bar_x = 240
    bar_w = 28
    bar_h_max = H - 60
    # throttle
    th_h = int(bar_h_max * float(throttle))
    cv2.rectangle(img, (bar_x, 30), (bar_x + bar_w, 30 + bar_h_max), (60, 60, 60), -1)
    cv2.rectangle(img, (bar_x, 30 + bar_h_max - th_h), (bar_x + bar_w, 30 + bar_h_max), (50, 200, 50), -1)
    cv2.putText(img, f"Throttle: {throttle:.2f}", (bar_x + bar_w -80, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    # brake
    br_x = bar_x + 120
    br_h = int(bar_h_max * float(brake))
    cv2.rectangle(img, (br_x, 30), (br_x + bar_w, 30 + bar_h_max), (60, 60, 60), -1)
    cv2.rectangle(img, (br_x, 30 + bar_h_max - br_h), (br_x + bar_w, 30 + bar_h_max), (200, 50, 50), -1)
    cv2.putText(img, f"Brake: {brake:.2f}", (br_x + bar_w -60, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    # Gear and handbrake text
    gear = 'R' if rev_f >= 0.5 else 'D'
    cv2.putText(img, f"Gear: {gear}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(img, f"Handbrake: {'ON' if handbrake >= 0.5 else 'OFF'}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # small steer angle readout
    cv2.putText(img, f"Steer: {steer:.2f}", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow(win, img)
    # process window events
    cv2.waitKey(1)
def run_loop(args):
    device = torch.device(args.device if args.device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    world = client.load_world('Town10HD')
    envs = [TruckEnv(max_steps=args.max_steps, phase=args.phase, map_location=0, world=world), TruckEnv(max_steps=args.max_steps, phase=args.phase, map_location=1, world=world), TruckEnv(max_steps=args.max_steps, phase=args.phase, map_location=2, world=world), TruckEnv(max_steps=args.max_steps, phase=args.phase, map_location=3, world=world)]
    #env = TruckEnv(max_steps=args.max_steps, phase=args.phase, map_location=2)
    obs_env = [None] * 4
    for i, env in enumerate(envs):
        obs_env[i] = env.reset()

    model = TruckNet(pretrained=args.pretrained)
    model.to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get('model_state', ckpt)
    model.load_state_dict(state)
    model.eval()

    # create visualizer if requested (OpenCV-based)
    vis = None
    if getattr(args, 'gui', False):
        try:
            import cv2
            # small window
            W, H = 420, 260
            win_name = 'Action Visualizer'
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, W, H)
            vis = dict(cv2=cv2, win_name=win_name, W=W, H=H)
        except Exception as e:
            print(f"OpenCV visualizer disabled (import/init error): {e}")

    step = 0
    try:
        while True:
            for i, env in enumerate(envs):
                obs = obs_env[i]
                imgs, pos_rel, vel_t, accel_t, trailer_t, rev_t = frame_to_model_inputs(obs, args.img_size, args.stats, device=device, map_location=i)

                imgs = imgs.to(device)
                pos_rel = pos_rel.to(device)
                vel_t = vel_t.to(device)
                accel_t = accel_t.to(device)
                trailer_t = trailer_t.to(device)
                rev_t = rev_t.to(device)

                with torch.no_grad():
                    mean, logvar, _ = model(imgs, pos_rel, vel_t, accel_t, trailer_t, rev_t)

                mean = mean.squeeze(0).cpu().numpy()  # (5,)

                if (step < 100): throttle = 1
                else: throttle = float(np.clip(mean[0], 0.0, 1.0))
                brake = float(np.clip(mean[1], 0.0, 1.0))
                if (brake < 0.1): brake = 0.0
                steer = float(mean[2])

                rev_toggle = bool(mean[3] >= 0.5)
                handbrake = bool(mean[4] >= 0.5)

                # if (rev_toggle):
                #     print("TOGGLED GEAR")
                action = [throttle, brake, steer, float(rev_toggle), float(handbrake)]

                if vis is not None:
                    try:
                        _update_visualizer_cv(vis, action)
                    except Exception as e:
                        print(f"Visualizer error: {e}")

                next_obs, reward, terminated, truncated = env.step(action)

                obs_env[i] = next_obs

                if terminated or truncated:
                    print("Episode finished, resetting env")
                    obs_env[i] = env.reset()

            step += 1
            print(f"Step {step} completed")
            if args.max_steps is not None and step >= args.max_steps:
                print("Reached max steps, exiting")
                break

    except KeyboardInterrupt:
        print("Keyboard interrupt, shutting down")
    finally:
        for env in envs:
            env.destroy()
        if vis is not None:
            try:
                if 'cv2' in vis and 'win_name' in vis:
                    vis['cv2'].destroyAllWindows()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--stats', type=str, default=None, help='Optional path to .norm_stats.json used for vel/accel normalization')
    parser.add_argument('--phase', type=int, default=1, help='Env phase (collision penalties)')
    parser.add_argument('--gui', action='store_true', help='Enable action visualizer GUI (OpenCV)')
    args = parser.parse_args()

    run_loop(args)


if __name__ == '__main__':
    main()
