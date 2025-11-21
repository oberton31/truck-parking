#!/usr/bin/env python3
"""
train.py

Behavior cloning trainer for truck-parking dataset saved as .npz chunks.

This script builds a ResNet-based policy that consumes 4 RGB cameras (separately
processed by a shared ResNet backbone), concatenates their features, and
predicts a 5-dim action (throttle, brake, steer, reverse_toggle, handbrake).

Usage: python train.py --data_dir data --epochs 10 --batch_size 8
"""
import argparse
import glob
import os
from pathlib import Path
import random
import time
import math
import multiprocessing as mp

import numpy as np
from PIL import Image
import json
import tempfile
from model import TruckNet
import cv2 as cv

WORKSPACE_BOUNDS = np.array([-69, 157, -30, 200]) # xmin, xmax, ymin, ymax


def _count_frames_worker(path):
    """Module-level worker function for counting frames in a .npz file.

    Placing this at module level ensures it can be pickled and used by
    multiprocessing.Pool workers.
    """
    try:
        import numpy as _np
        with _np.load(path, allow_pickle=True) as d:
            if 'frames' in d:
                return (path, int(len(d['frames'])), None)
            else:
                return (path, 0, None)
    except Exception as e:
        return (path, None, str(e))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from tqdm import tqdm
import importlib
SummaryWriter = None
_TB_BACKEND = None
try:
    tb_mod = importlib.import_module('torch.utils.tensorboard')
    SummaryWriter = getattr(tb_mod, 'SummaryWriter')
    _TB_BACKEND = 'torch'
except Exception:
    try:
        tbx = importlib.import_module('tensorboardX')
        SummaryWriter = getattr(tbx, 'SummaryWriter')
        _TB_BACKEND = 'tensorboardX'
    except Exception:
        SummaryWriter = None
        _TB_BACKEND = None

# cuDNN tuning: allow benchmark (may pick fastest algorithm) and allow TF32 where available.
try:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    cudnn.deterministic = False
    # allow TF32 for faster matmuls on Ampere+ GPUs (optional)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

# Prefer 'spawn' start method for DataLoader workers to avoid some forking-related
# crashes (especially when native libs are used). If it's already set, ignore.
try:
    mp.set_start_method('spawn', force=False)
except RuntimeError:
    pass


def _worker_init_fn(worker_id):
    # Limit OpenMP/MKL threads in each worker to avoid oversubscription and
    # potential crashes due to too many threads.
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    # seed numpy/random for reproducibility per worker
    seed = (int.from_bytes(os.urandom(4), 'little') + worker_id) & 0xffffffff
    random.seed(seed)
    np.random.seed(seed)


class NpzFramesDataset(Dataset):
    """Dataset that indexes individual frames inside `.npz` chunk files saved by data_collection.py.

    Each chunk contains a single entry key 'frames' which is a Python list of dictionaries.
    Each frame-dict contains at least the keys 'images' and 'actions'.
    - images: numpy array shape (4, H, W, 3), dtype uint8
        - actions: numpy array shape (5,), dtype float32
        - goal: optional numeric vector saved per-frame (e.g. goal coordinates / yaw). If present,
            the dataset will detect its dimensionality and return it as a float32 tensor.
    """

    def __init__(self, data_dir, transform=None, max_files=None):
        self.data_dir = Path(data_dir)
        files = sorted(self.data_dir.rglob('*.npz'))
        if max_files:
            files = files[:max_files]
        self.files = [str(p) for p in files]

        # Build an index mapping each global idx -> (file_path, local_idx)
        self.index = []

        # Use a small on-disk cache to avoid re-reading every npz on repeated runs.
        cache_path = self.data_dir / '.npz_index.json'

        def _file_meta(p):
            try:
                st = p.stat()
                return (st.st_mtime, st.st_size)
            except Exception:
                return (None, None)

        # We use the module-level worker `_count_frames_worker` for multiprocessing.

        # load existing cache if present
        cache = {}
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as fh:
                    cache = json.load(fh)
            except Exception:
                cache = {}

        # Determine which files need to be scanned (new or changed)
        to_scan = []
        results = {}
        for f in self.files:
            p = Path(f)
            mtime, size = _file_meta(p)
            key = f
            if key in cache:
                entry = cache[key]
                if entry.get('mtime') == mtime and entry.get('size') == size and 'n' in entry:
                    results[key] = entry.get('n')
                    continue
            to_scan.append(f)

        if len(to_scan) > 0:
            workers = max(1, min((os.cpu_count() or 1) - 1, len(to_scan)))
            # parallel scan with a pool
            from multiprocessing import Pool
            errors = []
            with Pool(processes=workers) as pool:
                for path, n, err in tqdm(pool.imap_unordered(_count_frames_worker, to_scan), total=len(to_scan), desc='Scanning .npz files'):
                    if err is not None:
                        print(f"Warning: failed to read {path} during indexing: {err}")
                        results[path] = 0
                        errors.append((path, err))
                    else:
                        results[path] = n

            # update cache entries for scanned files
            for f in to_scan:
                p = Path(f)
                mtime, size = _file_meta(p)
                cache[f] = {'mtime': mtime, 'size': size, 'n': results.get(f, 0)}

            # write cache atomically
            try:
                fd, tmp = tempfile.mkstemp(dir=str(self.data_dir))
                with os.fdopen(fd, 'w') as fh:
                    json.dump(cache, fh)
                os.replace(tmp, str(cache_path))
            except Exception:
                try:
                    with open(cache_path, 'w') as fh:
                        json.dump(cache, fh)
                except Exception:
                    pass

        # Build index from results/cache (show progress)
        total_frames = 0
        for f in tqdm(self.files, desc='Building in-memory index'):
            n = results.get(f)
            if n is None and f in cache:
                n = cache[f].get('n', 0)
            n = int(n or 0)
            for i in range(n):
                self.index.append((f, i))
            total_frames += n

        # Verbose summary
        try:
            scanned_files = len(results)
            total_files = len(self.files)
            corrupted = len(errors) if 'errors' in locals() else 0
            print(f"Indexing complete: files={total_files}, scanned={scanned_files}, total_frames={total_frames}, corrupted_files={corrupted}")
        except Exception:
            pass

        self.transform = transform
        # normalization stats file path
        self._stats_path = self.data_dir / '.norm_stats.json'
        # pos/vel/goal index selections
        self.pos_indices = (0, 1, 4)
        self.vel_indices = (0, 1)
        self.goal_indices = (0, 1, 4)
        self.accel_indices = (0, 1)

        # Try to load existing stats, otherwise compute them
        self.stats = None
        if self._stats_path.exists():
            try:
                with open(self._stats_path, 'r') as fh:
                    self.stats = json.load(fh)
            except Exception:
                self.stats = None
        if self.stats is None:
            self.stats = self._compute_stats()
            try:
                with open(self._stats_path, 'w') as fh:
                    json.dump(self.stats, fh)
            except Exception as e:
                print("Warning: failed to save normalization stats:", e)
                

    def _compute_stats(self):
        # Streaming computation of mean and std for selected indices
        pos_sum = np.zeros(len(self.pos_indices), dtype=np.float64)
        pos_sumsq = np.zeros(len(self.pos_indices), dtype=np.float64)
        vel_sum = np.zeros(len(self.vel_indices), dtype=np.float64)
        vel_sumsq = np.zeros(len(self.vel_indices), dtype=np.float64)
        accel_sum = np.zeros(len(self.accel_indices), dtype=np.float64)
        accel_sumsq = np.zeros(len(self.accel_indices), dtype=np.float64)
        goal_sum = np.zeros(len(self.goal_indices), dtype=np.float64)
        goal_sumsq = np.zeros(len(self.goal_indices), dtype=np.float64)
        count = 0

        for f in tqdm(self.files, desc='Computing normalization stats'):
            try:
                with np.load(f, allow_pickle=True) as d:
                    frames = d['frames']
                    for fr in frames:
                        if isinstance(fr, np.ndarray) and fr.dtype == object:
                            fr = fr.item()
                        # pos
                        pr = fr.get('pos', None)
                        pr_arr = np.array(pr, dtype=np.float32).ravel()

                        pvals = []
                        for i, idx in enumerate(self.pos_indices):
                            pvals.append(float(pr_arr[idx]) if idx < pr_arr.size else 0.0)
                        pvals = np.array(pvals, dtype=np.float64)

                        # vel
                        vr = fr.get('vel', None)
                        vr_arr = np.array(vr, dtype=np.float32).ravel()
                        
                        vvals = []
                        for i, idx in enumerate(self.vel_indices):
                            vvals.append(float(vr_arr[idx]) if idx < vr_arr.size else 0.0)
                        vvals = np.array(vvals, dtype=np.float64)

                        # accel
                        ar = fr.get('accel', None)
                        ar_arr = np.array(ar, dtype=np.float32).ravel()
                        avals = []
                        for i, idx in enumerate(self.accel_indices):
                            avals.append(float(ar_arr[idx]) if idx < ar_arr.size else 0.0)
                        avals = np.array(avals, dtype=np.float64)

                        # goal
                        gr = fr.get('goal', None)
                        gr_arr = np.array(gr, dtype=np.float32).ravel()

                        gvals = []
                        for i, idx in enumerate(self.goal_indices):
                            gvals.append(float(gr_arr[idx]) if idx < gr_arr.size else 0.0)
                        gvals = np.array(gvals, dtype=np.float64)

                        pos_sum += pvals
                        pos_sumsq += pvals * pvals
                        vel_sum += vvals
                        vel_sumsq += vvals * vvals
                        accel_sum += avals
                        accel_sumsq += avals * avals
                        goal_sum += gvals
                        goal_sumsq += gvals * gvals
                        count += 1
            except Exception:
                continue

        pos_mean = (pos_sum / count).tolist()
        pos_var = (pos_sumsq / count) - np.square(np.array(pos_mean, dtype=np.float64))
        pos_std = np.sqrt(np.maximum(pos_var, 1e-6)).tolist()

        vel_mean = (vel_sum / count).tolist()
        vel_var = (vel_sumsq / count) - np.square(np.array(vel_mean, dtype=np.float64))
        vel_std = np.sqrt(np.maximum(vel_var, 1e-6)).tolist()

        accel_mean = (accel_sum / count).tolist()
        accel_var = (accel_sumsq / count) - np.square(np.array(accel_mean, dtype=np.float64))
        accel_std = np.sqrt(np.maximum(accel_var, 1e-6)).tolist()

        goal_mean = (goal_sum / count).tolist()
        goal_var = (goal_sumsq / count) - np.square(np.array(goal_mean, dtype=np.float64))
        goal_std = np.sqrt(np.maximum(goal_var, 1e-6)).tolist()

        # include imagenet rgb stats
        rgb_mean = [0.485, 0.456, 0.406]
        rgb_std = [0.229, 0.224, 0.225]

        return {
            'pos_mean': pos_mean,
            'pos_std': pos_std,
            'vel_mean': vel_mean,
            'vel_std': vel_std,
            'accel_mean': accel_mean,
            'accel_std': accel_std,
            'goal_mean': goal_mean,
            'goal_std': goal_std,
            'rgb_mean': rgb_mean,
            'rgb_std': rgb_std,
        }

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fpath, local_idx = self.index[idx]
        try:
            with np.load(fpath, allow_pickle=True) as d:
                frames = d['frames']
                frame = frames[local_idx]
        except Exception as e:
            raise RuntimeError(f"Failed to load frame idx {local_idx} from {fpath}: {e}")

        if isinstance(frame, np.ndarray) and frame.dtype == object:
            frame = frame.item()

        images = frame.get('images')
        actions = frame.get('actions')

        imgs = []
        for i in range(images.shape[0]):
            im = Image.fromarray(images[i])

            if self.transform:
                im = self.transform(im)
            else:
                im = T.ToTensor()(im)
            imgs.append(im)

        # stack -> (4, 3, H, W)
        imgs = torch.stack(imgs, dim=0)

        a = np.array(actions, dtype=np.float32)


        actions = torch.from_numpy(a)
        if isinstance(frame, dict):
            # Read pos, vel, goal and construct tensors with requested indices:
            # pos -> [pos[0], pos[1], pos[4]]  (3-dim)
            # vel -> [vel[0], vel[1]]          (2-dim)
            # goal -> if carla Transform: (location.x, location.y, rotation.yaw)
            #         else: treat as list/array and pick indices [0,1,4]

            # POS
            pos_raw = frame.get('pos', None)
            if pos_raw is None:
                raise RuntimeError(f"Missing 'pos' in frame {fpath} idx {local_idx}")

            pos_arr = np.array(pos_raw, dtype=np.float32).ravel()

            p_sel = []
            for i in self.pos_indices:
                p_sel.append(float(pos_arr[i]))
            pos_t = torch.from_numpy(np.array(p_sel, dtype=np.float32))

            pos_t = pos_t.float()

            workspace_bounds = torch.tensor(WORKSPACE_BOUNDS, dtype=torch.float32)
            # normalize position (x, y to -1 and 1 within workspace bounds)
            # (x - x_min) / (x_max - x_min)  -> [0, 1], then scale to [-1, 1]
            pos_t[:2] = 2 * (pos_t[:2] - workspace_bounds[[0, 2]]) / (workspace_bounds[[1, 3]] - workspace_bounds[[0, 2]]) - 1
            pos_t[2] = pos_t[2] / 180 # leave yaw scaled from -1 to 1

            # VEL
            vel_raw = frame.get('vel', None)
            vel_arr = np.array(vel_raw, dtype=np.float32).ravel()

            v_sel = []
            for i in self.vel_indices:
                v_sel.append(float(vel_arr[i]))
            vel_t = torch.from_numpy(np.array(v_sel, dtype=np.float32))

            vel_t = vel_t.float()
            vm = torch.tensor(self.stats['vel_mean'], dtype=torch.float32)
            vs = torch.tensor(self.stats['vel_std'], dtype=torch.float32)
            vel_t = (vel_t - vm) / vs

            # GOAL
            goal_raw = frame.get('goal', None)

            # If goal is a CARLA Transform-like object
            goal_sel = []
            g_arr = np.array(goal_raw, dtype=np.float32).ravel()
            for i in self.goal_indices:
                goal_sel.append(float(g_arr[i]))

            goal_t = torch.from_numpy(np.array(goal_sel, dtype=np.float32))

            goal_t = goal_t.float()
            goal_t[:2] = 2 * (goal_t[:2] - workspace_bounds[[0, 2]]) / (workspace_bounds[[1, 3]] - workspace_bounds[[0, 2]]) - 1 # this scales from -1 to 1
            goal_t[2] = goal_t[2] / 180  # leave yaw scaled from -1 to 1

            # Acceleration
            accel_raw = frame.get('accel', None)
            if accel_raw is None:
                raise Exception("accel data not found")
            accel_arr = np.array(accel_raw, dtype=np.float32).ravel()
            accel_t = torch.tensor(accel_arr[:2], dtype=torch.float32) # just get x and y components of accel
            am = torch.tensor(self.stats['accel_mean'], dtype=torch.float32)
            as_ = torch.tensor(self.stats['accel_std'], dtype=torch.float32)
            accel_t = (accel_t - am) / as_

            # Trailer Angle
            trailer_angle_raw = frame.get('trailer_angle', None)
            if trailer_angle_raw is None:
                raise Exception("trailer_angle data not found")
            trailer_angle_arr = np.array(trailer_angle_raw, dtype=np.float32).ravel()
            # ensure trailer angle is a 1-D tensor of length 1 so stacking yields (B,1)
            ta_val = float(trailer_angle_arr[0]) if trailer_angle_arr.size > 0 else 0.0
            trailer_angle_t = torch.tensor([ta_val], dtype=torch.float32)
            trailer_angle_t /= 180.0  # scale to [-1, 1]

            # Reverse
            rev_raw = frame.get('reverse', None)
            if rev_raw is None:
                raise Exception("reverse data not found")
            # ensure reverse flag is 1-D tensor (B,1) after stacking
            rev_t = torch.tensor([float(rev_raw)], dtype=torch.float32)

            return imgs, actions, pos_t, vel_t, accel_t, trailer_angle_t, rev_t, goal_t


def collate_fn(batch):
    # Expect each element to be (imgs, actions, pos, vel, accel, trailer_angle, reverse, goal)
    if len(batch[0]) != 8:
        raise RuntimeError(f"Expected batch elements of length 8 (imgs, actions, pos, vel, accel, trailer_angle, reverse, goal), got {len(batch[0])}")

    imgs = [b[0] for b in batch]
    acts = [b[1] for b in batch]
    poss = [b[2] for b in batch]
    vels = [b[3] for b in batch]
    accels = [b[4] for b in batch]
    trailer_angles = [b[5] for b in batch]
    revs = [b[6] for b in batch]
    goals = [b[7] for b in batch]

    imgs = torch.stack(imgs, dim=0)
    acts = torch.stack(acts, dim=0)
    poss = torch.stack(poss, dim=0)
    vels = torch.stack(vels, dim=0)
    accels = torch.stack(accels, dim=0)
    trailer_angles = torch.stack(trailer_angles, dim=0)
    revs = torch.stack(revs, dim=0)
    goals = torch.stack(goals, dim=0)
    return imgs, acts, poss, vels, accels, trailer_angles, revs, goals


def train_epoch(model, loader, optim, device, desc=None, writer=None, global_step_base=0):
    """
    Train for one epoch using negative log-likelihood loss computed from the
    model's predicted (mean, var) outputs. The first 3 outputs are treated as
    continuous (Normal) and the last 2 as Bernoulli (binary).
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    it = loader
    if desc is None:
        pbar = None
    else:
        pbar = tqdm(total=len(loader), desc=desc, leave=False)

    for batch_idx, batch in enumerate(it):
        imgs, actions, poss, vels, accels, trailer_angles, revs, goals = batch

        # move to device; if pin_memory=True in DataLoader we can use non_blocking
        imgs = imgs.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        poss = poss.to(device, non_blocking=True)
        vels = vels.to(device, non_blocking=True)
        accels = accels.to(device, non_blocking=True)
        trailer_angles = trailer_angles.to(device, non_blocking=True)
        revs = revs.to(device, non_blocking=True)
        goals = goals.to(device, non_blocking=True)

        # forward/backward (optionally with AMP)
        from contextlib import nullcontext
        ctx = nullcontext()

        with ctx:
            mean, var, _ = model(imgs, goals-poss, vels, accels, trailer_angles, revs)

        cont_mean = mean[:, :3]
        cont_logvar = var[:, :3]

        cont_logvar = torch.clamp(cont_logvar, min=-10.0, max=2.0)
        cont_std = torch.exp(0.5 * cont_logvar)

        cont_std = torch.clamp(cont_std, min=1e-3)

        bin_prob = mean[:, 3:] # last two actions are binary
        bin_prob = torch.clamp(bin_prob, min=1e-6, max=1.0 - 1e-6)

        cont_target = actions[:, :3]

        bin_target = actions[:, 3:]
        #print("revs shape:", revs.shape)
        #print(bin_target[:, 0].shape)
        bin_target[:, 0] = revs[:, 0] # set reverse action to be a 1 or a 0 based on the rev flag (no longer based on toggle)

        normal = torch.distributions.Normal(cont_mean, cont_std)
        logp_cont = normal.log_prob(cont_target)
        logp_cont = logp_cont.sum(dim=1)

        # compute log-prob for binary using Bernoulli
        bern = torch.distributions.Bernoulli(probs=bin_prob)

        logp_bin = bern.log_prob(bin_target)
        logp_bin = logp_bin.sum(dim=1)

        logp = logp_cont + logp_bin
        loss = -logp.mean()

        optim.zero_grad()
        loss.backward()
        optim.step()

        batch_size = imgs.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        # per-batch tensorboard logging (if writer provided)
        if writer is not None:
            step = global_step_base + batch_idx
            try:
                writer.add_scalar('loss/train_batch', float(loss.item()), step)
                # log continuous std stats (first three dims)
                writer.add_scalar('cont_std/min', float(cont_std[:, :3].min().item()), step)
                writer.add_scalar('cont_std/mean', float(cont_std[:, :3].mean().item()), step)
                writer.add_scalar('cont_std/max', float(cont_std[:, :3].max().item()), step)
            except Exception:
                pass
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()
    return total_loss / max(1, total_samples)


def eval_epoch(model, loader, device, desc=None, writer=None, global_step_base=0):
    """
    Evaluate one epoch using negative log-likelihood from the model's
    (mean, var) outputs. Mirrors train_epoch behavior but without gradient steps.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    pbar = None
    if desc is not None:
        pbar = tqdm(total=len(loader), desc=desc, leave=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            imgs, actions, poss, vels, accels, trailer_angles, revs, goals = batch

            imgs = imgs.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            poss = poss.to(device, non_blocking=True)
            vels = vels.to(device, non_blocking=True)
            accels = accels.to(device, non_blocking=True)
            trailer_angles = trailer_angles.to(device, non_blocking=True)
            revs = revs.to(device, non_blocking=True)
            goals = goals.to(device, non_blocking=True)

            from contextlib import nullcontext
            ctx = nullcontext()

            with ctx:
                mean, var, _ = model(imgs, goals-poss, vels, accels, trailer_angles, revs)

            cont_mean = mean[:, :3]
            cont_logvar = var[:, :3]
            cont_logvar = torch.clamp(cont_logvar, min=-20.0, max=2.0)
            cont_std = torch.exp(0.5 * cont_logvar)

            bin_prob = mean[:, 3:]

            cont_target = actions[:, :3]
            bin_target = actions[:, 3:]
            bin_target[:, 0] = revs[:, 0]

            normal = torch.distributions.Normal(cont_mean, cont_std)
            logp_cont = normal.log_prob(cont_target).sum(dim=1)

            bern = torch.distributions.Bernoulli(probs=bin_prob)
            logp_bin = bern.log_prob(bin_target).sum(dim=1)

            logp = logp_cont + logp_bin
            loss = -logp.mean()

            batch_size = imgs.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            # per-batch tensorboard logging (if writer provided)
            if writer is not None:
                step = global_step_base + batch_idx
                try:
                    writer.add_scalar('loss/val_batch', float(loss.item()), step)
                    writer.add_scalar('cont_std_val/min', float((torch.exp(0.5*cont_logvar)[:, :3]).min().item()), step)
                    writer.add_scalar('cont_std_val/mean', float((torch.exp(0.5*cont_logvar)[:, :3]).mean().item()), step)
                    writer.add_scalar('cont_std_val/max', float((torch.exp(0.5*cont_logvar)[:, :3]).max().item()), step)
                except Exception:
                    pass
            if pbar is not None:
                pbar.update(1)
    if pbar is not None:
        pbar.close()
    return total_loss / max(1, total_samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='path to data root containing timestamp folders')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--pin_memory', action='store_true', help='use pinned memory for DataLoader to speed host->device transfers')
    parser.add_argument('--persistent_workers', action='store_true', help='keep DataLoader workers alive between epochs (PyTorch >= 1.7)')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='number of batches loaded in advance by each worker')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--pretrained', action='store_true', help='use imagenet pretrained resnet')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='runs', help='tensorboard log directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to model checkpoint to resume from')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = NpzFramesDataset(args.data_dir, transform=transform)
    if len(dataset) == 0:
        raise RuntimeError(f"No frames found in {args.data_dir}. Check that .npz files exist and contain 'frames'.")

    val_frac = 0.05
    n_val = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    # Clamp workers to a sensible number and attach worker_init_fn to reduce
    # thread oversubscription and improve stability. If args.workers is large
    # (e.g. 26), it can exhaust system resources or trigger segfaults in
    # native libraries.
    cpu_count = os.cpu_count() or 1
    max_workers = max(0, min(args.workers, max(1, cpu_count - 1)))
    val_workers = max(0, min(max(1, args.workers // 2), cpu_count - 1))
    if args.workers > cpu_count - 1:
        print(f"Warning: requested --workers={args.workers} but only {cpu_count} CPUs detected; using {max_workers} workers")

    dl_kwargs = dict(collate_fn=collate_fn, worker_init_fn=_worker_init_fn)
    if max_workers > 0:
        dl_kwargs.update(dict(num_workers=max_workers, pin_memory=args.pin_memory))
        # only set prefetch_factor / persistent_workers when workers > 0 and arg provided
        dl_kwargs['prefetch_factor'] = args.prefetch_factor
        dl_kwargs['persistent_workers'] = args.persistent_workers

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **dl_kwargs)

    val_dl_kwargs = dict(collate_fn=collate_fn, worker_init_fn=_worker_init_fn)
    if val_workers > 0:
        val_dl_kwargs.update(dict(num_workers=val_workers, pin_memory=args.pin_memory))
        val_dl_kwargs['prefetch_factor'] = max(1, args.prefetch_factor // 2)
        val_dl_kwargs['persistent_workers'] = args.persistent_workers

    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **val_dl_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TruckNet(pretrained=args.pretrained).to(device)
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt.get('model_state', ckpt)
        model.load_state_dict(state)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float('inf')
    # TensorBoard SummaryWriter (optional)
    writer = None
    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir=args.log_dir)
        print(f"TensorBoard logging to: {args.log_dir}")
    try:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, device, desc=f"Train E{epoch}", writer=writer, global_step_base=(epoch-1)*len(train_loader) if writer is not None else 0)
            val_loss = eval_epoch(model, val_loader, device, desc=f"Val E{epoch}", writer=writer, global_step_base=(epoch-1)*len(val_loader) if writer is not None else 0)
            dt = time.time() - t0

            print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  time={dt:.1f}s")

            # log to tensorboard if available
            if writer is not None:
                try:
                    writer.add_scalar('loss/train', float(train_loss), epoch)
                    writer.add_scalar('loss/val', float(val_loss), epoch)
                    # log learning rate
                    try:
                        lr = optimizer.param_groups[0].get('lr', None)
                        if lr is not None:
                            writer.add_scalar('lr', float(lr), epoch)
                    except Exception:
                        pass
                except Exception:
                    pass

            ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'args': vars(args),
            }
            ckpt_path = os.path.join(args.save_dir, f"bc_epoch_{epoch:03d}.pt")
            torch.save(ckpt, ckpt_path)

            if val_loss < best_val:
                best_val = val_loss
                best_path = os.path.join(args.save_dir, 'best.pt')
                torch.save(ckpt, best_path)
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        try:
            if writer is not None:
                writer.close()
        except Exception:
            pass

if __name__ == '__main__':
    main()
