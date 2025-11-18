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
            except Exception:
                pass

    def _compute_stats(self):
        # Streaming computation of mean and std for selected indices
        pos_sum = np.zeros(len(self.pos_indices), dtype=np.float64)
        pos_sumsq = np.zeros(len(self.pos_indices), dtype=np.float64)
        vel_sum = np.zeros(len(self.vel_indices), dtype=np.float64)
        vel_sumsq = np.zeros(len(self.vel_indices), dtype=np.float64)
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
                        if pr is None:
                            continue
                        try:
                            pr_arr = np.array(pr, dtype=np.float32).ravel()
                        except Exception:
                            try:
                                px = float(pr.location.x)
                                py = float(pr.location.y)
                                pyaw = float(pr.rotation.yaw)
                                pr_arr = np.array([px, py, 0.0, 0.0, pyaw], dtype=np.float32)
                            except Exception:
                                continue
                        pvals = []
                        for i, idx in enumerate(self.pos_indices):
                            pvals.append(float(pr_arr[idx]) if idx < pr_arr.size else 0.0)
                        pvals = np.array(pvals, dtype=np.float64)

                        # vel
                        vr = fr.get('vel', None)
                        if vr is None:
                            continue
                        try:
                            vr_arr = np.array(vr, dtype=np.float32).ravel()
                        except Exception:
                            try:
                                vx = float(vr.x)
                                vy = float(vr.y)
                                vr_arr = np.array([vx, vy, 0.0], dtype=np.float32)
                            except Exception:
                                continue
                        vvals = []
                        for i, idx in enumerate(self.vel_indices):
                            vvals.append(float(vr_arr[idx]) if idx < vr_arr.size else 0.0)
                        vvals = np.array(vvals, dtype=np.float64)

                        # goal
                        gr = fr.get('goal', None)
                        if gr is None:
                            gr = fr.get('goal_pos', None)
                        if gr is None:
                            continue
                        try:
                            if hasattr(gr, 'location') and hasattr(gr, 'rotation'):
                                gx = float(gr.location.x)
                                gy = float(gr.location.y)
                                gyaw = float(gr.rotation.yaw)
                                gr_arr = np.array([gx, gy, 0.0, 0.0, gyaw], dtype=np.float32)
                            else:
                                gr_arr = np.array(gr, dtype=np.float32).ravel()
                        except Exception:
                            continue
                        gvals = []
                        for i, idx in enumerate(self.goal_indices):
                            gvals.append(float(gr_arr[idx]) if idx < gr_arr.size else 0.0)
                        gvals = np.array(gvals, dtype=np.float64)

                        pos_sum += pvals
                        pos_sumsq += pvals * pvals
                        vel_sum += vvals
                        vel_sumsq += vvals * vvals
                        goal_sum += gvals
                        goal_sumsq += gvals * gvals
                        count += 1
            except Exception:
                continue

        if count == 0:
            # fallback defaults
            pos_mean = [0.0] * len(self.pos_indices)
            pos_std = [1.0] * len(self.pos_indices)
            vel_mean = [0.0] * len(self.vel_indices)
            vel_std = [1.0] * len(self.vel_indices)
            goal_mean = [0.0] * len(self.goal_indices)
            goal_std = [1.0] * len(self.goal_indices)
        else:
            pos_mean = (pos_sum / count).tolist()
            pos_var = (pos_sumsq / count) - np.square(np.array(pos_mean, dtype=np.float64))
            pos_std = np.sqrt(np.maximum(pos_var, 1e-6)).tolist()

            vel_mean = (vel_sum / count).tolist()
            vel_var = (vel_sumsq / count) - np.square(np.array(vel_mean, dtype=np.float64))
            vel_std = np.sqrt(np.maximum(vel_var, 1e-6)).tolist()

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
            # Re-raise a clearer error which will be propagated to the main
            # process so the DataLoader doesn't silently segfault.
            raise RuntimeError(f"Failed to load frame idx {local_idx} from {fpath}: {e}")

        # frame may be a dict-like object (saved as Python objects)
        if isinstance(frame, np.ndarray) and frame.dtype == object:
            frame = frame.item()

        images = frame.get('images')
        actions = frame.get('actions')

        if images is None or actions is None:
            raise RuntimeError(f"Malformed frame in {fpath} idx {local_idx}")

        # images: (4, H, W, 3) uint8
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

        # normalize actions and convert to tensor
        # steering (index 2) in recorded bags may be in range [-2, 2]; map to [0,1]
        a = np.array(actions, dtype=np.float32)
        if a.size > 2:
            try:
                a[2] = float(np.clip((a[2] + 2.0) / 4.0, 0.0, 1.0))
            except Exception:
                # if for any reason conversion fails, leave as-is
                pass

        actions = torch.from_numpy(a)
        if isinstance(frame, dict):
            # Read pos, vel, goal and construct tensors with requested indices:
            # pos -> [pos[0], pos[1], pos[4]]  (3-dim)
            # vel -> [vel[0], vel[1]]          (2-dim)
            # goal -> if carla Transform: (location.x, location.y, rotation.yaw)
            #         else: treat as list/array and pick indices [0,1,4]

            # POS
            pos_raw = None
            if isinstance(frame, dict):
                pos_raw = frame.get('pos', None)
            if pos_raw is None:
                raise RuntimeError(f"Missing 'pos' in frame {fpath} idx {local_idx}")
            pos_arr = None
            try:
                pos_arr = np.array(pos_raw, dtype=np.float32).ravel()
            except Exception:
                # try to access attributes
                try:
                    px = float(pos_raw.location.x)
                    py = float(pos_raw.location.y)
                    pyaw = float(pos_raw.rotation.yaw)
                    pos_arr = np.array([px, py, 0.0, 0.0, pyaw], dtype=np.float32)
                except Exception:
                    raise RuntimeError("Unable to parse 'pos' field")
            p_sel = []
            for i in self.pos_indices:
                if i < pos_arr.size:
                    p_sel.append(float(pos_arr[i]))
                else:
                    p_sel.append(0.0)
            pos_t = torch.from_numpy(np.array(p_sel, dtype=np.float32))

            # Normalize position using dataset mean/std (z-score). Clamp to reasonable range.
            try:
                pos_t = pos_t.float()
                pm = torch.tensor(self.stats['pos_mean'], dtype=torch.float32)
                ps = torch.tensor(self.stats['pos_std'], dtype=torch.float32)
                pos_t = (pos_t - pm) / ps
                pos_t = torch.clamp(pos_t, -5.0, 5.0)
            except Exception:
                pass

            # VEL
            vel_raw = None
            if isinstance(frame, dict):
                vel_raw = frame.get('vel', None)
            if vel_raw is None:
                raise RuntimeError(f"Missing 'vel' in frame {fpath} idx {local_idx}")
            try:
                vel_arr = np.array(vel_raw, dtype=np.float32).ravel()
            except Exception:
                try:
                    vx = float(vel_raw.x)
                    vy = float(vel_raw.y)
                    vel_arr = np.array([vx, vy, 0.0], dtype=np.float32)
                except Exception:
                    raise RuntimeError("Unable to parse 'vel' field")
            v_sel = []
            for i in self.vel_indices:
                if i < vel_arr.size:
                    v_sel.append(float(vel_arr[i]))
                else:
                    v_sel.append(0.0)
            vel_t = torch.from_numpy(np.array(v_sel, dtype=np.float32))

            # Normalize velocity using dataset mean/std (z-score). Clamp to reasonable range.
            try:
                vel_t = vel_t.float()
                vm = torch.tensor(self.stats['vel_mean'], dtype=torch.float32)
                vs = torch.tensor(self.stats['vel_std'], dtype=torch.float32)
                vel_t = (vel_t - vm) / vs
                vel_t = torch.clamp(vel_t, -5.0, 5.0)
            except Exception:
                pass

            # GOAL
            goal_raw = None
            if isinstance(frame, dict):
                goal_raw = frame.get('goal', None)
                if goal_raw is None:
                    goal_raw = frame.get('goal_pos', None)
            if goal_raw is None:
                raise RuntimeError(f"Missing 'goal' in frame {fpath} idx {local_idx}")

            # If goal is a CARLA Transform-like object
            goal_sel = []
            if hasattr(goal_raw, 'location') and hasattr(goal_raw, 'rotation'):
                try:
                    gx = float(goal_raw.location.x)
                    gy = float(goal_raw.location.y)
                    gyaw = float(goal_raw.rotation.yaw)
                    arr = np.array([gx, gy, 0.0, 0.0, gyaw], dtype=np.float32)
                    for i in self.goal_indices:
                        goal_sel.append(float(arr[i]))
                except Exception:
                    # fallback to array conversion
                    g_arr = np.array(goal_raw, dtype=np.float32).ravel()
                    for i in self.goal_indices:
                        goal_sel.append(float(g_arr[i]) if i < g_arr.size else 0.0)
            else:
                try:
                    g_arr = np.array(goal_raw, dtype=np.float32).ravel()
                except Exception:
                    raise RuntimeError("Unable to parse 'goal' field")
                for i in self.goal_indices:
                    goal_sel.append(float(g_arr[i]) if i < g_arr.size else 0.0)

            goal_t = torch.from_numpy(np.array(goal_sel, dtype=np.float32))

            # Normalize goal using dataset mean/std (z-score). Clamp to reasonable range.
            try:
                goal_t = goal_t.float()
                gm = torch.tensor(self.stats['goal_mean'], dtype=torch.float32)
                gs = torch.tensor(self.stats['goal_std'], dtype=torch.float32)
                goal_t = (goal_t - gm) / gs
                goal_t = torch.clamp(goal_t, -5.0, 5.0)
            except Exception:
                pass

            return imgs, actions, pos_t, vel_t, goal_t





def collate_fn(batch):
    # Expect each element to be (imgs, actions, pos, vel, goal)
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
    if len(batch[0]) != 5:
        raise RuntimeError(f"Expected batch elements of length 5 (imgs, actions, pos, vel, goal), got {len(batch[0])}")

    imgs = [b[0] for b in batch]
    acts = [b[1] for b in batch]
    poss = [b[2] for b in batch]
    vels = [b[3] for b in batch]
    goals = [b[4] for b in batch]

    imgs = torch.stack(imgs, dim=0)
    acts = torch.stack(acts, dim=0)
    poss = torch.stack(poss, dim=0)
    vels = torch.stack(vels, dim=0)
    goals = torch.stack(goals, dim=0)
    return imgs, acts, poss, vels, goals


def train_epoch(model, loader, optim, device, loss_fn_cont, loss_fn_bce, desc=None, scaler=None, use_amp=False):
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

    for batch in it:
        imgs, actions, poss, vels, goals = batch

        # move to device; if pin_memory=True in DataLoader we can use non_blocking
        imgs = imgs.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        poss = poss.to(device, non_blocking=True)
        vels = vels.to(device, non_blocking=True)
        goals = goals.to(device, non_blocking=True)

        # forward/backward (optionally with AMP)
        from contextlib import nullcontext
        if use_amp:
            # create a real autocast context manager
            ctx = torch.cuda.amp.autocast()
        else:
            ctx = nullcontext()

        with ctx:
            # model now returns (mean, var)
            mean, var, _ = model(imgs, poss, vels, goals)

        # continuous: first 3 dims. The model returns log-variance for these dims.
        cont_mean = mean[:, :3]
        cont_logvar = var[:, :3]
        # clamp logvar for numerical stability (prevents extremely large/small std)
        # tightened min so std can't become extremely small causing huge log-probs
        cont_logvar = torch.clamp(cont_logvar, min=-10.0, max=2.0)
        cont_std = torch.exp(0.5 * cont_logvar)
        # enforce a minimum std to avoid division by very small numbers / extreme gradients
        cont_std = torch.clamp(cont_std, min=1e-3)

        # binary: last 2 dims (probabilities)
        bin_prob = mean[:, 3:]
        # clamp probabilities away from 0/1 to avoid -inf log-probabilities
        bin_prob = torch.clamp(bin_prob, min=1e-6, max=1.0 - 1e-6)

        cont_target = actions[:, :3]
        bin_target = actions[:, 3:]

        # compute log-prob for continuous using Normal
        normal = torch.distributions.Normal(cont_mean, cont_std)
        logp_cont = normal.log_prob(cont_target)  # (B,3)
        logp_cont = logp_cont.sum(dim=1)  # per-sample

        # compute log-prob for binary using Bernoulli
        bern = torch.distributions.Bernoulli(probs=bin_prob)
        # Bernoulli.log_prob supports float targets (0/1)
        logp_bin = bern.log_prob(bin_target)  # (B,2)
        logp_bin = logp_bin.sum(dim=1)

        # total negative log-likelihood (average over batch)
        logp = logp_cont + logp_bin
        loss = -logp.mean()

        optim.zero_grad()
        # catch non-finite loss early and print diagnostics
        if not torch.isfinite(loss):
            print("Non-finite loss detected during training. Dumping batch stats:")
            try:
                print(f"loss={loss}")
                print(f"cont_mean min/max: {cont_mean.min().item()}/{cont_mean.max().item()}")
                print(f"cont_std min/max: {cont_std.min().item()}/{cont_std.max().item()}")
                print(f"cont_target min/max: {cont_target.min().item()}/{cont_target.max().item()}")
                print(f"bin_prob min/max: {bin_prob.min().item()}/{bin_prob.max().item()}")
                print(f"bin_target min/max: {bin_target.min().item()}/{bin_target.max().item()}")
            except Exception:
                pass
            # skip this batch to avoid corrupting optimizer state
            continue

        if scaler is not None:
            # scale loss and backprop
            scaler.scale(loss).backward()
            # unscale before clipping
            try:
                scaler.unscale_(optim)
            except Exception:
                pass
            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            except Exception:
                pass
            try:
                scaler.step(optim)
                scaler.update()
            except Exception:
                # if step failed, skip optimizer update to keep things stable
                optim.zero_grad()
        else:
            loss.backward()
            # gradient clipping prevents exploding gradients from causing large parameter jumps
            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            except Exception:
                pass
            optim.step()

        batch_size = imgs.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()
    return total_loss / max(1, total_samples)


def eval_epoch(model, loader, device, loss_fn_cont, loss_fn_bce, desc=None, use_amp=False):
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
        for batch in loader:
            imgs, actions, poss, vels, goals = batch

            imgs = imgs.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            poss = poss.to(device, non_blocking=True)
            vels = vels.to(device, non_blocking=True)
            goals = goals.to(device, non_blocking=True)

            from contextlib import nullcontext
            if use_amp:
                ctx = torch.cuda.amp.autocast()
            else:
                ctx = nullcontext()

            with ctx:
                mean, var, _ = model(imgs, poss, vels, goals)

            # model returns log-variance for continuous outputs
            cont_mean = mean[:, :3]
            cont_logvar = var[:, :3]
            cont_logvar = torch.clamp(cont_logvar, min=-20.0, max=2.0)
            cont_std = torch.exp(0.5 * cont_logvar)

            bin_prob = mean[:, 3:]

            cont_target = actions[:, :3]
            bin_target = actions[:, 3:]

            normal = torch.distributions.Normal(cont_mean, cont_std)
            logp_cont = normal.log_prob(cont_target).sum(dim=1)

            bern = torch.distributions.Bernoulli(probs=bin_prob)
            logp_bin = bern.log_prob(bin_target).sum(dim=1)

            logp = logp_cont + logp_bin
            loss = -logp.mean()

            batch_size = imgs.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            if pbar is not None:
                pbar.update(1)
    if pbar is not None:
        pbar.close()
    return total_loss / max(1, total_samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='path to data root containing timestamp folders')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--pin_memory', action='store_true', help='use pinned memory for DataLoader to speed host->device transfers')
    parser.add_argument('--persistent_workers', action='store_true', help='keep DataLoader workers alive between epochs (PyTorch >= 1.7)')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='number of batches loaded in advance by each worker')
    parser.add_argument('--amp', action='store_true', help='use automatic mixed precision (CUDA only)')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--pretrained', action='store_true', help='use imagenet pretrained resnet')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no_cuda', action='store_true', help='force CPU even if CUDA is available')
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

    # small validation split
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
        try:
            dl_kwargs['prefetch_factor'] = args.prefetch_factor
            dl_kwargs['persistent_workers'] = args.persistent_workers
        except Exception:
            pass

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **dl_kwargs)

    val_dl_kwargs = dict(collate_fn=collate_fn, worker_init_fn=_worker_init_fn)
    if val_workers > 0:
        val_dl_kwargs.update(dict(num_workers=val_workers, pin_memory=args.pin_memory))
        try:
            val_dl_kwargs['prefetch_factor'] = max(1, args.prefetch_factor // 2)
            val_dl_kwargs['persistent_workers'] = args.persistent_workers
        except Exception:
            pass
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **val_dl_kwargs)

    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    # Use TruckNet (expects rgb, pos(3), vel(2), goal(3))
    model = TruckNet(pretrained=args.pretrained).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn_cont = nn.MSELoss()
    # TruckNet outputs sigmoid-activated values; use BCELoss for binary outputs
    loss_fn_bce = nn.BCELoss()

    # set up AMP scaler if requested and CUDA is available
    scaler = None
    use_amp = bool(args.amp and (not args.no_cuda) and torch.cuda.is_available())
    if use_amp:
        try:
            scaler = torch.cuda.amp.GradScaler()
            print('Using AMP (mixed precision) for training')
        except Exception:
            scaler = None

    best_val = float('inf')
    try:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, device, loss_fn_cont, loss_fn_bce, desc=f"Train E{epoch}", scaler=scaler, use_amp=use_amp)
            val_loss = eval_epoch(model, val_loader, device, loss_fn_cont, loss_fn_bce, desc=f"Val E{epoch}", use_amp=use_amp)
            dt = time.time() - t0

            print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  time={dt:.1f}s")

            # save checkpoint
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
    except RuntimeError as e:
        msg = str(e)
        if 'out of memory' in msg:
            print('\nRuntimeError: CUDA out of memory encountered during training.')
            print('Suggestions:')
            print(" - Reduce --batch_size (try 2 or 1) and/or --img_size (e.g. 160 or 128).")
            print(" - Run with --no_cuda to force CPU (slower).")
            print(" - Set environment variable PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 to reduce fragmentation.")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        # cuDNN algorithm selection failures
        if 'cuDNN' in msg or 'cudnn' in msg or 'Unable to find a valid cuDNN algorithm' in msg:
            print('\nRuntimeError: cuDNN algorithm selection failure encountered.')
            print('Suggestions:')
            print(' - Try reducing --batch_size and/or --img_size.')
            print(' - Set environment variable PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 to reduce fragmentation.')
            print(' - Try setting torch.backends.cudnn.benchmark = True (enabled by default in this script).')
            print(' - If problem persists, try running with --no_cuda to use CPU or reinstall PyTorch matching the system cuDNN/CUDA.')
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        raise


if __name__ == '__main__':
    main()
