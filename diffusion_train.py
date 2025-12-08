#!/usr/bin/env python3
import argparse
import json
import math
import os
import importlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from diffusion_model import ActionConditionedDiffusion

WORKSPACE_BOUNDS = np.array([-69, 157, -30, 200], dtype=np.float32)

# Optional TensorBoard
SummaryWriter = None
try:
    tb_mod = importlib.import_module("torch.utils.tensorboard")
    SummaryWriter = getattr(tb_mod, "SummaryWriter")
except Exception:
    SummaryWriter = None

def _count_frames(path: str) -> int:
    try:
        with np.load(path, allow_pickle=True) as d:
            if "frames" not in d:
                return 0
            return len(d["frames"])
    except Exception:
        return 0


class DiffusionDataset(Dataset):
    """
    Loads per-step samples from the recorded .npz trajectories (no images).
    Returns (actions[3], condition[9]) where condition = [pos_rel(3), vel(2), accel(2), trailer(1), reverse(1)].
    """

    def __init__(self, data_dir: str, stats_path: str = None, max_files: int = None):
        super().__init__()
        self.data_dir = Path(data_dir)
        files = sorted(self.data_dir.rglob("*.npz"))
        if max_files:
            files = files[:max_files]
        self.files = [str(f) for f in files]

        self.pos_indices = (0, 1, 4)
        self.vel_indices = (0, 1)
        self.accel_indices = (0, 1)
        self.goal_indices = (0, 1, 4)

        self.index = []
        for f in self.files:
            n = _count_frames(f)
            for i in range(n):
                self.index.append((f, i))

        if len(self.index) == 0:
            raise RuntimeError(f"No frames found in {data_dir}")

        self.stats_path = Path(stats_path) if stats_path else (self.data_dir / ".norm_stats.json")
        self.stats = self._load_or_compute_stats()

    def _load_or_compute_stats(self):
        if self.stats_path.exists():
            try:
                with open(self.stats_path, "r") as fh:
                    return json.load(fh)
            except Exception:
                pass
        vel_sum = np.zeros(len(self.vel_indices), dtype=np.float64)
        vel_sumsq = np.zeros(len(self.vel_indices), dtype=np.float64)
        accel_sum = np.zeros(len(self.accel_indices), dtype=np.float64)
        accel_sumsq = np.zeros(len(self.accel_indices), dtype=np.float64)
        count = 0

        for f in self.files:
            try:
                with np.load(f, allow_pickle=True) as d:
                    frames = d["frames"]
                    for fr in frames:
                        if isinstance(fr, np.ndarray) and fr.dtype == object:
                            fr = fr.item()
                        vr = np.array(fr.get("vel", []), dtype=np.float32).ravel()
                        ar = np.array(fr.get("accel", []), dtype=np.float32).ravel()
                        vel = np.array([vr[i] if i < vr.size else 0.0 for i in self.vel_indices], dtype=np.float64)
                        accel = np.array([ar[i] if i < ar.size else 0.0 for i in self.accel_indices], dtype=np.float64)
                        vel_sum += vel
                        vel_sumsq += vel * vel
                        accel_sum += accel
                        accel_sumsq += accel * accel
                        count += 1
            except Exception:
                continue

        count = max(count, 1)
        vel_mean = (vel_sum / count).tolist()
        vel_var = vel_sumsq / count - np.square(np.array(vel_mean, dtype=np.float64))
        vel_std = np.sqrt(np.maximum(vel_var, 1e-6)).tolist()

        accel_mean = (accel_sum / count).tolist()
        accel_var = accel_sumsq / count - np.square(np.array(accel_mean, dtype=np.float64))
        accel_std = np.sqrt(np.maximum(accel_var, 1e-6)).tolist()

        stats = dict(vel_mean=vel_mean, vel_std=vel_std, accel_mean=accel_mean, accel_std=accel_std)
        try:
            with open(self.stats_path, "w") as fh:
                json.dump(stats, fh)
        except Exception:
            pass
        return stats

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        fpath, local_idx = self.index[idx]
        with np.load(fpath, allow_pickle=True) as d:
            frames = d["frames"]
            fr = frames[local_idx]
        if isinstance(fr, np.ndarray) and fr.dtype == object:
            fr = fr.item()

        actions = torch.tensor(np.array(fr["actions"], dtype=np.float32), dtype=torch.float32)
        actions = actions[:3]  # keep throttle/steer/reverse format used in offline data

        def select(arr, inds):
            arr = np.array(arr, dtype=np.float32).ravel()
            return np.array([arr[i] if i < arr.size else 0.0 for i in inds], dtype=np.float32)

        pos = select(fr["pos"], self.pos_indices)
        goal = select(fr["goal"], self.goal_indices)

        # adjust workspace scaling
        pos_t = torch.from_numpy(pos)
        goal_t = torch.from_numpy(goal)
        bounds = torch.tensor(WORKSPACE_BOUNDS, dtype=torch.float32)
        pos_t[:2] = 2 * (pos_t[:2] - bounds[[0, 2]]) / (bounds[[1, 3]] - bounds[[0, 2]]) - 1
        pos_t[2] = pos_t[2] / 180.0
        goal_t[:2] = 2 * (goal_t[:2] - bounds[[0, 2]]) / (bounds[[1, 3]] - bounds[[0, 2]]) - 1
        goal_t[2] = goal_t[2] / 180.0
        pos_rel = goal_t - pos_t

        vel = select(fr["vel"], self.vel_indices)
        accel = select(fr["accel"], self.accel_indices)
        vel = (torch.from_numpy(vel) - torch.tensor(self.stats["vel_mean"], dtype=torch.float32)) / torch.tensor(
            self.stats["vel_std"], dtype=torch.float32
        )
        accel = (torch.from_numpy(accel) - torch.tensor(self.stats["accel_mean"], dtype=torch.float32)) / torch.tensor(
            self.stats["accel_std"], dtype=torch.float32
        )

        trailer = torch.tensor([float(np.array(fr["trailer_angle"]).ravel()[0]) / 180.0], dtype=torch.float32)
        rev = torch.tensor([float(fr.get("reverse", 0.0))], dtype=torch.float32)

        cond = torch.cat([pos_rel, vel, accel, trailer, rev], dim=0)
        return actions, cond


def loss_on_batch(model, batch, device):
    actions, cond = batch
    actions = actions.to(device)
    cond = cond.to(device)

    bsz = actions.shape[0]
    t = torch.randint(0, model.num_diffusion_steps, (bsz,), device=device, dtype=torch.long)
    noise = torch.randn_like(actions)
    sqrt_alpha = model.sqrt_alphas_cumprod[t].unsqueeze(-1)
    sqrt_one_minus = model.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
    x_t = sqrt_alpha * actions + sqrt_one_minus * noise
    pred = model(x_t, t, cond)
    return F.mse_loss(pred, noise)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total = 0
    for batch in loader:
        loss = loss_on_batch(model, batch, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bsz = batch[0].shape[0]
        total_loss += loss.item() * bsz
        total += bsz
    return total_loss / max(1, total)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    total = 0
    for batch in loader:
        loss = loss_on_batch(model, batch, device)
        bsz = batch[0].shape[0]
        total_loss += loss.item() * bsz
        total += bsz
    return total_loss / max(1, total)


def save_checkpoint(model, optimizer, epoch, path):
    ckpt = {
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(ckpt, path)


def main():
    parser = argparse.ArgumentParser(description="Train a diffusion policy over truck actions (state-only).")
    parser.add_argument("--data_dir", type=str, default="data_no_image", help="Path with recorded .npz trajectories")
    parser.add_argument("--stats_path", type=str, default=None, help="Optional path to normalization stats JSON")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_steps", type=int, default=100, help="Diffusion steps (T)")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--time_embed_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_files", type=int, default=None, help="Optional cap on number of .npz files")
    parser.add_argument("--log_dir", type=str, default="runs/diffusion", help="TensorBoard log directory (if available)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    dataset = DiffusionDataset(args.data_dir, stats_path=args.stats_path, max_files=args.max_files)
    n_val = max(1, int(len(dataset) * args.val_frac))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    dl_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=max(0, min(args.num_workers, max(1, (os.cpu_count() or 1) - 1))),
        shuffle=True,
        pin_memory=True,
        drop_last=False,
    )
    train_loader = DataLoader(train_set, **dl_kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=dl_kwargs["num_workers"], pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionConditionedDiffusion(
        action_dim=3,
        cond_dim=9,
        hidden_dim=args.hidden_dim,
        time_embed_dim=args.time_embed_dim,
        num_layers=args.num_layers,
        num_diffusion_steps=args.num_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    writer = None
    if SummaryWriter is not None and args.log_dir:
        try:
            writer = SummaryWriter(log_dir=args.log_dir)
            print(f"TensorBoard logging to {args.log_dir}")
        except Exception as e:
            print(f"Warning: failed to create SummaryWriter: {e}")

    start_epoch = 1
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt.get("epoch", 0) + 1

    best_val = math.inf
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            val_loss = eval_epoch(model, val_loader, device)
            print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

            if writer is not None:
                try:
                    writer.add_scalar("loss/train", float(train_loss), epoch)
                    writer.add_scalar("loss/val", float(val_loss), epoch)
                    lr = optimizer.param_groups[0].get("lr", None)
                    if lr is not None:
                        writer.add_scalar("lr", float(lr), epoch)
                except Exception:
                    pass

            ckpt_path = os.path.join(args.save_dir, "diffusion_latest.pt")
            save_checkpoint(model, optimizer, epoch, ckpt_path)

            if val_loss < best_val:
                best_val = val_loss
                best_path = os.path.join(args.save_dir, "diffusion_best.pt")
                save_checkpoint(model, optimizer, epoch, best_path)
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
