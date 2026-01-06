#!/usr/bin/env python3
from __future__ import print_function

from audioop import avg
import glob
import importlib
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
import torch
import torch.nn.functional as F
from modelv2 import TruckNet
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import torch.nn as nn
import json
from env import TruckEnv
import tqdm
import datetime

# seed for reproducibility
torch.manual_seed(0)

# --------------------------- Buffer (unchanged) ---------------------------
class Buffer:
    def __init__(self, size, obs_dim, act_dim, device):
        self.capacity = size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self.reset()

    def reset(self):
        self.ptr = 0
        self.size_ = 0
        self.obs = torch.zeros((self.capacity, self.obs_dim), dtype=torch.float32, device=self.device)
        self.next_obs = torch.zeros((self.capacity, self.obs_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.capacity, self.act_dim), dtype=torch.float32, device=self.device)
        self.log_probs = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)
        self.values = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)
        self.advantages = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)
        self.returns = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)
        self.iteration = torch.zeros((self.capacity,), dtype=torch.int32, device=self.device)

    @property
    def size(self):
        return self.size_

    def add(self, obs, next_obs, action, log_probs=0.0, reward=0.0, done=0.0, value=0.0,
            advantage=0.0, curr_return=0.0, iteration=0):
        i = self.ptr
        #print(self.obs[i].shape, obs.shape)
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.log_probs[i] = log_probs
        self.rewards[i] = reward
        self.dones[i] = done
        self.values[i] = value
        self.advantages[i] = advantage
        self.returns[i] = curr_return
        self.iteration[i] = iteration
        self.ptr = (self.ptr + 1) % self.capacity
        self.size_ = min(self.size_ + 1, self.capacity)

    def add_batch(self, batch: dict):
        n = batch["obs"].shape[0]
        for j in range(n):
            self.add(
                obs=batch["obs"][j],
                next_obs=batch["next_obs"][j],
                action=batch["actions"][j],
                log_probs=batch["log_probs"][j],
                reward=batch["rewards"][j],
                done=batch["dones"][j],
                value=batch["values"][j],
                advantage=batch["advantages"][j],
                curr_return=batch["returns"][j],
                iteration=batch["iteration"][j]
            )

    def sample(self, num_samples, filter):
        if self.size == 0:
            raise ValueError("The buffer is empty")

        if filter is not None:
            mask = torch.ones(self.size, dtype=torch.bool, device=self.device)
            for key, val_list in filter.items():
                if not hasattr(self, key):
                    raise KeyError(f"Buffer has no field '{key}'")
                attr = getattr(self, key)[:self.size]
                val_tensor = torch.as_tensor(val_list, device=self.device, dtype=attr.dtype)
                mask &= torch.isin(attr, val_tensor)
            idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        else:
            idx = torch.arange(self.size, device=self.device)

        if idx.numel() == 0:
            raise ValueError("No samples match the filter conditions")

        if num_samples is not None and num_samples < idx.numel():
            perm = torch.randperm(idx.numel(), device=self.device)[:num_samples]
            idx = idx[perm]

        return dict(
            obs=self.obs[idx],
            next_obs=self.next_obs[idx],
            actions=self.actions[idx],
            log_probs=self.log_probs[idx],
            rewards=self.rewards[idx],
            dones=self.dones[idx],
            values=self.values[idx],
            advantages=self.advantages[idx],
            returns=self.returns[idx],
            iteration=self.iteration[idx],
        )

WORKSPACE_BOUNDS = np.array([-69, 157, -30, 200])
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = os.path.join(LOG_DIR, f"training_log_{timestamp}.txt")

tb_mod = importlib.import_module('torch.utils.tensorboard')
SummaryWriter = getattr(tb_mod, 'SummaryWriter')
writer = SummaryWriter(log_dir="runs/ppo_singleagent_no_image")

def log_to_file(msg):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def process_observation(obs, vel_mean, vel_std, map_location=0):
    """
    NEW: No accel/rev. obs = (pos_list, vel_list, trailer_angle, goal_list)
    Returns flat tensor [6]: pos_rel(3)+vel(2)+trailer(1)
    """
    pos_list, vel_list, _, trailer_angle, _, goal_list = obs
    
    def sel_list(arr, indices):
        a = np.array(arr, dtype=np.float32).ravel()
        out = [float(a[i]) if i < a.size else 0.0 for i in indices]
        return np.array(out, dtype=np.float32)
    
    pos_list[0] -= map_location * 11000 / 100
    goal_list[0] -= map_location * 11000 / 100
    
    pos = sel_list(pos_list, [0, 1, 4])
    vel = sel_list(vel_list, [0, 1])
    goal = sel_list(goal_list, [0, 1, 4])
    
    # Workspace normalization [-1,1]
    workspace_bounds = np.array(WORKSPACE_BOUNDS, dtype=np.float32)
    pos_norm = np.copy(pos)
    goal_norm = np.copy(goal)
    pos_norm[:2] = 2 * (pos[:2] - workspace_bounds[[0, 2]]) / (workspace_bounds[[1, 3]] - workspace_bounds[[0, 2]]) - 1
    pos_norm[2] = pos[2] / 180.0
    goal_norm[:2] = 2 * (goal[:2] - workspace_bounds[[0, 2]]) / (workspace_bounds[[1, 3]] - workspace_bounds[[0, 2]]) - 1
    goal_norm[2] = goal[2] / 180.0
    
    pos_rel = goal_norm - pos_norm  # [3]
    
    # Z-score normalization for vel
    vel_norm = (np.array(vel) - vel_mean.numpy()) / (vel_std.numpy() + 1e-8)   # FIXED
    
    # Trailer angle
    trailer_norm = np.array([float(trailer_angle)]) / 180.0  # [1]
    
    # Flatten: pos_rel(3) + vel(2) + trailer(1) = 6
    state = np.concatenate([pos_rel, vel_norm, trailer_norm])
    return torch.from_numpy(state).float()  # [6]

class PPOSingleAgent:
    def __init__(self, lr=5e-5, gamma=0.999, gae_lambda=0.99,
                 clip_coef=0.2, vf_coef=0.3, ent_coef=0.005, max_grad_norm=0.5,
                 update_epochs=6, minibatch_size=128, rollout_steps=1024, 
                 policy=None, device="cuda", vel_mean=None, vel_std=None):
        
        self.device = torch.device(device)
        self.obs_dim = 6  # NEW: pos_rel(3)+vel(2)+trailer(1)
        self.act_dim = 2
        
        self.policy = policy.to(self.device) if policy is not None else TruckNet(state_dim=6).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.base_lr = lr
        
        # Normalization stats (no accel)
        self.vel_mean = vel_mean
        self.vel_std = vel_std
        
        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.rollout_steps = rollout_steps
        
        self._rollout_buffer = Buffer(self.rollout_steps * 2, self.obs_dim, self.act_dim, device)
        self._steps_collected = 0
        self._steps_collected_with_curr_policy = 0
        self._policy_iteration = 1
        self._curr_policy_rollout = []

    def act(self, obs):
        if obs.dim() > 1:
            obs = obs.squeeze().flatten()
        
        obs_t = obs.unsqueeze(0).to(self.device)  # ALWAYS [1,6]
        assert obs_t.shape == (1, 6), f"obs_t shape wrong: {obs_t.shape}"
        
        pos = obs_t[:, :3]      # [1,3]
        vel = obs_t[:, 3:5]     # [1,2]
        trailer = obs_t[:, 5:6] # [1,1]
        
        dist, value = self.policy(pos, vel, trailer)
        action = dist.sample()  # [1,2]
        log_prob = dist.log_prob(action)  # [1,2]
        return {
            "action": action.squeeze(0).cpu().numpy(),
            "log_prob": log_prob.item(),
            "value": value.item()
        }


    def value_only(self, obs):
        """For value head warmup"""
        with torch.no_grad():
            obs_t = obs.unsqueeze(0).to(self.device)
            pos = obs_t[:, :3]
            vel = obs_t[:, 3:5]
            trailer = obs_t[:, 5:6]
            _, value = self.policy(pos, vel, trailer)
            return value.squeeze(0).item()

    def step(self, transition: Dict[str, Any]) -> Dict[str, float]:
        # transition now has "obs", "next_obs" directly - NO RAW!
        # self._rollout_buffer.add(
        #     obs=transition["obs"],      # Already processed [6]
        #     next_obs=transition["next_obs"],
        #     action=torch.as_tensor(transition["action"], dtype=torch.float32),
        #     log_probs=transition["log_prob"],
        #     reward=transition["reward"],
        #     done=transition["done"],
        #     value=transition["value"],
        #     iteration=self._policy_iteration
        # )
        self._curr_policy_rollout.append(transition)
        
        self._steps_collected += 1
        self._steps_collected_with_curr_policy += 1
        stop = transition['done']
        ret = None
        if stop:
            advantages, returns = self._compute_gae(self._curr_policy_rollout)
            batch = self._prepare_batch(advantages, returns)
            self._rollout_buffer.add_batch(batch)
            if (self._steps_collected_with_curr_policy >= self.rollout_steps):
                ret = self._perform_update()
                self._steps_collected_with_curr_policy = 0
                self._curr_policy_rollout = []
        
        return ret


    def _compute_gae(self, rollout) -> Tuple[np.ndarray, np.ndarray]:
        T = len(rollout)
        rewards = np.array([t["reward"] for t in rollout])
        values = np.array([t["value"] for t in rollout])
        dones = np.array([t["done"] for t in rollout])  # Get done flag for each timestep

        # Get the final value for bootstrap
        #print(rollout)
        next_obs = rollout[-1]["next_obs"]
        with torch.no_grad():
            obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            pos = obs_t[:, :3]
            vel = obs_t[:, 3:5]
            trailer = obs_t[:, 5:6]
            _, final_v = self.policy(pos, vel, trailer)
            final_v = float(final_v.squeeze(0).item())

        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)

        # ---------------- Problem 1.2: Compute GAE ----------------
        # Rember: TD approx for Q(s, a) = r + V(s_t+1), and monte carlo
        # approx Q(s, a) with full rollout.
        # Now, we sum the TD error from every timestep from now until the end of the 
        # rollout, and use lambda * gamma exponential to weight them. This is like
        # taking WEIGHTED AVERAGE OF ALL N STEP RETURNS. Higher lambda, more like
        # monte carlo, smaller lambda, more like TD
        ### BEGIN STUDENT SOLUTION - 1.2 ###
        advantages[-1] = rewards[-1] + final_v - values[-1]
        for t in range(T-2, -1, -1):
            # If ever at done, reset with just pure advantage at that point (reward - value)
            if (dones[t]):
                advantages[t] = rewards[t] - values[t]
            else:
                advantages[t] = rewards[t] + self.gamma * values[t+1] - values[t] + self.gamma * self.gae_lambda * advantages[t+1]

        returns = values + advantages
        # returns_mean = returns.mean()
        # returns_std = returns.std() + 1e-8
        # returns = (returns - returns_mean) / returns_std
        ### END STUDENT SOLUTION - 1.2 ###
        return advantages, returns

    def _perform_update(self) -> Dict[str, float]:
        all_stats = []
        batch = self._rollout_buffer.sample(self.rollout_steps//2, 
                                          filter={"iteration": [self._policy_iteration]})
        
        # Normalize advantages
        adv = batch["advantages"]
        batch["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)
        batch["advantages"] = torch.clamp(batch["advantages"], -10.0, 10.0)
        
        for _ in range(self.update_epochs):
            perm = torch.randperm(len(batch["obs"]))
            for i in range(0, len(batch["obs"]), self.minibatch_size):
                idx = perm[i:i+self.minibatch_size]
                minibatch = {k: v[idx] for k, v in batch.items()}
                loss, stats = self._ppo_loss(minibatch)
                all_stats.append(stats)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        self._policy_iteration += 1

        return {k: np.mean([s[k] for s in all_stats]) for k in all_stats[0].keys()}

    def _ppo_loss(self, batch):
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        old_log_probs = batch["log_probs"].to(self.device)
        advantages = batch["advantages"].to(self.device)
        returns = batch["returns"].to(self.device)
        old_values = batch["values"].to(self.device)

        # NEW: Split [6] into pos(3), vel(2), trailer(1)
        pos = obs[:, :3]
        vel = obs[:, 3:5]
        trailer = obs[:, 5:6]
        
        dist, values = self.policy(pos, vel, trailer)
        log_probs = dist.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)
        # print(max(ratio).item(), min(ratio).item(), ratio.mean().item())
        # print((abs(ratio - 1) > self.clip_coef).float().mean().item())
        policy_loss = -torch.min(ratio * advantages, 
                               torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * advantages).mean()
        value_clip = old_values + torch.clamp(values.squeeze(-1) - old_values, -self.clip_coef, self.clip_coef)
        #value_loss_unclipped = (values.squeeze(-1) - returns) ** 2
        #value_loss_clipped = (value_clip - returns) ** 2
        #value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
        value_loss = F.mse_loss(values.squeeze(-1), returns)
        entropy_loss = -dist.entropy().mean()
        brake_loss = torch.relu(-actions[:, 0]).mean()  # Penalize negative throttle (reverse)
        #print(policy_loss.item(), brake_loss.item(), self.vf_coef * value_loss.item(), self.ent_coef * entropy_loss.item())
        total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss #+ 0.0001 * brake_loss
        
        with torch.no_grad():
            approx_kl = (old_log_probs - log_probs).mean().item()
            clipfrac = (abs((ratio - 1.0)) > self.clip_coef).float().mean().item()
            
        stats = {
            "loss/total": float(total_loss.item()),
            "loss/policy": float(policy_loss.item()),
            "loss/value": float(value_loss.item()),
            "loss/entropy": float(entropy_loss.item()),
            "stats/kl": approx_kl,
            "stats/clipfrac": clipfrac,
        }
        return total_loss, stats

    def _prepare_batch(self, advantages, returns):
        """Collate the current rollout into a batch for the buffer"""
        obs = torch.stack([torch.as_tensor(t["obs"], dtype=torch.float32) for t in self._curr_policy_rollout])
        next_obs = torch.stack([torch.as_tensor(t["next_obs"], dtype=torch.float32) for t in self._curr_policy_rollout])
        actions = torch.stack([torch.as_tensor(t["action"], dtype=torch.float32) for t in self._curr_policy_rollout])
        log_probs = torch.tensor([t["log_prob"] for t in self._curr_policy_rollout], dtype=torch.float32)
        values = torch.tensor([t["value"] for t in self._curr_policy_rollout], dtype=torch.float32)
        rewards = torch.tensor([t["reward"] for t in self._curr_policy_rollout], dtype=torch.float32)

        return {
            "obs": obs.to(self.device),
            "next_obs": next_obs.to(self.device),
            "actions": actions.to(self.device),
            "log_probs": log_probs.to(self.device),
            "rewards": rewards.to(self.device),
            "values": values.to(self.device),
            "dones": torch.tensor([t["done"] for t in self._curr_policy_rollout], dtype=torch.float32, device=self.device),
            "advantages": torch.as_tensor(advantages, dtype=torch.float32, device=self.device),
            "returns": torch.as_tensor(returns, dtype=torch.float32, device=self.device),
            "iteration": torch.full((len(self._curr_policy_rollout),), self._policy_iteration, dtype=torch.int32, device=self.device)
        }

def warmup_value_head(ppo, env, vel_mean, vel_std, warmup_steps=15000):
    """Warm up value head with value-only updates"""
    print("Warming up value head...")
    obs_raw = env.reset()
    obs = process_observation(obs_raw, vel_mean, vel_std)
    
    ppo._rollout_buffer.reset()
    step = 0
    
    while step < warmup_steps:
        # Collect transitions (deterministic actions)
        action_dict = ppo.act(obs)
        env.apply_control(np.concatenate([action_dict["action"], np.ones(1)]))
        for _ in range(2):
            env.world.tick()
            
        next_obs_raw, reward, terminated, truncated = env.get_observation()
        next_obs = process_observation(next_obs_raw, vel_mean, vel_std)
        #print(obs.shape, next_obs.shape)
        # Store for value warmup (no policy gradients)
        ppo._rollout_buffer.add(
            obs=obs,
            next_obs=next_obs,
            action=torch.as_tensor(action_dict["action"], dtype=torch.float32),
            log_probs=action_dict["log_prob"],
            reward=float(reward),
            done=float(terminated or truncated),
            value=action_dict["value"],
            iteration=0  # Special iteration for warmup
        )
        
        obs_raw, obs = next_obs_raw, next_obs
        if terminated or truncated:
            obs_raw = env.reset()
            obs = process_observation(obs_raw, vel_mean, vel_std)
            #print(obs.shape)
        
        step += 1
        
        # Value-only update every 512 steps
        if step % 512 == 0 and ppo._rollout_buffer.size > 256:
            batch = ppo._rollout_buffer.sample(256, filter={"iteration": [0]})
            
            # Value-only loss
            batch_obs = batch["obs"].to(ppo.device)
            pos = batch_obs[:, :3]
            vel = batch_obs[:, 3:5]
            trailer = batch_obs[:, 5:6]
            _, values = ppo.policy(pos, vel, trailer)
            value_loss = F.mse_loss(values.squeeze(-1), batch["returns"].to(ppo.device))
            
            ppo.optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(ppo.policy.parameters(), ppo.max_grad_norm)
            ppo.optimizer.step()
            
            writer.add_scalar("warmup/value_loss", value_loss.item(), step)
    
    print("Value head warmup complete.")
    ppo._rollout_buffer.reset()

def train_singleagent(env, ppo, vel_mean, vel_std, max_training_steps=10_000_000, warmup=False):
    if warmup:
        warmup_value_head(ppo, env, vel_mean, vel_std)
    
    obs_raw = env.reset()
    obs = process_observation(obs_raw, vel_mean, vel_std)  # Process ONCE
    
    step, episode = 0, 0
    current_return = 0.0
    
    pbar = tqdm.tqdm(range(max_training_steps), dynamic_ncols=True)
    running_stats = {"reward": [], "value": [], "log_prob": [], "success": [], 
                     "throttle": [], "steer": [], "return": []}
    
    update_metrics = None
    while step < max_training_steps:
        action_dict = ppo.act(obs)
        action = action_dict["action"]
        
        env.apply_control(np.concatenate([action, np.ones(1)]))
        for _ in range(2):
            env.world.tick()
        
        next_obs_raw, reward, terminated, truncated = env.get_observation()
        next_obs = process_observation(next_obs_raw, vel_mean, vel_std)  # Process ONCE
        
        # CLEAN: Store PROCESSED observations only
        transition = {
            "obs": obs,           # [6] processed
            "next_obs": next_obs, # [6] processed  
            "action": action,
            "reward": float(reward),
            "done": float(terminated or truncated),
            "log_prob": action_dict["log_prob"],
            "value": action_dict["value"]
        }
        
        metrics = ppo.step(transition)
        if metrics:#only replace if not empty
            update_metrics = metrics
        
        running_stats["reward"].append(reward)
        running_stats["value"].append(action_dict["value"])
        running_stats["log_prob"].append(action_dict["log_prob"])
        running_stats["throttle"].append(action[0])
        running_stats["steer"].append(action[1])

        obs = next_obs  # Simple update
        current_return = current_return * ppo.gamma + reward
                
        if terminated or truncated:
            # ... episode logic unchanged ...
            obs_raw = env.reset()
            obs = process_observation(obs_raw, vel_mean, vel_std)
            running_stats["return"].append(current_return)
            if terminated:
                running_stats["success"].append(1.0)
            else:
                running_stats["success"].append(0.0)
            current_return = 0.0
            episode += 1

        
        step += 1
        #print(running_stats["success"])
        if step % 1000 == 0:
            window = min(1024, len(running_stats["reward"]))
            for key, values in running_stats.items():
                if key == "success" or key == "return":
                    writer.add_scalar(f"rollout/{key}", np.mean(values[-100:]), step) if len(values) > 0 else 0.0
                else:
                    writer.add_scalar(f"rollout/{key}", np.mean(values[-min(window, len(values)):]), step) if len(values) > 0 else 0.0
            
            #print(update_metrics)
            if update_metrics is not None:
                for key, value in update_metrics.items():
                    writer.add_scalar(f"ppo/{key}", value, step)
                
            lr = ppo.optimizer.param_groups[0]['lr']
            writer.add_scalar("learning_rate", lr, step)
            
            avg_reward = np.mean(running_stats["reward"][-window:])
            avg_success = np.mean(running_stats["success"][-100:]) #if len(running_stats["success"]) > 20 else 0.0
            if avg_success > 0.85 and len(running_stats["success"]) >= 100:
                torch.save({
                    'model_state': ppo.policy.state_dict(),
                    'optimizer_state': ppo.optimizer.state_dict(),
                    'step': step,
                }, f"checkpoints/ppo_singleagent_difficulty{env.difficulty:.3f}_success{avg_success:.2f}.pt")
                print(f"Increasing difficulty to {env.difficulty + 0.05:.3f} at step {step} (success {avg_success:.2f})")
                env.difficulty = min(env.difficulty + 0.05, 1.0)
                # reset log std to higher initial value
                # lower learning rate
                for param_group in ppo.optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'] * 0.95, ppo.base_lr * 0.05)
                running_stats["success"] = [] 
            pbar.set_description(f"Step {step} | Rwd {avg_reward:.2f} | Succ {avg_success:.2f}")
        
        pbar.update(1)
    
    writer.close()
    return avg_success

if __name__ == "__main__":
    stats_path = "data_no_image/.norm_stats.json"
    with open(stats_path, 'r') as fh:
        stats = json.load(fh)
    vel_mean = torch.tensor(stats.get('vel_mean', [0.0, 0.0]), dtype=torch.float32)
    vel_std = torch.tensor(stats.get('vel_std', [1.0, 1.0]), dtype=torch.float32)
    print(f"Normalization stats: vel_mean={vel_mean} vel_std={vel_std}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    world = client.load_world('Town10HD')

    max_steps = 1024
    checkpoint_filepath = None #"checkpoints/ppo_singleagent_difficulty0.00_success0.81.pt"
    
    policy = TruckNet(state_dim=6).to(device)  # NEW: state_dim=6
    if checkpoint_filepath and os.path.exists(checkpoint_filepath):
        ckpt = torch.load(checkpoint_filepath, map_location=device)
        state = ckpt.get('model_state', ckpt)
        policy.load_state_dict(state)
        print(f"Loaded policy from {checkpoint_filepath}")

    env = TruckEnv(max_steps=max_steps, world=world, phase=2, map_location=0, use_cameras=False)
    ppo = PPOSingleAgent(
        device=device,
        minibatch_size=128,
        rollout_steps=4096,
        vf_coef=0.2,
        ent_coef=0.005,
        clip_coef=0.2,
        lr=5e-5,
        policy=policy,
        vel_mean=vel_mean,
        vel_std=vel_std
    )

    success = train_singleagent(env, ppo, vel_mean, vel_std, max_training_steps=10_000_000, warmup=False)


    # ablation study of hyperparameters
    # start_lr = [1e-5, 5e-5, 1e-4]
    # rollout_steps = [1024, 2048, 4096]
    # vf_coef = [0.05, 0.1, 0.2, 0.3]
    # ent_coef = [0.0, 0.001, 0.01]
    # minibatch_size = [64, 128, 256]
    # clip_coeff = [0.1, 0.2, 0.3]

    # # Choose one set of hyperparameters to run
    # tested_configs = {}
    # highest_difficulty = 0.0
    # best_success = 0.0
    # for _ in range (10):
    #     policy = TruckNet(state_dim=6).to(device)  # NEW: state_dim=6
    #     env = TruckEnv(max_steps=max_steps, world=world, phase=2, map_location=0, use_cameras=False)

    #     lr = float(np.random.choice(start_lr))
    #     rs = int(np.random.choice(rollout_steps))
    #     vfc = float(np.random.choice(vf_coef))
    #     enc = float(np.random.choice(ent_coef))
    #     mbs = int(np.random.choice(minibatch_size))
    #     cc = float(np.random.choice(clip_coeff))
    #     config_key = f"lr{lr}_rs{rs}_vfc{vfc}_enc{enc}_mbs{mbs}_cc{cc}"
    #     if config_key in tested_configs:
    #         continue
    #     tested_configs[config_key] = True
    #     print(f"Testing config: {config_key}")
    #     ppo = PPOSingleAgent(
    #         device=device,
    #         minibatch_size=mbs,
    #         rollout_steps=rs,
    #         policy=policy,
    #         vel_mean=vel_mean,
    #         vel_std=vel_std,
    #         lr=lr,
    #         vf_coef=vfc,
    #         ent_coef=enc,
    #         clip_coef=cc
    #     )
    #     try:
    #         success = train_singleagent(env, ppo, vel_mean, vel_std, max_training_steps=500_000, warmup=False)
    #         if (env.difficulty >= highest_difficulty):
    #             if (env.difficulty == highest_difficulty and success < best_success):
    #                 continue
    #             print(f"Best Hyperparams so far: {config_key} with difficulty {env.difficulty}")
    #             best_success = success
    #             highest_difficulty = env.difficulty
    #     except KeyboardInterrupt:
    #         print("Training interrupted by user.")
    #         pass
    #     finally:
    #         env.destroy()