#!/usr/bin/env python3
from __future__ import print_function

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
from model import TruckNet
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torch.utils.data import BatchSampler, RandomSampler

from tensordict import TensorDict
from torch.distributions import Normal, Independent, Bernoulli

import numpy as np
import torchvision.transforms as T
import torch.nn as nn
import json
from env import TruckEnv
import tqdm
import datetime
WORKSPACE_BOUNDS = np.array([-69, 157, -30, 200])

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = os.path.join(LOG_DIR, f"training_log_{timestamp}.txt")

tb_mod = importlib.import_module('torch.utils.tensorboard')
SummaryWriter = getattr(tb_mod, 'SummaryWriter')
_TB_BACKEND = 'torch'
writer = SummaryWriter(log_dir="runs/ppo_multiagent_no_image")

def log_to_file(msg):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

class PPOMultiAgent:
    def __init__(self, lr=5e-4, gamma=0.99, gae_lambda=0.95,
                 clip_coef=0.2, vf_coef=0.5, ent_coef=0.02, max_grad_norm=0.7,
                 update_epochs=10, minibatch_size=16, rollout_steps=500, 
                 policy=None, device="cuda", checkpoint_filepath=None, num_agents=1, use_images=True):
        
        self.device = torch.device(device)

        # instantiate model and move to device
        self.use_images = use_images
        if policy is None:
            self.policy = TruckNet(pretrained=False, use_images=use_images).to(self.device)
            if checkpoint_filepath is not None:
                ckpt = torch.load(checkpoint_filepath, map_location=self.device)
                state = ckpt.get('model_state', ckpt)
                self.policy.load_state_dict(state)
            self.policy.to(self.device)        
        else:
            self.policy = policy.to(self.device)

        self.num_agents = num_agents
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.rollout_steps = rollout_steps

        self.rollout = None
        self.ptr = 0

    def init_multiagent_rollout(self, obs_sample: TensorDict, num_agents: int):
        """Initialize multi-agent rollout buffer."""
        obs_td = TensorDict(
            {k: v.clone() for k, v in obs_sample.items()},
            batch_size=[],
            device=self.device,
        )

        self.num_agents = num_agents
        T = self.rollout_steps
        self.rollout = TensorDict(
            {
                "obs": obs_td.expand(T, num_agents),
                "action": torch.zeros((T, num_agents, 3), device=self.device),
                "reward": torch.zeros((T, num_agents), device=self.device),
                "done": torch.zeros((T, num_agents), dtype=torch.bool, device=self.device),
                "log_prob": torch.zeros((T, num_agents), device=self.device),
                "value": torch.zeros((T, num_agents), device=self.device),
                "mask": torch.ones((T, num_agents), device=self.device),  # <-- NEW
            },
            batch_size=[T, num_agents],
            device=self.device
        )
        self.ptr = 0

    def store_multiagent_transition(self, obs_list, actions_env, raw_cont_actions, rewards, dones, log_probs, values):
        """
        Store transitions in the rollout.
        obs_list: list of TensorDicts
        actions_env: list of actions sent to env (squashed cont + bern)
        raw_cont_actions: list of raw continuous actions (before squashing)
        rewards: list of rewards
        dones: list of done flags
        """
        if self.rollout is None:
            self.init_multiagent_rollout(obs_list[0], num_agents=len(obs_list))

        for agent_id, obs in enumerate(obs_list):
            self.rollout["obs"][self.ptr, agent_id] = TensorDict(
                {k: v.clone() for k, v in obs.items()},
                batch_size=[],
                device=self.device
            )

            # Store raw cont + bern in buffer
            bern_action = actions_env[agent_id][2:3]  # last 1
            action_to_store = torch.cat([raw_cont_actions[agent_id], bern_action], dim=0)
            self.rollout["action"][self.ptr, agent_id] = action_to_store

            self.rollout["reward"][self.ptr, agent_id] = rewards[agent_id]
            self.rollout["done"][self.ptr, agent_id] = dones[agent_id]
            self.rollout["mask"][self.ptr, agent_id] = 1.0 - dones[agent_id].float()

            self.rollout["log_prob"][self.ptr, agent_id] = log_probs[agent_id]
            self.rollout["value"][self.ptr, agent_id] = values[agent_id]

        self.ptr += 1


    def act(self, obs: TensorDict, deterministic=False):
        """Sample an action from the policy given obs (hybrid action)."""
        with torch.no_grad():
            batch_obs = {k: v.unsqueeze(0).to(self.device) for k, v in obs.items()}

            # Forward pass
            cont_logits, bern_logits, log_std, value = self.policy(
                batch_obs["pos"],
                batch_obs["vel"],
                batch_obs["accel"],
                batch_obs["trailer_angle"],
                batch_obs["reverse"],
                batch_obs["images"] if self.use_images else None
            )

            std = log_std.exp()  # [1,3]
            
            cont_dist = Independent(Normal(cont_logits, std), 1)

            # Sample continuous action (raw)
            if deterministic:
                raw_cont_action = cont_logits
            else:
                raw_cont_action = cont_dist.rsample()

            cont_logp = cont_dist.log_prob(raw_cont_action)

            # Squash actions only for environment
            a0 = torch.tanh(raw_cont_action[:, 0:1])
            a1 = 2.0 * torch.tanh(raw_cont_action[:, 1:2])
            cont_action_env = torch.cat([a0, a1], dim=1)

            # Bernoulli actions
            bern_dist = Bernoulli(logits=bern_logits)
            bern_action = bern_dist.sample()
            bern_logp = bern_dist.log_prob(bern_action).sum(-1)

            action_env = torch.cat([cont_action_env, bern_action], dim=1)
            log_prob = cont_logp + bern_logp

            writer.add_scalar("Action/cont_log_std_0", log_std[0], global_step=self.ptr)
            writer.add_scalar("Action/cont_log_std_1", log_std[1], global_step=self.ptr)
            writer.add_scalar("Log_prob/cont_log_prob", cont_logp, global_step=self.ptr)
            writer.add_scalar("Log_prob/bern_log_prob", bern_logp, global_step=self.ptr)
        return action_env.squeeze(0), log_prob.squeeze(0), value.squeeze(0), raw_cont_action.squeeze(0)
    
    def compute_gae_multiagent(self, last_values=None):
        """
        Compute GAE using masks to stop at episode boundaries.
        last_values: torch tensor [num_agents], critic values for the state after last rollout step
        """
        T, A = self.ptr, self.num_agents

        rewards = self.rollout["reward"][:T]
        values = self.rollout["value"][:T]
        masks = self.rollout["mask"][:T]

        advantages = torch.zeros_like(rewards)
        last_adv = torch.zeros(A, device=self.device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_values if last_values is not None else torch.zeros(A, device=self.device)
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * masks[t] - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * masks[t] * last_adv
            last_adv = advantages[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values
        self.rollout["advantage"] = advantages
        self.rollout["return"] = returns

    def update(self):
        T, A = self.ptr, self.num_agents
        flat_rollout = self.rollout[:T].reshape(T * A)

        sampler = BatchSampler(
            RandomSampler(range(T * A), replacement=False),
            batch_size=self.minibatch_size,
            drop_last=False
        )

        for _ in range(self.update_epochs):
            for batch_indices in sampler:
                batch = flat_rollout[batch_indices]

                batch_obs = {k: v.to(self.device) for k, v in batch["obs"].items()}
                batch_actions = batch["action"].to(self.device)
                batch_log_probs = batch["log_prob"].to(self.device)
                batch_advantages = batch["advantage"].to(self.device)
                batch_returns = batch["return"].to(self.device)

                # Forward pass
                cont_mean, bern_logits, log_std, values = self.policy(
                    batch_obs["pos"],
                    batch_obs["vel"],
                    batch_obs["accel"],
                    batch_obs["trailer_angle"],
                    batch_obs["reverse"],
                    batch_obs["images"] if self.use_images else None
                )

                std = log_std.exp()
                cont_dist = Independent(Normal(cont_mean, std), 1)
                cont_logp = cont_dist.log_prob(batch_actions[:, :2])


                # Bernoulli log-prob
                bern_dist = Bernoulli(logits=bern_logits)
                bern_logp = bern_dist.log_prob(batch_actions[:, 2:3]).sum(-1)

                log_probs = cont_logp + bern_logp

                # PPO loss
                ratio = (log_probs - batch_log_probs).exp()
                writer.add_scalar("PPO/ratio_mean", ratio.mean(), global_step=self.ptr)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((values.squeeze(-1) - batch_returns)**2).mean()
                entropy_loss = cont_dist.entropy().mean() + bern_dist.entropy().mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    self.policy.log_std.clamp_(min=-5.0, max=0.5)

    def evaluate(self, evaluate_steps, vel_mean, vel_std, accel_mean, accel_std, env, map_location=0):
        obs = env.reset()
        obs_td = frame_to_tensordict(obs, img_size=224, device=self.device, map_location=map_location,
                                    vel_mean=vel_mean, vel_std=vel_std,
                                    accel_mean=accel_mean, accel_std=accel_std, use_images=self.use_images)
        total_reward = 0.0
        running_success_rates = []
        for step in range(evaluate_steps):
            with torch.no_grad():
                action, _, _, _ = self.act(obs_td)

            next_obs, reward, terminated, truncated = env.step(action.cpu().numpy())
            total_reward += reward

            if terminated or truncated:
                running_success_rates.append(1.0 if terminated else 0.0)
                obs = env.reset()
            else:
                obs = next_obs

            obs_td = frame_to_tensordict(obs, img_size=224, device=self.device, map_location=map_location,
                                        vel_mean=vel_mean, vel_std=vel_std,
                                        accel_mean=accel_mean, accel_std=accel_std, use_images=self.use_images)
        avg_reward = total_reward / evaluate_steps
        avg_success_rate = np.mean(running_success_rates) if len(running_success_rates) > 0 else 0.0
        return avg_reward, avg_success_rate
        


def make_transform(img_size):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def frame_to_tensordict(obs, img_size, device='cpu', map_location=0, vel_mean=0, vel_std=1, accel_mean=0, accel_std=1, use_images=True):
    """
    Convert env observation to model inputs.

    obs is (image_list, pos_list, vel_list, accel_list, trailer_angle, reverse, goal_list)
    Returns tensors: imgs(1,num_cams,3,H,W), pos_rel(1,3), vel(1,2), accel(1,2), trailer(1,1), rev(1,1)
    """
    if use_images:
        images, pos_list, vel_list, accel_list, trailer_angle, reverse_flag, goal_list = obs
    else:
        pos_list, vel_list, accel_list, trailer_angle, reverse_flag, goal_list = obs
    
    if use_images:
        #print(trailer_angle, reverse_flag)
        transform = make_transform(img_size)
        imgs = []
        for i in range(len(images)):
            im = images[i]
            im_pil = T.functional.to_pil_image(im)
            imgs.append(transform(im_pil))
        imgs = torch.stack(imgs, dim=0)  # (1, num_cams, 3, H, W)

    # select indices used in dataset: pos [0,1,4], vel [0,1], accel [0,1], goal [0,1,4]
    def sel_list(arr, indices):
        a = np.array(arr, dtype=np.float32).ravel()
        out = [float(a[i]) if i < a.size else 0.0 for i in indices]
        return np.array(out, dtype=np.float32)

    pos_list[0] -= map_location * 11000 / 100  # adjust x pos based on map location
    goal_list[0] -= map_location * 11000 / 100  # adjust x pos based on map location

    pos = sel_list(pos_list, [0, 1, 4])
    vel = sel_list(vel_list, [0, 1])
    accel = sel_list(accel_list, [0, 1])
    goal = sel_list(goal_list, [0, 1, 4])

    pos_t = torch.from_numpy(pos)
    goal_t = torch.from_numpy(goal)

    workspace_bounds = torch.tensor(WORKSPACE_BOUNDS, dtype=torch.float32)
    pos_t[:2] = 2 * (pos_t[:2] - workspace_bounds[[0, 2]]) / (workspace_bounds[[1, 3]] - workspace_bounds[[0, 2]]) - 1
    pos_t[2] = pos_t[2] / 180.0

    goal_t[:2] = 2 * (goal_t[:2] - workspace_bounds[[0, 2]]) / (workspace_bounds[[1, 3]] - workspace_bounds[[0, 2]]) - 1
    goal_t[2] = goal_t[2] / 180.0

    pos_rel = goal_t - pos_t

    vel_t = (torch.from_numpy(vel) - vel_mean) / vel_std
    accel_t = (torch.from_numpy(accel) - accel_mean) / accel_std

    trailer_t = torch.tensor([float(trailer_angle)], dtype=torch.float32) / 180.0
    rev_t = torch.tensor([1.0 if reverse_flag else 0.0], dtype=torch.float32)

    if use_images:
        obs = TensorDict({
            "images": imgs.to(device),
            "pos": pos_rel.to(device),
            "vel": vel_t.to(device),
            "accel": accel_t.to(device),
            "trailer_angle": trailer_t.to(device),
            "reverse": rev_t.to(device),
        })
    else:
        obs = TensorDict({
            "pos": pos_rel.to(device),
            "vel": vel_t.to(device),
            "accel": accel_t.to(device),
            "trailer_angle": trailer_t.to(device),
            "reverse": rev_t.to(device),
        })

    return obs


def train_multiagent(envs, ppo, img_size, num_agents=4, vel_mean=0, vel_std=1, accel_mean=0, accel_std=1, max_training_steps=7_000_000, use_images=True):

    obs_list = [None] * num_agents
    for i in range(num_agents):
        obs = envs[i].reset()
        obs_list[i] = frame_to_tensordict(obs, img_size, device=ppo.device, map_location=i, vel_mean=vel_mean, vel_std=vel_std, accel_mean=accel_mean, accel_std=accel_std, use_images=use_images)

    step = 0
    episode = 0
    best_success_rate = -float('inf')
    evaluate = True
    
    pbar = tqdm.tqdm(range(max_training_steps), dynamic_ncols=True)

    running_rewards = []
    running_values = []
    running_log_probs = []
    running_throttle_values = []
    running_steer_values = []
    running_reverse_values = []
    running_success_rates = []

    for step in pbar:        
        actions = []
        raw_cont_actions = []
        values_this_step = []
        log_probs_this_step = []
        throttle_values_this_step = []
        # brake_values_this_step = []
        steer_values_this_step = []
        reverse_values_this_step = []

        for obs in obs_list:
            action, log_prob, value, raw_cont_action = ppo.act(obs)
            actions.append(action)
            raw_cont_actions.append(raw_cont_action)
            log_probs_this_step.append(log_prob)
            values_this_step.append(value.item())
            throttle_values_this_step.append(action[0].detach().cpu().item())
            steer_values_this_step.append(action[1].detach().cpu().item())
            reverse_values_this_step.append(action[2].detach().cpu().item())

        next_obs = []
        next_rewards = []
        next_terminated = []
        next_truncated = []

        for i in range(num_agents):
            envs[i].apply_control(actions[i].cpu().numpy())
        envs[0].world.tick()
        
        for i in range(num_agents):
            o, r, term, trunc = envs[i].get_observation()
            next_obs.append(frame_to_tensordict(o, img_size, device=ppo.device, map_location=i, vel_mean=vel_mean, vel_std=vel_std, accel_mean=accel_mean, accel_std=accel_std, use_images=use_images))
            next_rewards.append(r)
            next_terminated.append(term)
            next_truncated.append(trunc)
            if term:
                running_success_rates.append(1.0)
            elif trunc:
                running_success_rates.append(0.0)

        rewards = torch.tensor(next_rewards, device=ppo.device, dtype=torch.float32)
        dones = torch.tensor(next_terminated, device=ppo.device, dtype=torch.bool) | torch.tensor(next_truncated, device=ppo.device, dtype=torch.bool)

        ppo.store_multiagent_transition(
            obs_list,
            torch.stack(actions),
            raw_cont_actions,
            rewards,
            dones,
            log_probs_this_step,
            values_this_step
        )

        obs_list = next_obs

        if (step + 1) % 100000 == 0:
            evaluate = True
        
        running_rewards.extend(next_rewards)
        running_values.extend(values_this_step)
        running_log_probs.extend([lp.item() for lp in log_probs_this_step])
        running_throttle_values.extend(throttle_values_this_step)
        running_steer_values.extend(steer_values_this_step)
        running_reverse_values.extend(reverse_values_this_step)
        avg_reward = np.mean(running_rewards[-1000:])
        avg_value = np.mean(running_values[-1000:])
        avg_log_prob = np.mean(running_log_probs[-1000:])
        avg_throttle = np.mean(running_throttle_values[-1000:])
        avg_steer = np.mean(running_steer_values[-1000:])
        avg_reverse = np.mean(running_reverse_values[-1000:])
        avg_success_rate = np.mean(running_success_rates[-100:]) if len(running_success_rates) > 0 else 0.0

        if (step + 1) % 1000 == 0:
            writer.add_scalar("Control/AvgThrottle", avg_throttle, step)
            writer.add_scalar("Control/AvgSteer", avg_steer, step)
            writer.add_scalar("Control/AvgReverse", avg_reverse, step)
            writer.add_scalar("AvgReward", avg_reward, step)
            writer.add_scalar("AvgValue", avg_value, step)
            writer.add_scalar("AvgLogProb", avg_log_prob, step)
            writer.add_scalar("AvgSuccessRate", avg_success_rate, step)
            log_to_file(f"step={step}, avg_reward={avg_reward:.4f}, avg_value={avg_value:.4f}, avg_log_prob={avg_log_prob:.4f}")


        pbar.set_description(f"Step {step} | Avg Reward {avg_reward:.2f} | Avg Value {avg_value:.2f}")

        if ppo.ptr >= ppo.rollout_steps:
            ppo.compute_gae_multiagent(last_values=torch.tensor(values_this_step).to(ppo.device))
            ppo.update()
            ppo.rollout = None
            episode += 1

        reset_flag = False
        for i in range(num_agents):
            if dones[i] or reset_flag: # uf we perform an evaluation, reset all envs
                if evaluate:
                    evaluate = False
                    reset_flag = True
                    print("Evaluating...")
                    avg_reward, avg_success_rate = ppo.evaluate(evaluate_steps=6000, vel_mean=vel_mean, vel_std=vel_std, accel_mean=accel_mean, accel_std=accel_std, env = envs[i], map_location=i)
                    print(f"[EVAL] episode={episode} steps={step} avg_reward={avg_reward:.3f} avg_success_rate={avg_success_rate:.3f}")
                    log_to_file(f"[EVAL] episode={episode}, step={step}, avg_reward={avg_reward:.4f}, avg_success_rate={avg_success_rate:.4f}")

                    torch.save({'model_state': ppo.policy.state_dict()},
                        f"checkpoints/ppo_{episode}_no_image.pt")
                    if (avg_success_rate >= best_success_rate):
                        best_success_rate = avg_success_rate
                        torch.save({'model_state': ppo.policy.state_dict()},
                                f"checkpoints/ppo_multiagent_no_image_best.pt")
                        print(f"New best model saved with avg_success_rate={best_success_rate:.3f}")
                        log_to_file(f"[BEST] episode={episode}, step={step}, new_best_success_rate={best_success_rate:.4f}")
                if not reset_flag:
                    obs_list[i] = frame_to_tensordict(envs[i].reset(), img_size, device=ppo.device, map_location=i, vel_mean=vel_mean, vel_std=vel_std, accel_mean=accel_mean, accel_std=accel_std, use_images=use_images)
        if reset_flag:
            for i in range(num_agents):
                obs_list[i] = frame_to_tensordict(envs[i].reset(), img_size, device=ppo.device, map_location=i, vel_mean=vel_mean, vel_std=vel_std, accel_mean=accel_mean, accel_std=accel_std, use_images=use_images)

if __name__ == "__main__":
    stats_path = "data/.norm_stats.json"
    with open(stats_path, 'r') as fh:
        stats = json.load(fh)
    vel_mean = torch.tensor(stats.get('vel_mean', [0.0, 0.0]), dtype=torch.float32)
    vel_std = torch.tensor(stats.get('vel_std', [1.0, 1.0]), dtype=torch.float32)
    accel_mean = torch.tensor(stats.get('accel_mean', [0.0, 0.0]), dtype=torch.float32)
    accel_std = torch.tensor(stats.get('accel_std', [1.0, 1.0]), dtype=torch.float32)
    print(f"Normalization stats: vel_mean={vel_mean} vel_std={vel_std} accel_mean={accel_mean} accel_std={accel_std}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    world = client.load_world('Town10HD')

    max_steps = 2000
    num_agents = 4

    use_images = False

    checkpoint_filepath = "checkpoints/ppo_100_no_image.pt"
    policy = TruckNet(pretrained=False, use_images=use_images).to(device)
    if checkpoint_filepath is not None:
        ckpt = torch.load(checkpoint_filepath, map_location=device)
        state = ckpt.get('model_state', ckpt)
        policy.load_state_dict(state)
        with torch.no_grad():
            policy.log_std.fill_(-0.5)        
        print(f"Loaded policy from {checkpoint_filepath}")

    policy.to(device)        

    envs = [TruckEnv(max_steps=max_steps, world=world, phase=1, map_location=i, use_cameras=use_images) for i in range(num_agents)]
    ppo = PPOMultiAgent(
        device=device,
        num_agents=num_agents,
        use_images=use_images,
        minibatch_size=64,
        rollout_steps=1000, 
        policy=policy
    )

    train_multiagent(
        envs,
        ppo,
        img_size=224,
        num_agents=num_agents,
        vel_mean=vel_mean,
        vel_std=vel_std,
        accel_mean=accel_mean,
        accel_std=accel_std,
        use_images=use_images
    )