#!/usr/bin/env python3
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

WORKSPACE_BOUNDS = np.array([-69, 157, -30, 200])


class PPOMultiAgent:
    def __init__(self, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_coef=0.2, vf_coef=0.5, ent_coef=0.02, max_grad_norm=0.5,
                 update_epochs=10, minibatch_size=16, rollout_steps=500, 
                 policy=None, device="cuda", checkpoint_filepath=None, num_agents=1):
        self.device = torch.device(device)

        # instantiate model and move to device
        if policy is None:
            self.policy = TruckNet(pretrained=False).to(self.device)
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

        '''
        Example of how to use buffer:
        transition = TensorDict({
            "state": torch.randn(4),
            "action": torch.tensor([1]),
            "reward": torch.tensor([1.0]),
            "next_state": torch.randn(4),
            "done": torch.tensor([0], dtype=torch.bool)
        }, batch_size=[])

        buffer.add(transition)
        batch = buffer.sample(32)
        '''
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
                "action": torch.zeros((T, num_agents, 5), device=self.device),
                "reward": torch.zeros((T, num_agents), device=self.device),
                "done": torch.zeros((T, num_agents), dtype=torch.bool, device=self.device),
                "log_prob": torch.zeros((T, num_agents), device=self.device),
                "value": torch.zeros((T, num_agents), device=self.device),
            },
            batch_size=[T, num_agents],
            device=self.device
        )
        self.ptr = 0

    def store_multiagent_transition(self, obs_list, actions, rewards, dones):
        """
        obs_list: list of 4 TensorDicts
        actions:  (4,5)
        rewards:  (4,)
        dones:    (4,)
        """
        if self.rollout is None:
            self.init_multiagent_rollout(obs_list[0], num_agents=len(obs_list))

        for agent_id, obs in enumerate(obs_list):
            self.rollout["obs"][self.ptr, agent_id] = TensorDict(
                {k: v.clone() for k, v in obs.items()},
                batch_size=[],
                device=self.device
            )
            self.rollout["action"][self.ptr, agent_id] = actions[agent_id]
            self.rollout["reward"][self.ptr, agent_id] = rewards[agent_id]
            self.rollout["done"][self.ptr, agent_id] = dones[agent_id]

            _, log_prob, value = self.act(obs)
            self.rollout["log_prob"][self.ptr, agent_id] = log_prob
            self.rollout["value"][self.ptr, agent_id] = value

        self.ptr += 1

    def act(self, obs: TensorDict):
        """Sample an action from policy given obs (hybrid action)."""
        with torch.no_grad():
            batch_obs = {k: v.unsqueeze(0).to(self.device) for k, v in obs.items()}

            mean, logvar, value = self.policy(
                batch_obs["images"],
                batch_obs["pos"],
                batch_obs["vel"],
                batch_obs["accel"],
                batch_obs["trailer_angle"],
                batch_obs["reverse"]
            )

            # Continuous actions
            std = (0.5 * logvar).exp()
            cont_dist = Independent(Normal(mean[:, :3], std[:, :3]), 1)
            cont_action = cont_dist.rsample()
            cont_logp = cont_dist.log_prob(cont_action)

            # Bernoulli actions
            bern_probs = mean[:, 3:5]
            bern_dist = Bernoulli(probs=bern_probs)
            bern_action = bern_dist.sample()
            bern_logp = bern_dist.log_prob(bern_action).sum(-1)

            # Full action
            action = torch.zeros_like(mean)
            action[:, :3] = cont_action
            action[:, 3:5] = bern_action

            log_prob = cont_logp + bern_logp

        return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)
    
    def compute_gae_multiagent(self):
        T, A = self.ptr, self.num_agents

        rewards = self.rollout["reward"][:T]
        values = self.rollout["value"][:T]
        dones = self.rollout["done"][:T]

        advantages = torch.zeros_like(rewards)
        last_adv = torch.zeros(A, device=self.device)

        for t in reversed(range(T)):
            mask = 1.0 - dones[t].float()
            next_value = values[t + 1] if t < T - 1 else torch.zeros(A, device=self.device)

            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * mask * last_adv
            last_adv = advantages[t]

        returns = advantages + values
        self.rollout["advantage"] = advantages
        self.rollout["return"] = returns

    def update(self):
        self.compute_gae_multiagent()

        T, A = self.ptr, self.num_agents

        # Flatten
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
                mean, logvar, values = self.policy(
                    batch_obs["images"],
                    batch_obs["pos"],
                    batch_obs["vel"],
                    batch_obs["accel"],
                    batch_obs["trailer_angle"],
                    batch_obs["reverse"]
                )

                # log probs
                std = (0.5 * logvar).exp()
                cont_dist = Independent(Normal(mean[:, :3], std[:, :3]), 1)
                cont_logp = cont_dist.log_prob(batch_actions[:, :3])

                bern_probs = mean[:, 3:5]
                bern_dist = Bernoulli(probs=bern_probs)
                bern_logp = bern_dist.log_prob(batch_actions[:, 3:5]).sum(-1)

                log_probs = cont_logp + bern_logp

                # PPO losses
                ratio = (log_probs - batch_log_probs).exp()
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

    def evaluate(self, evaluate_steps, vel_mean, vel_std, accel_mean, accel_std, env):
        obs = env.reset()
        obs_td = frame_to_tensordict(obs, img_size=224, device=self.device, map_location=0,
                                    vel_mean=vel_mean, vel_std=vel_std,
                                    accel_mean=accel_mean, accel_std=accel_std)
        total_reward = 0.0
        for step in range(evaluate_steps):
            with torch.no_grad():
                action, _, _ = self.act(obs_td)

            next_obs, reward, terminated, truncated = env.step(action.cpu().numpy())
            total_reward += reward

            if terminated or truncated:
                obs = env.reset()
            else:
                obs = next_obs

            obs_td = frame_to_tensordict(obs, img_size=224, device=self.device, map_location=0,
                                        vel_mean=vel_mean, vel_std=vel_std,
                                        accel_mean=accel_mean, accel_std=accel_std)
        avg_reward = total_reward / evaluate_steps
        return avg_reward
        


def make_transform(img_size):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def frame_to_tensordict(obs, img_size, device='cpu', map_location=0, vel_mean=0, vel_std=1, accel_mean=0, accel_std=1):
    """
    Convert env observation to model inputs.

    obs is (image_list, pos_list, vel_list, accel_list, trailer_angle, reverse, goal_list)
    Returns tensors: imgs(1,num_cams,3,H,W), pos_rel(1,3), vel(1,2), accel(1,2), trailer(1,1), rev(1,1)
    """
    images, pos_list, vel_list, accel_list, trailer_angle, reverse_flag, goal_list = obs
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
    #print(trailer_t.shape)
    rev_t = torch.tensor([1.0 if reverse_flag else 0.0], dtype=torch.float32)

    obs = TensorDict({
        "images": imgs.to(device),
        "pos": pos_rel.to(device),
        "vel": vel_t.to(device),
        "accel": accel_t.to(device),
        "trailer_angle": trailer_t.to(device),
        "reverse": rev_t.to(device),
    })

    return obs

def train_multiagent(envs, ppo, img_size, num_agents=4, vel_mean=0, vel_std=1, accel_mean=0, accel_std=1, max_training_steps=5_000_000):

    obs_list = [None] * num_agents
    for i in range(num_agents):
        obs = envs[i].reset()
        obs_list[i] = frame_to_tensordict(obs, img_size, device=ppo.device, map_location=i, vel_mean=vel_mean, vel_std=vel_std, accel_mean=accel_mean, accel_std=accel_std)

    step = 0
    episode = 0
    best_reward = -float('inf')
    evaluate = True
    
    pbar = tqdm.tqdm(range(max_training_steps), dynamic_ncols=True)

    running_rewards = []
    running_values = []

    for step in pbar:        
        actions = []
        values_this_step = []

        for obs in obs_list:
            action, _, value = ppo.act(obs)
            actions.append(action)
            values_this_step.append(value.item())


        # env.step expects list-of-4 actions
        next_obs = []
        next_rewards = []
        next_terminated = []
        next_truncated = []

        for i in range(num_agents):
            envs[i].apply_control(actions[i].cpu().numpy())
        envs[0].world.tick()
        
        for i in range(num_agents):
            o, r, term, trunc = envs[i].get_observation()
            next_obs.append(frame_to_tensordict(o, img_size, device=ppo.device, map_location=i, vel_mean=vel_mean, vel_std=vel_std, accel_mean=accel_mean, accel_std=accel_std))
            next_rewards.append(r)
            next_terminated.append(term)
            next_truncated.append(trunc)

        rewards = torch.tensor(next_rewards, device=ppo.device, dtype=torch.float32)
        dones = torch.tensor(next_terminated, device=ppo.device, dtype=torch.bool) | torch.tensor(next_truncated, device=ppo.device, dtype=torch.bool)

        # Store transition
        ppo.store_multiagent_transition(
            obs_list,
            torch.stack(actions),
            rewards,
            dones
        )

        # Move to next obs
        obs_list = next_obs

        if (step + 1) % 10000 == 0:
            # Perform evaluation
            evaluate = True
        
        running_rewards.extend(next_rewards)
        running_values.extend(values_this_step)
        avg_reward = np.mean(running_rewards[-100:])  # moving average over last 100 steps
        avg_value = np.mean(running_values[-100:])

        pbar.set_description(f"Step {step} | Avg Reward {avg_reward:.2f} | Avg Value {avg_value:.2f}")

        # Rollout complete → update PPO
        if ppo.ptr >= ppo.rollout_steps:
            ppo.update()
            ppo.rollout = None
            print(f"[UPDATE] episode={episode} steps={step}")
            episode += 1

        # Reset if any agent is done
        if any(dones):
            obs_list = [None] * num_agents
            if evaluate:
                evaluate = False
                print("Evaluating...")
                avg_reward = ppo.evaluate(evaluate_steps=5000, vel_mean=vel_mean, vel_std=vel_std, accel_mean=accel_mean, accel_std=accel_std, env = envs[np.random.randint(num_agents)])
                print(f"[EVAL] episode={episode} steps={step} avg_reward={avg_reward:.3f}")
                torch.save({'model_state': ppo.policy.state_dict()},
                    f"checkpoints/ppo_{episode}.pt")
                if (avg_reward > best_reward):
                    best_reward = avg_reward
                    torch.save({'model_state': ppo.policy.state_dict()},
                               f"checkpoints/ppo_multiagent_best.pt")
                    print(f"New best model saved with avg_reward={best_reward:.3f}")

            for i in range(num_agents):
                obs_list[i] = frame_to_tensordict(envs[i].reset(), img_size, device=ppo.device, map_location=i, vel_mean=vel_mean, vel_std=vel_std, accel_mean=accel_mean, accel_std=accel_std)
            episode += 1

if __name__ == "__main__":
        # load normalization stats if provided; otherwise use zero/one fallback
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

    envs = [TruckEnv(max_steps=max_steps, world=world, phase=1, map_location=i) for i in range(num_agents)]
    ppo = PPOMultiAgent(
        device=device,
        checkpoint_filepath="checkpoints/bc_epoch_013.pt",
        num_agents=num_agents
    )

    train_multiagent(
        envs,
        ppo,
        img_size=224,
        num_agents=num_agents,
        vel_mean=vel_mean,
        vel_std=vel_std,
        accel_mean=accel_mean,
        accel_std=accel_std
    )