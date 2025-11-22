#!/usr/bin/env python3
import torch
from model import TruckNet
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from tensordict import TensorDict
from torch.distributions import Normal, Independent, Bernoulli

class PPOAgent:
    def __init__(self, env_info, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_coef=0.2, vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5,
                 update_epochs=10, minibatch_size=64, rollout_steps=4096, checkpoint_filepath="checkpoints/best.pt", device="cuda"):
        self.device = torch.device(device)

        # instantiate model and move to device
        self.policy = TruckNet().to(self.device)

        ckpt = torch.load(checkpoint_filepath, map_location=self.device)
        state = ckpt.get('model_state', ckpt)
        self.policy.load_state_dict(state)
        self.policy.to(self.device)

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

        self.buffer = ReplayBuffer(storage=LazyTensorStorage(10000), sampler=SamplerWithoutReplacement()) # TODO: gonna need to really look at this
        self.current_rollout_buffer = ReplayBuffer(storage=LazyTensorStorage(10000), sampler=SamplerWithoutReplacement())
        self.steps_collected_with_curr_policy = 0
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

    def _init_rollout(self, obs_sample: TensorDict):
        """Initialize a rollout TensorDict with proper shapes for all observation components."""
        # Use clone for safety
        obs_td = TensorDict({k: v.clone() for k, v in obs_sample.items()}, batch_size=[], device=self.device)

        self.rollout = TensorDict(
            {
                "obs": obs_td,
                "action": torch.zeros((self.rollout_steps, 5), device=self.device),
                "reward": torch.zeros(self.rollout_steps, device=self.device),
                "done": torch.zeros(self.rollout_steps, dtype=torch.bool, device=self.device),
                "log_prob": torch.zeros(self.rollout_steps, device=self.device),
                "value": torch.zeros(self.rollout_steps, device=self.device)
            },
            batch_size=[self.rollout_steps],
            device=self.device
        )
        self.ptr = 0

    def store_transition(self, obs: TensorDict, action, reward, done):
        if self.rollout is None:
            self._init_rollout(obs)   # pass full observation

        # store full obs at this timestep
        self.rollout["obs"][self.ptr] = TensorDict({k: v.clone() for k, v in obs.items()}, batch_size=[], device=self.device)
        self.rollout["action"][self.ptr] = action
        self.rollout["reward"][self.ptr] = reward
        self.rollout["done"][self.ptr] = done

        _, log_prob, value = self.act(obs)
        self.rollout["log_prob"][self.ptr] = log_prob
        self.rollout["value"][self.ptr] = value
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
    
    def compute_gae(self):
        rewards = self.rollout["reward"][:self.ptr]
        values = self.rollout["value"][:self.ptr]
        dones = self.rollout["done"][:self.ptr]

        advantages = torch.zeros_like(rewards, device=self.device)
        last_adv = 0.0

        for t in reversed(range(self.ptr)):
            mask = 1.0 - dones[t].float()  # 0 if done, 1 otherwise
            # delta_t = r_t + γ * V(s_{t+1}) * mask - V(s_t)
            # V(s_{t+1}) is zero for terminal states
            next_value = 0.0 if t == self.ptr - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * mask * last_adv
            last_adv = advantages[t]

        returns = advantages + values
        self.rollout["advantage"] = advantages
        self.rollout["return"] = returns

    # TODO: add update loop


# TODO: add general training loop, need to normalize each input ahead of time, and then save as a tensordict
    


    



        