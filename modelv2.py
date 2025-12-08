import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TruckNet(nn.Module):
    """
    PPO-friendly state-only network for truck control.
    Returns continuous action logits (raw), log_std, and value.
    """
    def __init__(
        self,
        state_dim=3+2+2+1+1,  # pos + vel + accel + trailer_angle + rev
        hidden_dim=512,
        policy_hidden_dim=512,
        out_dim=3  # 2 continuous + 1 bernoulli (can ignore)
    ):
        super().__init__()

        # Shared state embedding
        self.state_mlp = MLP(state_dim, hidden_dim, hidden_dim, num_hidden=4)

        # Policy trunk
        self.policy_mlp = MLP(hidden_dim, policy_hidden_dim, policy_hidden_dim, num_hidden=3)
        self.mean_head = nn.Linear(policy_hidden_dim, out_dim)

        # Value trunk
        self.value_mlp = MLP(hidden_dim, policy_hidden_dim, policy_hidden_dim, num_hidden=3)
        self.value_head = nn.Linear(policy_hidden_dim, 1)

        # Learnable log_std per action (continuous only)
        self.log_std = nn.Parameter(-1.0 * torch.ones(out_dim - 1))  # ignore bernoulli

    def forward(self, pos, vel, accel, trailer_angle, rev):
        # Concatenate all state inputs
        state = torch.cat([pos, vel, accel, trailer_angle, rev], dim=1)
        state_emb = self.state_mlp(state)

        # Policy
        policy_feat = self.policy_mlp(state_emb)
        mean_logits = self.mean_head(policy_feat)
        cont_logits = mean_logits[:, :2]  # continuous actions
        bern_logits = mean_logits[:, 2:3]  # can ignore / override

        # Value
        value_feat = self.value_mlp(state_emb)
        value = self.value_head(value_feat)

        return cont_logits, bern_logits, self.log_std, value
