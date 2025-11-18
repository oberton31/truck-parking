import torch.nn as nn
import torch
import torchvision.models as models

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]

        for _ in range(num_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class TruckNet(nn.Module):
    def __init__(self, pretrained=True, num_cams=4, mlp_hidden=512, out_dim=5, goal_dim=3, pos_dim=3, vel_dim=2, goal_embed=128, posvel_embed=128):

        super().__init__()
        base = models.resnet18(pretrained=pretrained)
        feat_dim = base.fc.in_features

        # remove final fc
        modules = list(base.children())[:-1]
        self.backbone = nn.Sequential(*modules)  # outputs (B, feat_dim, 1, 1)
        self.feat_dim = feat_dim
        self.num_cams = num_cams

        self.goal_dim = goal_dim
        self.pos_dim = pos_dim
        self.vel_dim = vel_dim

        
        self.goal_mlp = MLP(self.goal_dim, goal_embed, goal_embed)
        self.state_mlp = MLP(self.pos_dim + self.vel_dim, posvel_embed, posvel_embed)
        self.shared_mlp = MLP(self.num_cams * self.feat_dim + goal_embed + posvel_embed, mlp_hidden, mlp_hidden, num_hidden_layers=3)
        self.mean_head = MLP(mlp_hidden, mlp_hidden, out_dim)
        self.value_head = MLP(mlp_hidden, mlp_hidden, 1)

        self.var_head = MLP(mlp_hidden, mlp_hidden, out_dim) #logvar
        try:
            self.var_head.net[-1].bias.data.fill_(-3.0)
        except Exception:
            pass

        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb, pos, vel, goal):

        # rgb: (B, num_cams, 3, H, W)
        B, Cc, ch, H, W = rgb.shape
        assert Cc == self.num_cams
        # reshape to (B * num_cams, 3, H, W)
        rgb = rgb.view(B * self.num_cams, ch, H, W)
        rgb_feats = self.backbone(rgb)  # (B*num_cams, feat_dim, 1, 1)
        rgb_feats = rgb_feats.view(B, self.num_cams * self.feat_dim)

        state = torch.cat([pos, vel], dim=1)
        state_emb = self.state_mlp(state)
        goal_emb = self.goal_mlp(goal)

        feats = torch.cat([rgb_feats, state_emb, goal_emb], dim=1)

        shared_feats = self.shared_mlp(feats)

        mean_logits = self.mean_head(shared_feats)
        mean = self.sigmoid(mean_logits)

        # predict per-sample log-variance from shared features
        logvar = self.var_head(shared_feats)
        # clamp log-variance to a reasonable range to avoid numerical issues
        min_val, max_val = -10.0, 2.0
        logvar = 0.5 * (max_val + min_val) + 0.5 * (max_val - min_val) * torch.tanh(logvar)

        value = self.value_head(shared_feats)
        return mean, logvar, value