import torch.nn as nn
import torch
import torchvision.models as models
import math
from torchvision.models import ResNet18_Weights


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
    def __init__(self, pretrained=True, num_cams=4, mlp_hidden=512, out_dim=3, pos_dim=3, vel_dim=2, accel_dim=2, trailer_angle_dim=1, rev_dim=1, state_embed=128, use_images=True):

        super().__init__()
        if use_images:
            if pretrained:
                    base = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            else:
                base = models.resnet18(weights=None)
            feat_dim = base.fc.in_features

            # remove final fc
            modules = list(base.children())[:-1]
            self.backbone = nn.Sequential(*modules)  # outputs (B, feat_dim, 1, 1)
            self.feat_dim = feat_dim
            self.num_cams = num_cams
            
        self.use_images = use_images

        self.pos_dim = pos_dim # position is relative (goal - pos)
        self.vel_dim = vel_dim
        self.accel_dim = accel_dim
        self.trailer_angle_dim = trailer_angle_dim
        self.rev_dim = rev_dim

        self.state_mlp = MLP(self.pos_dim + self.vel_dim + self.accel_dim + self.trailer_angle_dim + self.rev_dim, state_embed, state_embed, num_hidden_layers=4)
        total_feat_dim = (self.feat_dim * self.num_cams if use_images else 0) + state_embed

        self.shared_mlp = MLP(total_feat_dim, mlp_hidden, mlp_hidden, num_hidden_layers=3)
        self.mean_head = MLP(mlp_hidden, mlp_hidden, out_dim)
        self.value_head = MLP(mlp_hidden, mlp_hidden, 1)

        self.log_std = nn.Parameter(-1.6 * torch.ones(out_dim-1))

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, pos, vel, accel, trailer_angle, rev, rgb=None):

        state = torch.cat([pos, vel, accel, trailer_angle, rev], dim=1)
        state_emb = self.state_mlp(state)

        if self.use_images:
            # rgb: (B, num_cams, 3, H, W)
            B, Cc, ch, H, W = rgb.shape
            assert Cc == self.num_cams
            # reshape to (B * num_cams, 3, H, W)
            rgb = rgb.view(B * self.num_cams, ch, H, W)
            rgb_feats = self.backbone(rgb)  # (B*num_cams, feat_dim, 1, 1)
            rgb_feats = rgb_feats.view(B, self.num_cams * self.feat_dim)

            feats = torch.cat([rgb_feats, state_emb], dim=1)
        else:
            feats = state_emb

        shared_feats = self.shared_mlp(feats)

        mean_logits = self.mean_head(shared_feats)
        
        bern_logits = mean_logits[:, 2:3]

        cont_logits = mean_logits[:, :2]

        value = self.value_head(shared_feats)
        return cont_logits, bern_logits, self.log_std, value

