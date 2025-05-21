import torch
import torch.nn as nn
import numpy as np
import math 
class BezierCurve(nn.Module):
    def __init__(self, input_dim, num_ctrl_pts=5, action_dim=7, hidden_dim=256, act_horizon=1024):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_ctrl_pts = num_ctrl_pts
        self.action_dim = action_dim
        self.act_horizon = act_horizon
        self.output_dim = action_dim * num_ctrl_pts  # 关键修改点

        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim)
        )
        
        t = torch.linspace(0, 1, act_horizon)
        n = num_ctrl_pts - 1
        j = torch.arange(num_ctrl_pts).unsqueeze(1)
        comb = torch.tensor([math.comb(n, j_val) for j_val in j.squeeze().numpy()])
        self.register_buffer('coeff', comb * (1 - t)**(n - j) * t**j)  # (K, T)

    def forward(self, x):
        return self.predictor(x)

    def compute_loss(self, action, output):
        B, T, Da = action.shape
        ctrl_pts = output.view(B, Da, self.num_ctrl_pts)  
        output_upsample = torch.einsum('d k,b d k,b k t->b t d', 
                                      self.coeff, ctrl_pts)  
        return F.mse_loss(output_upsample, action)