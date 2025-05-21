import torch
import torch.nn as nn
import numpy as np
import math 
import torch.nn.functional as F

class BezierCurve(nn.Module):
    def __init__(self, input_dim, num_ctrl_pts=5, se3_dim=2, hidden_dim=256, act_horizon=1024):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_ctrl_pts = num_ctrl_pts
        self.se3_dim = se3_dim
        self.act_horizon = act_horizon
        self.output_dim = se3_dim * num_ctrl_pts 

        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim)
        )

        t = torch.linspace(0, 1, act_horizon) 
        n = num_ctrl_pts - 1
        j = torch.arange(num_ctrl_pts).float()  
        comb = torch.tensor([math.comb(n, int(j_val)) for j_val in j])  
        self.register_buffer('coeff', comb.unsqueeze(1) * (1 - t).unsqueeze(0)**(n - j.unsqueeze(1)) * t.unsqueeze(0)**j.unsqueeze(1))

    def forward(self, x):
        return self.predictor(x)

    def compute_loss(self, action, output):
        B = action.shape[0]
        T = action.shape[1]
        ctrl_pts = output.view(B, 2, self.num_ctrl_pts)  
        output_upsample = torch.einsum('kt,bdk->btd', self.coeff, ctrl_pts)
        return F.mse_loss(output_upsample, action)
