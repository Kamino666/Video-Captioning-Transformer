import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .loss import ClipSymmetricalLoss, ClipSymmetricalLoss_WithDualSoftmax
from typing import Optional, Tuple

"""
进行 joint embedding 多任务学习的类
"""


class Matching(nn.Module):
    def __init__(self, vt_shape: Tuple[int, int], enable_tem=False, loss="CSL", loss_tem=None,
                 device=torch.device('cuda')):
        super().__init__()
        self.device = device
        self.vt_shape = vt_shape
        self.loss = loss
        self.v_proj = nn.Linear(vt_shape[0], vt_shape[1]) if vt_shape[0] != vt_shape[1] else None
        if loss == "CSL":
            self.loss_fn = ClipSymmetricalLoss(enable_tem, tem=loss_tem, device=device)
        elif loss == "CSL_WDS":
            self.loss_fn = ClipSymmetricalLoss_WithDualSoftmax(enable_tem, tem=loss_tem, device=device)

    def forward(self, text_feat: Tensor, vid_feat: Tensor):
        if self.v_proj is not None:
            vid_feat = self.v_proj(vid_feat)
        return self.loss_fn(text_feat, vid_feat)
