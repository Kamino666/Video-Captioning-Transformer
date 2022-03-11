import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ClipSymmetricalLoss(nn.Module):
    def __init__(self, enable_tem=False, tem: float = None, device=torch.device("cuda")):
        super().__init__()
        self.device = device
        self.enable_tem = enable_tem
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        if tem is not None:
            self.temperature = torch.tensor([tem]).to(torch.float32).to(self.device)
        elif enable_tem is True:
            self.temperature = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)

    def forward(self, batch_video: Tensor, batch_text: Tensor):
        """
        :param batch_video: B, E
        :param batch_text:  B, E
        :return:
        """
        batch_video = batch_video / torch.linalg.norm(batch_video, dim=-1, keepdim=True)
        batch_text = batch_text / torch.linalg.norm(batch_text, dim=-1, keepdim=True)
        # [B E] @ [E B] = v[B B]t
        if self.temperature is not None:
            sim_matrix = torch.matmul(batch_video, batch_text.T) * torch.exp(self.temperature)
        else:
            sim_matrix = torch.matmul(batch_video, batch_text.T)

        target = torch.linspace(0, len(batch_video) - 1, len(batch_video), dtype=torch.long, device=self.device)
        sim_loss1 = self.cross_entropy(sim_matrix, target)
        sim_loss2 = self.cross_entropy(sim_matrix.T, target)
        return (sim_loss1 + sim_loss2) / 2


class ClipSymmetricalLoss_WithDualSoftmax(nn.Module):
    def __init__(self, enable_tem=False, tem: float = None, device=torch.device("cuda")):
        super().__init__()
        self.device = device
        self.enable_tem = enable_tem
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        # self.temperature = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        # self.temperature = torch.tensor([0.002])
        if tem is not None:
            self.temperature = torch.tensor([tem]).to(torch.float32).to(self.device)
        elif enable_tem is True:
            self.temperature = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)

    def forward(self, batch_video: Tensor, batch_text: Tensor):
        """
        :param batch_video: B, E
        :param batch_text:  B, E
        :return:
        """
        batch_video = batch_video / torch.linalg.norm(batch_video, dim=-1, keepdim=True)
        batch_text = batch_text / torch.linalg.norm(batch_text, dim=-1, keepdim=True)
        # [B E] @ [E B] = v[B B]t
        sim_matrix = torch.matmul(batch_video, batch_text.T)
        sim_matrix = sim_matrix * F.softmax(sim_matrix / self.temperature, dim=0) * len(sim_matrix)

        target = torch.linspace(0, len(batch_video) - 1, len(batch_video), dtype=torch.long, device=self.device)
        sim_loss1 = self.cross_entropy(sim_matrix, target)
        sim_loss2 = self.cross_entropy(sim_matrix.T, target)
        return (sim_loss1 + sim_loss2) / 2


class SCELoss(nn.Module):
    def __init__(self, alpha, beta, ignore_index, num_classes=10, device=torch.device('cuda')):
        super(SCELoss, self).__init__()
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        # rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        rce = -torch.sum(pred * torch.log(label_one_hot), dim=1)

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss
