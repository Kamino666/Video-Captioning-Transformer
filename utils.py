import logging
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, ignore_index, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    """This class is from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py"""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class Meter:
    def __init__(self, mode="avg"):
        assert mode in ["avg", "max"]
        self.mode = mode
        self.count = 0
        self.sum = 0

    def add(self, x):
        if self.mode == "avg":
            self.sum += x
            self.count += 1
        elif self.mode == "max":
            self.sum = x if x > self.sum else self.sum

    def get(self):
        if self.mode == "avg":
            return self.sum / self.count
        elif self.mode == "max":
            return self.sum

    def pop(self):
        rslt = self.get()
        self.count = 0
        self.sum = 0
        return rslt


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def show_input_shape(**kwargs):
    print("\n***************************************")
    for name, arg in kwargs.items():
        if type(arg) is torch.Tensor:
            print(f"{name}: {arg.shape}")
        elif type(arg) is dict:
            print(f"{name}: ", end="")
            for k, v in arg.items():
                print(f"{k}:{v.shape}", end="  ")
            print("")
    print("***************************************\n")


def build_collate_fn(pad_id: int, include_id: bool):
    def func1(data):
        batch_size = len(data)
        # video id
        id_data = [i[2] for i in data]

        # video feature
        feat_dim = data[0][0].shape[1]
        feat_data = [i[0] for i in data]
        feat_len = [len(i) for i in feat_data]
        max_len = max(feat_len)
        feat_ts = torch.zeros([batch_size, max_len, feat_dim], dtype=torch.float)
        feat_mask_ts = torch.ones([batch_size, max_len], dtype=torch.long)
        for i in range(batch_size):
            feat_ts[i, :feat_len[i]] = feat_data[i]
            feat_mask_ts[i, :feat_len[i]] = 0
        feat_mask_ts = (feat_mask_ts == 1)

        # text
        text_data = [i[1] for i in data]
        text_len = [len(i) for i in text_data]
        max_len = max(text_len)
        text_ts = torch.ones([batch_size, max_len], dtype=torch.long) * pad_id
        for i in range(batch_size):
            text_ts[i, :text_len[i]] = text_data[i]
        text_mask_ts = (text_ts == pad_id)
        return feat_ts, text_ts, feat_mask_ts, text_mask_ts, id_data

    def func2(data):
        """
        :param data:
        :return tuple(N T E, N T):
        """
        batch_size = len(data)

        # video feature
        feat_dim = data[0][0].shape[1]
        feat_data = [i[0] for i in data]
        feat_len = [len(i) for i in feat_data]
        max_len = max(feat_len)
        feat_ts = torch.zeros([batch_size, max_len, feat_dim], dtype=torch.float)
        feat_mask_ts = torch.ones([batch_size, max_len], dtype=torch.long)
        for i in range(batch_size):
            feat_ts[i, :feat_len[i]] = feat_data[i]
            feat_mask_ts[i, :feat_len[i]] = 0
        feat_mask_ts = (feat_mask_ts == 1)

        # text
        text_data = [i[1] for i in data]
        text_len = [len(i) for i in text_data]
        max_len = max(text_len)
        text_ts = torch.ones([batch_size, max_len], dtype=torch.long) * pad_id
        for i in range(batch_size):
            text_ts[i, :text_len[i]] = text_data[i]
        text_mask_ts = (text_ts == pad_id)
        return feat_ts, text_ts, feat_mask_ts, text_mask_ts

    if include_id is True:
        return func1
    else:
        return func2
