import logging
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger_initialized = {}


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.

    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # handle duplicate logs to the console
    # Starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
    # to the root logger. As logger.propagate is True by default, this root
    # level handler causes logging messages from rank>0 processes to
    # unexpectedly show up on the console, creating much unwanted clutter.
    # To fix this issue, we set the root logger's StreamHandler, if any, to log
    # at the ERROR level.
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger


# class MaskCriterion(nn.Module):
#     """calculate the CrossEntropyLoss in mask=1 area"""
#
#     def __init__(self):
#         super(MaskCriterion, self).__init__()
#         self.loss_fn = nn.CrossEntropyLoss()
#
#     def forward(self, logits, target, mask):
#         """
#         logits: shape of (N, seq_len - 1, vocab_size)
#         target: shape of (N, seq_len)
#         mask: shape of (N, seq_len)
#         """
#         item_sum = logits.shape[0]*logits.shape[1]  # N * seq_len
#         target, mask = target[:, 1:], mask[:, 1:]
#         # loss [N*seq_len]
#         loss = self.loss_fn(logits.contiguous().view(item_sum, -1),
#                             target.contiguous().view(-1))
#         mask_loss = loss * mask.contiguous().view(-1)
#         output = torch.sum(mask_loss) / torch.sum(mask)
#         return output
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing: float = 0.1, reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.weight = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
            if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(loss / n, nll)


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


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


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
