import torch
import numpy as np
import json
import random
import os


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    """This class is modified from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py"""

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

    def __call__(self, val_loss, model, do_save):

        val_loss = -val_loss

        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model, do_save)
        elif val_loss < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model, do_save)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, do_save):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if do_save is True:
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


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


class Config:
    def __init__(self, path: str):
        """
        Load json config file from disk.
        :param path: The path of config file
        """
        with open(path) as f:
            self.data = json.load(f)

    def display(self, l: int = 90):
        self.data: dict
        bold_line = "=" * l
        thin_list = "-" * l
        print(bold_line)
        print("{:^{}}".format("Config", l))
        print(bold_line)
        for mk, mv in self.data.items():
            print("{:^{}}".format(f"{mk}", l))
            print(thin_list)
            if type(mv) != dict:
                print(mv)
            else:
                for k, v in mv.items():
                    print("{:<20}| {}".format(k, v))
            print(bold_line)

    def check(self):
        model_cfg = self.data['model']
        if model_cfg['video_encoder'].get('type', 'mme') == 'simple':
            if self.data['train']['task'] != "caption":
                raise ValueError("Simple video encoder does NOT support 'cross' task")


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def configure_hardware(args):
    import torch.distributed as dist
    if args.cpu:
        args.device = torch.device('cpu')
        args.is_main_rank = True
        print("\033[1;33;40m Using CPU as backend \033[0m")
    elif args.gpu:
        args.device = torch.device('cuda')
        # args._multi_gpu = False
        args.is_main_rank = True
        print("\033[1;33;40m Using CUDA as backend \033[0m")
    elif args.multi_gpu:
        local_rank = int(os.environ["LOCAL_RANK"])
        args.local_rank = local_rank
        args.is_main_rank = True if local_rank == 0 else False
        # args.world_size = 4
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')  # 一般使用的后端为nccl
        args.device = torch.device("cuda", local_rank)
        if args.is_main_rank:
            print("\033[1;33;40m Using multiple CUDA as backend \033[0m")
    else:
        raise ValueError("No hardware configured")
    return args

