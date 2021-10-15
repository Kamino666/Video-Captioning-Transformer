import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from visualdl import LogWriter
import os
import time
from tqdm import tqdm
from transformers import AutoTokenizer

from model import VideoCaptionSwinTransformer
from dataloader import msrvtt_collate_fn, MSR_VTT_VideoDataset
from utils import MaskCriterion, EarlyStopping

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = LogWriter(logdir="./log")
local_rank = int(os.environ['LOCAL_RANK'])  # int 0/1/2/3

# 新增：DDP backend初始化
dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

# 构造模型
device = torch.device("cuda", local_rank)


class Opt:
    """train config"""
    batch_size = 8
    lr = 0.001
    weight_decay = 0
    learning_rate_patience = 20
    early_stopping_patience = 30
    MAX_EPOCHS = 200
    save_freq = -1

    """model config"""
    model_cfg = dict(
        patch_size=(2, 4, 4),
        drop_path_rate=0.1,
        patch_norm=True,
        window_size=(8, 7, 7),
        depths=(2, 2, 6, 2),
        embed_dim=96,
        checkpoint_pth=r"./checkpoint/swin_tiny_patch244_window877_kinetics400_1k.pth",
        pretrained2d=False,
        frozen_stages=-1,
        # newly added
        decoder_head=3,
        decoder_layers=2,
        device=device,
        tokenizer_type="bert-base-uncased",
        bert="bert-base-uncased",  # or None
    )

    """save config"""
    save_path = r"../checkpoint"
    training_token = "video_swin_patch{}_window{}_embed{}_depth{}_dhead{}_dlayer{}".format(
        "".join([str(i) for i in model_cfg["patch_size"]]),
        "".join([str(i) for i in model_cfg["window_size"]]),
        model_cfg["embed_dim"],
        "".join([str(i) for i in model_cfg["depths"]]),
        model_cfg["decoder_head"],
        model_cfg["decoder_layers"],
    )
    start_time = time.strftime('%y_%m_%d_%H_%M_%S-', time.localtime())


def train():
    opt = Opt()
    trainset = MSR_VTT_VideoDataset(r"./data/msrvtt-train-buffer.npz",
                                    r"./data/MSRVTT-annotations/train_val_videodatainfo.json")
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    train_loader = DataLoader(trainset, collate_fn=msrvtt_collate_fn, batch_size=opt.batch_size, sampler=train_sampler)
    valset = MSR_VTT_VideoDataset(r"./data/msrvtt-validate-buffer.npz",
                                  r"./data/MSRVTT-annotations/train_val_videodatainfo.json", mode="validate")
    val_sampler = torch.utils.data.distributed.DistributedSampler(valset)
    val_loader = DataLoader(valset, collate_fn=msrvtt_collate_fn, batch_size=opt.batch_size, sampler=val_sampler)

    vcst_model = VideoCaptionSwinTransformer(**opt.model_cfg)
    vcst_model = DDP(vcst_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    tokenizer = AutoTokenizer.from_pretrained(opt.model_cfg["tokenizer_type"])

    optimizer = optim.Adam(
        vcst_model.parameters(),
        lr=opt.lr,
        weight_decay=opt.weight_decay
    )
    # dynamic learning rate
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, verbose=True, patience=opt.learning_rate_patience
    )
    early_stopping = EarlyStopping(patience=opt.early_stopping_patience,
                                   verbose=True,
                                   path=os.path.join(opt.save_path,
                                                     opt.training_token + opt.start_time + 'earlystop.pth'))
    criterion = MaskCriterion()

    # Start Training
    for epoch in range(opt.MAX_EPOCHS):
        train_running_loss = train_epoch(vcst_model, train_loader, optimizer, epoch, criterion, tokenizer)
        valid_running_loss = val_epoch(vcst_model, val_loader, epoch, criterion, tokenizer)
        print("train loss:{} valid loss: {}".format(train_running_loss, valid_running_loss))

        writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], step=epoch)
        lr_scheduler.step(valid_running_loss)

        # early stopping
        early_stopping(valid_running_loss, vcst_model.module)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # save checkpoint
        if dist.get_rank() == 0 and opt.save_freq != -1 and epoch % opt.save_freq == 0:
            print('epoch:{}, saving checkpoint'.format(epoch))
            torch.save(vcst_model.module,
                       os.path.join(opt.save_path, opt.training_token + opt.start_time + str(epoch) + 'final.pth'))
    # save model
    if dist.get_rank() == 0:
        torch.save(vcst_model.module,
                   os.path.join(opt.save_path, opt.training_token + opt.start_time + 'final.pth'))


def train_epoch(model, train_loader, optimizer, epoch, criterion, tokenizer):
    """train one epoch"""
    train_loader.sampler.set_epoch(epoch)
    train_running_loss = 0.0
    loss_count = 0
    for index, (frames, caption) in enumerate(
            tqdm(train_loader, desc="epoch:{}".format(epoch))):
        optimizer.zero_grad()
        model.train()

        tokenized_cap = tokenizer(caption, padding=True, return_tensors="pt")
        tokenized_cap = {k: v.to(device) for k, v in tokenized_cap.items()}
        probs = model(frames, tokenized_cap)[:, :-1, :]
        loss = criterion(probs, tokenized_cap["input_ids"], tokenized_cap["attention_mask"])

        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()
        loss_count += 1

    train_running_loss /= loss_count
    writer.add_scalar('train_loss', train_running_loss, step=epoch)
    return train_running_loss


def val_epoch(model, val_loader, epoch, criterion, tokenizer):
    val_loader.sampler.set_epoch(epoch)
    valid_running_loss = 0.0
    loss_count = 0
    for index, (frames, caption) in enumerate(val_loader):
        model.eval()

        with torch.no_grad():
            tokenized_cap = tokenizer(caption, padding=True, return_tensors="pt")
            tokenized_cap = {k: v.to(device) for k, v in tokenized_cap.items()}
            probs = model(frames, tokenized_cap)[:, :-1, :]
            loss = criterion(probs, tokenized_cap["input_ids"], tokenized_cap["attention_mask"])

        valid_running_loss += loss.item()
        loss_count += 1

    valid_running_loss /= loss_count
    writer.add_scalar('valid_loss', valid_running_loss, step=epoch)
    return valid_running_loss


if __name__ == "__main__":
    train()
