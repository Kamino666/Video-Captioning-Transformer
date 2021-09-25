from mmcv import Config
from mmaction.models import build_model
from mmcv.runner import load_checkpoint
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from tqdm import tqdm
device = torch.device("cpu")

"""build model"""
config = './checkpoint/swin_tiny_patch244_window877_kinetics400_1k.py'
checkpoint = './checkpoint/swin_tiny_patch244_window877_kinetics400_1k.pth'

cfg = Config.fromfile(config)
model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, checkpoint, map_location='cpu')

"""load data"""
class mini_dataset(Dataset):
    def __init__(self, buffer_save_path):
        self.video2data = np.load(buffer_save_path)
        self.keys = list(self.video2data.keys())

    def __getitem__(self, item):
        data = self.video2data[self.keys[item]]
        data = torch.tensor(data).to(device)  # T, H, W, C
        data = rearrange(data, "t h w c -> t c h w")
        return data

    def __len__(self):
        return len(self.keys)
video_dataset = mini_dataset(r"./data/msrvtt-train-buffer.npz")
video_dataloader = DataLoader(video_dataset, batch_size=16)

"""extracting feats"""
# [batch_size, channel, temporal_dim, height, width]
feats = []
for video_x in tqdm(video_dataloader):
    # dummy_x = torch.rand(1, 3, 32, 224, 224)
    feat = model.extract_feat(video_x).mean_(dim=[3, 4])
    feats.append(feat)
feats = torch.stack(feats)
feats = rearrange(feats, 'n b t c -> (n b) t c').numpy()
feats_dict = dict(zip(video_dataset.keys, feats))
np.savez_compressed("./data/train_feats.npz", **feats_dict)

# SwinTransformer3D without cls_head
# feat = model.extract_feat(dummy_x)

# mean pooling
# feat = feat.mean(dim=[3, 4])  # [batch_size, hidden_dim, temporal_dim/2]


