from mmcv import Config
from mmaction.models import build_model
from mmcv.runner import load_checkpoint
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from tqdm import tqdm
import torchvision
device = torch.device("cuda")

"""build model"""
config = './checkpoint/swin_tiny_patch244_window877_kinetics400_1k.py'
checkpoint = './checkpoint/swin_tiny_patch244_window877_kinetics400_1k.pth'

cfg = Config.fromfile(config)
model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, checkpoint, map_location='cuda')

"""load data"""
class mini_dataset(Dataset):
    def __init__(self, buffer_save_path):
        self.video2data = np.load(buffer_save_path)
        self.keys = list(self.video2data.keys())
        self.tf = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
        ])

    def __getitem__(self, item):
        data = self.video2data[self.keys[item]]
        frames = []
        for frame in data:
            frames.append(self.tf(frame))
        frames = torch.stack(frames).to(dtype=torch.float)
        frames = rearrange(frames, "t c h w -> c t h w")
        return frames

    def __len__(self):
        return len(self.keys)
video_dataset = mini_dataset(r"./data/msrvtt-validate-buffer.npz")
video_dataloader = DataLoader(video_dataset, batch_size=8)

# """extracting feats v1"""
# # [batch_size, channel, temporal_dim, height, width]
# feats = []
# for video_x in tqdm(video_dataloader):
#     # dummy_x = torch.rand(1, 3, 32, 224, 224)
#     feat = model.extract_feat(video_x)
#     feat = torch.mean(feat, dim=[3, 4])
#     feats.append(feat)
# feats = torch.stack(feats)
# feats = rearrange(feats, 'n b t c -> (n b) t c').numpy()
# feats_dict = dict(zip(video_dataset.keys, feats))
# np.savez_compressed("./data/train_feats.npz", **feats_dict)

"""extracting feats v2"""
# [batch_size, channel, temporal_dim, height, width]
for i, video_x in enumerate(tqdm(video_dataloader)):
    # dummy_x = torch.rand(1, 3, 32, 224, 224)
    with torch.no_grad():
        feats = model.extract_feat(video_x)
        feats = torch.mean(feats, dim=[3, 4])
        for j, feat in enumerate(feats):
            vid = video_dataset.keys[i*8+j]
            np.save("./data/msrvtt-validate-feats/{}.npy".format(vid), feat.numpy())




