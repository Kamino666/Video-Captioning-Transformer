import mmcv
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from tqdm import tqdm
import torchvision
import pretrainedmodels
import pathlib as plb
import multiprocessing
device = torch.device("cuda")


def extract_frames(video_path, mode="fix", value=40, frames_size=224):
    video = mmcv.VideoReader(str(video_path))
    if mode == "fix":
        samples_ix = np.linspace(0, video.frame_cnt - 1, value).astype(int)
    elif mode == "fps":
        time_length = video.frame_cnt / video.fps
        frame_num = int(time_length * value)
        samples_ix = np.linspace(0, video.frame_cnt - 1, frame_num).astype(int)
    else:
        raise ValueError("模式错误，只能从fix和fps二选一")
    frames = map(lambda x: video.get_frame(x), samples_ix)
    frames = list(map(lambda x: mmcv.imresize(x, (frames_size, frames_size)), frames))
    assert frames[0].shape[0] == frames[0].shape[1] == frames_size
    return frames

"""load data"""
class mini_dataset(Dataset):
    def __init__(self, video_dir, mode, value, frames_size=224, std=None, mean=None):
        video_dir = plb.Path(video_dir)
        assert video_dir.is_dir()
        self.video_paths = list(video_dir.glob("*.mp4"))
        
        std = std if std is not None else [0.229, 0.224, 0.225]
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ])
        
        self.mode = mode
        self.value = value
        self.frames_size = frames_size

    def __getitem__(self, item):
        frames = extract_frames(self.video_paths[item], self.mode, self.value, self.frames_size)
        trans_frames = []
        for frame in frames:
            trans_frames.append(
                self.transform(frame)
            )
        trans_frames = torch.stack(trans_frames)
        return trans_frames, trans_frames.shape[0], self.video_paths[item].stem

    def __len__(self):
        return len(self.video_paths)
    
def collate_fn(data):
    trans_frames, lengths, vids = zip(*data)
    trans_frames = torch.cat(trans_frames, 0)  # B*T H W C
    return trans_frames, lengths, vids
video_dataset = mini_dataset(r"./data/MSRVTT_trainval", "fps", 3)
video_dataloader = DataLoader(video_dataset, batch_size=32, collate_fn=collate_fn)

"""load model from pretrainedmodels"""
model = pretrainedmodels.resnet152(pretrained='imagenet')
model.last_linear = torch.nn.Identity()
model = model.to(device)
model.eval()

"""extracting feats v3"""
def feat_dumper(feats, lengths, vids):
    ptr = 0
    for length, vid in zip(lengths, vids):
        feat = feats[ptr:ptr+length]
        np.save(f"./data/msrvtt_resnet152_fps3_feats/{vid}.npy", feat.numpy())
        ptr += length
        
# [BT, H, W, C]
pool = multiprocessing.Pool(processes = 3)
results = []
for i, (video_x, lengths, vids) in enumerate(tqdm(video_dataloader)):
    video_x = video_x.to(device)
    with torch.no_grad():
        feats = model(video_x).cpu()  # B*T dim
        feat_dumper(feats, lengths, vids)
        results.append(
            pool.apply_async(feat_dumper, (feats, lengths, vids))
        )
        
for res in tqdm(results):
    res.get()
        




