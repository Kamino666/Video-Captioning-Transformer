import mmcv
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torchvision
import torch

import json
import pathlib as plb
from tqdm import tqdm
import logging
import random
import os
from einops import rearrange

logger = logging.getLogger("main")


class MSR_VTT_VideoDataset(Dataset):
    def __init__(self, video_dir: str, annotation_file: str,
                 frames_num=40, frames_size=224,
                 mode="train", buffer_save_path=r"./data/buffer.npz",
                 random_seed=10503):
        """
        load MSR_VTT dataset, output video tensor and raw text.
        if input data is still video, this will generate a buffer numpy array file, which
        will help accelerate the data loading during training.
        :param video_dir: buffer file or raw videos dir
        :param annotation_file: the path of annotation_file
        :param int frames_num: the num of frames to be extracted from each video
        :param int frames_size: the size of frame
        :param str mode: captions will not be loaded if mode is test
        :param str buffer_save_path: the path to save buffer data
        :param int random_seed: random seed of select caption of a video
        :return CPU frames, caption
        """
        random.seed(random_seed)
        self.seed = random_seed
        self.frames_num = frames_num
        self.frames_size = frames_size

        # video transforms
        self.tf = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
        ])

        # load video
        if os.path.isdir(video_dir):
            make_video_buffer(plb.Path(video_dir).rglob("*.mp4"),
                              save_path=buffer_save_path,
                              frames_num=frames_num,
                              frames_size=frames_size,
                              compress=True)
            self.video2data = np.load(buffer_save_path)
        elif os.path.isfile(video_dir):
            self.video2data = np.load(video_dir)
        self.video_ids = list(self.video2data.keys())

        # read video split info
        with open(annotation_file, encoding='utf-8') as f:
            annotation = json.load(f)
        self.video2split = {i["video_id"]: i["split"] for i in annotation["videos"]}
        logger.info("successfully loading {} videos".format(len(self.video2data)))

        # load captions
        self.video2caption = {}
        if mode == "train" or "val":
            captions = annotation["sentences"]
            for cap in tqdm(captions, desc="Loading annotations"):
                if self.video2split[cap["video_id"]] != mode:
                    continue
                if cap["video_id"] not in self.video2caption:
                    self.video2caption[cap["video_id"]] = [cap["caption"]]
                else:
                    self.video2caption[cap["video_id"]].append(cap["caption"])
            logger.info("successfully loading {} captions".format(len(captions)))

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        caption = random.choice(self.video2caption[video_id])
        frames = []
        for frame in self.video2data[video_id][::2]:
            frames.append(self.tf(frame))
        frames = torch.stack(frames).to(dtype=torch.float)
        frames = rearrange(frames, "t c h w -> t h w c")

        return frames, caption

    def __len__(self):
        return len(self.video_ids)


def msrvtt_collate_fn(data):
    """
    stack the video data
    :param data: [(Tensor(T, H, W, C), str)]
    :return: Tensor(N, T, H, W, C), list[str, str, ...]
    """
    video_data = torch.stack([i[0] for i in data])
    text_data = [i[1] for i in data]
    return video_data, text_data


def load_single_video(video_path, frames_num, frames_size, tensor=False):
    video = mmcv.VideoReader(str(video_path))
    frame_cnt = video.frame_cnt
    samples_ix = np.linspace(0, frame_cnt - 1, frames_num).astype(int)
    frames = map(lambda x: video.get_frame(x), samples_ix)
    resized_frames = torch.stack(
        list(map(lambda x: torch.tensor(mmcv.imresize(x, (frames_size, frames_size))), frames)))
    if tensor is True:
        return resized_frames
    else:
        return resized_frames.numpy()


def make_video_buffer(video_paths, save_path, frames_num, frames_size, compress=True):
    """
    preprocess video and save buffer as numpy array
    :param compress: whether or not to do compress
    :param video_paths: list of video paths
    :param save_path: the path to save buffer
    :param frames_num: fix number of frames to extract
    :param frames_size: frames size of the video
    :return:
    """
    # video_ids = [i.stem for i in video_paths]
    all_video_arr_dict = {}  # dict(N, T, H, W, C)
    for path in tqdm(list(video_paths), desc="making buffer"):
        video_arr = load_single_video(path, frames_num, frames_size, tensor=False)
        all_video_arr_dict[path.stem] = video_arr
    if compress is True:
        np.savez_compressed(save_path, **all_video_arr_dict)
    else:
        np.savez(save_path, **all_video_arr_dict)


class MSRVTT(Dataset):
    def __init__(self, video_feat_dir: str, annotation_file: str, tokenizer,
                 mode: str = "train", include_id: bool = False):
        super(MSRVTT, self).__init__()
        self.tokenizer = tokenizer
        # load video list
        video_feat_dir = plb.Path(video_feat_dir)
        self.video_feat_list = list(video_feat_dir.glob("*.npy"))
        self.mode = mode
        self.include_id = include_id

        # load caption
        if mode == "train" or "validate":
            self.video2caption = {}
            with open(annotation_file, encoding='utf-8') as f:
                annotation = json.load(f)
            self.video2split = {i["video_id"]: i["split"] for i in annotation["videos"]}
            for cap in tqdm(annotation["sentences"], desc="Loading annotations"):
                if self.video2split[cap["video_id"]] != mode:
                    continue
                if cap["video_id"] not in self.video2caption:
                    self.video2caption[cap["video_id"]] = [cap["caption"]]
                else:
                    self.video2caption[cap["video_id"]].append(cap["caption"])

    def __getitem__(self, index):
        video_path = self.video_feat_list[index]
        vid = video_path.stem
        v_feat = torch.tensor(np.load(str(video_path)), dtype=torch.float)
        if v_feat.shape[0] > v_feat.shape[1]:
            v_feat = v_feat.transpose(0, 1)
        if self.mode == "train" or "validate":
            caption = np.random.choice(self.video2caption[vid])
            caption = self.tokenizer.encode(caption, return_tensors="pt").squeeze()
            return v_feat, caption, vid if self.include_id is True else v_feat, caption
        else:
            return v_feat, vid if self.include_id is True else v_feat

    def __len__(self):
        return len(self.video_feat_list)

    def get_a_sample(self, index=None, ori_video_dir=None):
        return_dict = {}
        index = random.randrange(0, len(self)) if index is None else index

        video_path = self.video_feat_list[index]
        vid = video_path.stem
        v_feat = torch.tensor(np.load(str(video_path)), dtype=torch.float)
        return_dict["v_feat"] = v_feat if v_feat.shape[0] > v_feat.shape[1] else v_feat.transpose(0, 1)
        return_dict["v_id"] = vid

        if self.mode == "train" or "validate":
            caption = np.random.choice(self.video2caption[vid])
            enc_caption = self.tokenizer.encode(caption, return_tensors="pt").squeeze()
            return_dict["raw_caption"] = caption
            return_dict["enc_caption"] = enc_caption

        if ori_video_dir is not None:
            ori_video_dir = plb.Path(ori_video_dir)
            assert ori_video_dir.is_dir()
            raw_video_path = list(ori_video_dir.glob(f"{vid}*"))[0]
            return_dict["raw_v_path"] = raw_video_path

        return return_dict


class VATEX(Dataset):
    def __init__(self, video_feat_dir: str, annotation_file: str, tokenizer,
                 mode: str = "train", include_id: bool = False):
        super(VATEX, self).__init__()
        self.tokenizer = tokenizer
        # load video list
        video_feat_list = list(plb.Path(video_feat_dir).glob("*.npy"))
        self.video2path = {i.stem[:11]: str(i) for i in video_feat_list}
        self.mode = mode
        self.include_id = include_id

        # load caption
        if mode == "train" or "validate":
            self.video_ids = []
            self.video2caption = {}
            with open(annotation_file, encoding='utf-8') as f:
                annotation = json.load(f)
            for cap in tqdm(annotation, desc="Loading annotations"):
                self.video_ids.append(cap["videoID"][:11])
                self.video2caption[cap["videoID"][:11]] = cap["enCap"]
        elif mode == "test":
            self.video_ids = [i.stem[:11] for i in video_feat_list]

    def __getitem__(self, index):
        vid = self.video_ids[index]
        video_path = self.video2path[vid]
        v_feat = torch.tensor(np.load(str(video_path)), dtype=torch.float).squeeze()
        if v_feat.shape[0] > v_feat.shape[1]:
            v_feat = v_feat.transpose(0, 1)
        if self.mode == "train" or "validate":
            caption = np.random.choice(self.video2caption[vid])
            caption = self.tokenizer.encode(caption, return_tensors="pt").squeeze()
            return v_feat, caption, vid if self.include_id is True else v_feat, caption
        else:
            return v_feat, vid if self.include_id is True else v_feat

    def __len__(self):
        return len(self.video_ids)


if __name__ == "__main__":
    dataset = MSR_VTT_VideoDataset(r"./data/msrvtt-train-buffer.npz",
                                   r"/data3/lzh/MSRVTT/MSRVTT-annotations/train_val_videodatainfo.json", )
    train_loader = DataLoader(dataset, collate_fn=msrvtt_collate_fn, batch_size=4)
    a = next(iter(train_loader))  # B,T,H,W,C
    # trainval = plb.Path(r"/data3/lzh/MSRVTT/MSRVTT_trainval")
    # make_video_buffer(trainval.rglob("*.mp4"),
    #                   save_path=r"./data/buffer.npz",
    #                   frames_num=40,
    #                   frames_size=224,
    #                   compress=True)

# /data3/lzh/VATEX
