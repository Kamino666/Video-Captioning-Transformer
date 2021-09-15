import mmcv
import numpy as np

from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AutoTokenizer
import torch

import json
import pathlib as plb
from tqdm import tqdm
import logging
import random
import os

logger = logging.getLogger("main")
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')


def make_tokenizer(bert_type='bert-base-cased'):
    global bert_tokenizer
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_type)


class MSR_VTT_VideoDataset(Dataset):
    def __init__(self, video_dir, annotation_file,
                 frames_num=80, frames_size=224,
                 mode="train", buffer_save_path=r"./data/buffer.npz",
                 random_seed=10503):
        """ Video Dataset
            Args:
                video_dir (str): buffer file or raw videos dir
                annotation_file (str): the path of annotation_file
                frames_num (int): the num of frames to be extracted from each video
                frames_size (int): the size of frame
                mode (str): captions will not be loaded if mode is test
            """
        random.seed(random_seed)
        self.seed = random_seed
        # self.video_paths = [i for i in plb.Path(video_dir).glob("*.mp4")]
        # self.video_ids = [i.stem for i in self.video_paths]
        self.frames_num = frames_num
        self.frames_size = frames_size

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
        logger.info("successfully loading {} videos".format(len(self.video2data)))

        # load captions
        self.video2caption = {}
        if mode == "train" or "val":
            with open(annotation_file, encoding='utf-8') as f:
                annotation = json.load(f)
            captions = annotation["sentences"]
            for cap in tqdm(captions, desc="Loading annotations"):
                if cap["video_id"] not in self.video2caption:
                    self.video2caption[cap["video_id"]] = [cap["caption"]]
                else:
                    self.video2caption[cap["video_id"]].append(cap["caption"])
            logger.info("successfully loading {} captions".format(len(captions)))

    def __getitem__(self, index):
        caption = random.choice(self.video2caption[self.video_ids[index]])
        # video_path = self.video_paths[index]
        # frames = load_single_video(video_path, self.frames_num, self.frames_size, tensor=True)  # Tensor(T H W C)
        frames = torch.tensor(self.video2data[self.video_ids[index]])
        return frames, caption

    def __len__(self):
        return len(self.video_ids)


def msrvtt_collate_fn(data):
    """
    :param data: [(Tensor(T, H, W, C), str)]
    :return: Tensor(N, T, H, W, C), {"input_ids", "token_type_ids", "attention_mask"}
    """
    global bert_tokenizer
    video_data = torch.stack([i[0] for i in data]).cuda()
    text_data = [i[1] for i in data]
    tokenized_caption = bert_tokenizer(text_data)
    return video_data, tokenized_caption


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


if __name__ == "__main__":
    dataset = MSR_VTT_VideoDataset(r"./data/buffer.npz",
                                   r"/data3/lzh/MSRVTT/MSRVTT-annotations/train_val_videodatainfo.json", )
    train_loader = DataLoader(dataset, collate_fn=msrvtt_collate_fn, batch_size=4)
    a = next(iter(train_loader))  # B,T,H,W,C
    # trainval = plb.Path(r"/data3/lzh/MSRVTT/MSRVTT_trainval")
    # make_video_buffer(trainval.rglob("*.mp4"),
    #                   save_path=r"./data/buffer.npz",
    #                   frames_num=40,
    #                   frames_size=224,
    #                   compress=True)
