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

logger = logging.getLogger("main")
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')


def make_tokenizer(bert_type='bert-base-cased'):
    global bert_tokenizer
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_type)


class MSR_VTT_VideoDataset(Dataset):
    def __init__(self, video_dir, annotation_file, bert_tokenizer,
                 frames_num=80, frames_size=224, ):
        """ Video Dataset
            Args:
                video_dir (str): the path of videos
                annotation_file (str): the path of annotation_file
                frames_num (int): the num of frames to be extracted from each video
                frames_size (int): the size of frame
                bert_tokenizer (obj AutoTokenizer): the type of BERT model. Could be "bert-base-uncased", "bert-base-cased" an so on.
                    Full list https://huggingface.co/models?sort=downloads&search=bert
            """
        # load captions
        self.video2caption = {}
        with open(annotation_file, encoding='utf-8') as f:
            annotation = json.load(f)
        captions = annotation["senteces"]
        for cap in tqdm(captions, desc="Loading annotations"):
            if cap["video_id"] not in self.video2caption:
                self.video2caption[cap["video_id"]] = [cap["caption"]]
            else:
                self.video2caption[cap["video_id"]].append(cap["caption"])
        logger.info("successfully loading {} captions".format(len(captions)))
        self.video_ids = list(self.video2caption.keys())

        # init BERT
        self.bert_tokenizer = bert_tokenizer

        # load video paths
        self.video_paths = [i for i in plb.Path(video_dir).glob("*.mp4")]

        # load video frames into memory and resize
        self.video2frames = {}
        for video_path in tqdm(self.video_paths):
            video = mmcv.VideoReader(video_path)
            frame_cnt = video_path.frame_cnt
            samples_ix = np.linspace(0, frame_cnt - 1, frames_num).astype(int)
            frames = map(lambda x: video.get_frame(x), samples_ix)
            resized_frames = torch.stack(list(map(lambda x: mmcv.imresize(x, (frames_size, frames_size)), frames)))
            self.video2frames[video_path.stem] = resized_frames
        logger.info("successfully loading {} videos".format(len(self.video2frames)))

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        frames = self.video2frames[video_id]  # Tensor(T, H, W, C)
        caption = random.choice(self.video2caption[video_id])
        return frames, caption


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


if __name__ == "__main__":
    dataset = MSR_VTT_VideoDataset()
    train_loader = DataLoader(dataset, collate_fn=msrvtt_collate_fn, batch_size=4)
