import mmcv
import numpy as np

from torch.utils.data import Dataset
import torch

import json
import pathlib as plb
from tqdm import tqdm
import random

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
                 mode: str = "train", include_id: bool = False, return_all_captions=False,
                 by_caption=True):
        super(MSRVTT, self).__init__()
        self.tokenizer = tokenizer
        # load video list
        video_feat_dir = plb.Path(video_feat_dir)
        self.video_feat_list = list(video_feat_dir.glob("*.npy"))
        self.mode = mode
        self.include_id = include_id
        self.return_all_captions = return_all_captions
        self.by_caption = by_caption

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

            if self.by_caption is True:
                video2path = {i.stem:i for i in self.video_feat_list}
                self.cap_vid_list = []
                for video, captions in self.video2caption.items():
                    for cap in captions:
                        self.cap_vid_list.append((cap, video2path[video]))

    def _getitem_by_video(self, index):
        video_path = self.video_feat_list[index]
        vid = video_path.stem
        v_feat = torch.tensor(np.load(str(video_path)), dtype=torch.float)
        v_feat = v_feat.transpose(0, 1) if v_feat.shape[0] > v_feat.shape[1] else v_feat
        if self.mode == "train" or "validate":
            if self.return_all_captions:
                captions = self.video2caption[vid]
                captions = [self.tokenizer.encode(caption, return_tensors="pt").squeeze()
                            for caption in captions]
                return v_feat, captions, vid if self.include_id is True else v_feat, captions
            else:
                caption = np.random.choice(self.video2caption[vid])
                caption = self.tokenizer.encode(caption, return_tensors="pt").squeeze()
                return v_feat, caption, vid if self.include_id is True else v_feat, caption
        else:
            return v_feat, vid if self.include_id is True else v_feat

    def _getitem_by_caption(self, index):
        caption, v_path = self.cap_vid_list[index]
        v_feat = torch.tensor(np.load(str(v_path)), dtype=torch.float)
        v_feat = v_feat.transpose(0, 1) if v_feat.shape[0] > v_feat.shape[1] else v_feat
        caption = self.tokenizer.encode(caption, return_tensors="pt").squeeze()
        return v_feat, caption, v_path.stem if self.include_id is True else v_feat, caption

    def __getitem__(self, index):
        if self.by_caption is False:
            return self._getitem_by_video(index)
        else:
            assert self.mode != "test"
            return self._getitem_by_caption(index)

    def __len__(self):
        return len(self.cap_vid_list) if self.by_caption is True else len(self.video_feat_list)

    def get_a_sample(self, index=None, ori_video_dir=None):
        return_dict = {}
        index = random.randrange(0, len(self.video_feat_list)) if index is None else index

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

