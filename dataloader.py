import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
from torch import Tensor

import json
import pathlib as plb
from tqdm import tqdm
import random
import abc
from typing import List, Tuple, Any, Callable, Union, Dict


# out of date
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
                video2path = {i.stem: i for i in self.video_feat_list}
                self.cap_vid_list = []
                for video, captions in self.video2caption.items():
                    for cap in captions:
                        if len(cap) <= 80:  # DEBUG: 暂时过滤太长的句子
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


# out of date
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


# out of date
class MultiModalMSRVTT(Dataset):
    def __init__(self, video_feat_dirs: List[str], annotation_file: str, tokenizer,
                 mode: str = "train"):
        super(MultiModalMSRVTT, self).__init__()
        self.tokenizer = tokenizer
        self.mode = mode

        # load video list
        self.video_feat_list = self._load_video_list(video_feat_dirs)

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

            video2path = {i[0].stem: i for i in self.video_feat_list}
            self.cap_vid_list = []
            for video, captions in self.video2caption.items():
                for cap in captions:
                    self.cap_vid_list.append((cap, video2path[video]))

    def _getitem_by_caption(self, index) -> Any:
        caption, v_paths = self.cap_vid_list[index]
        v_feats = []
        for v_path in v_paths:
            v_feat = torch.tensor(np.load(str(v_path)), dtype=torch.float)
            v_feats.append(
                v_feat.transpose(0, 1) if v_feat.shape[0] > v_feat.shape[1] else v_feat
            )
        caption = self.tokenizer.encode(caption, return_tensors="pt").squeeze()
        return v_feats, caption, v_paths[0].stem

    def __getitem__(self, index):
        assert self.mode != "test"
        return self._getitem_by_caption(index)

    def __len__(self):
        return len(self.cap_vid_list)

    def get_a_sample(self, index=None, ori_video_dir=None):
        return_dict = {}
        index = random.randrange(0, len(self.video_feat_list)) if index is None else index

        v_feats, _, vid = self._getitem_by_caption(index)

        return_dict["v_feat"] = v_feats
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

    def _load_video_list(self, video_feat_dirs) -> List[Tuple]:
        video_feat_list = []
        for vdirs in video_feat_dirs:
            video_feat_list.append(plb.Path(vdirs).glob("*.npy"))
        return list(zip(*video_feat_list))


def _make_mask_video(ts: Union[List[Tensor], Tuple[Tensor]]):
    """
    :param ts: List[Tensor(t, E)]
    """
    batch_size = len(ts)
    feat_dim = ts[0].shape[1]
    feat_len = [i.shape[0] for i in ts]
    max_len = max(feat_len)
    feat_ts = torch.zeros([batch_size, max_len, feat_dim], dtype=torch.float)
    feat_mask_ts = torch.ones([batch_size, max_len], dtype=torch.long)
    for i in range(batch_size):
        feat_ts[i, :feat_len[i]] = ts[i]
        feat_mask_ts[i, :feat_len[i]] = 0
    feat_mask_ts: Tensor = (feat_mask_ts == 1)  # IT IS TENSOR!
    return feat_ts, feat_mask_ts


# out of date
def _make_mask_text(text: List[Tensor], pad_id: int):
    """
    :param text: List[Tensor(t, E)]
    """
    # text
    batch_size = len(text)
    text_len = [len(i) for i in text]
    max_len = max(text_len)
    text_ts = torch.ones([batch_size, max_len], dtype=torch.long) * pad_id
    for i in range(batch_size):
        text_ts[i, :text_len[i]] = text[i]
    text_mask_ts = (text_ts == pad_id)
    return text_ts, text_mask_ts


def _make_multi_mask_video(batch_modal_feat: List[Tuple[Tensor]]):
    modal_batch_feat = list(zip(*batch_modal_feat))  # list: M, B, T, E
    modal_feat_ts, modal_feat_mask_ts = [], []
    for batch_feat in modal_batch_feat:
        feat_ts, feat_mask_ts = _make_mask_video(batch_feat)
        modal_feat_ts.append(feat_ts)
        modal_feat_mask_ts.append(feat_mask_ts)
    # return (M),B,T,E and (M),B,T   M is the length of list
    return modal_feat_ts, modal_feat_mask_ts


# out of date
def build_collate_fn(pad_id: int, include_id: bool, multi_modal: bool = False) -> Callable:
    ItemType = Tuple[List[Tensor], Tensor, str]

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

    def mm_func(data: List[ItemType]):
        """
        v_feats, caption, v_paths[0].stem
        """
        batch_feats, batch_captions, batch_vids = list(zip(*data))
        feat_ts, feat_mask_ts = _make_multi_mask_video(batch_feats)
        text_ts, text_mask_ts = _make_mask_text(batch_captions, pad_id=pad_id)
        return feat_ts, text_ts, feat_mask_ts, text_mask_ts, batch_vids

    if multi_modal is True:
        return mm_func
    if include_id is True:
        return func1
    else:
        return func2


class Core_Dataset(Dataset, metaclass=abc.ABCMeta):
    def __init__(self, video_feat_dirs: List[str], annotation_file: str):
        """
        Core abstract class for loading dataset.

        :param video_feat_dirs: the directory of single split video features. all features are .npy format.
        :param annotation_file: the path of annotation file of dataset
        """
        self.annotation_file = annotation_file
        self.video_feat_dirs = video_feat_dirs
        # load video list
        self.video_feat_list: List[Tuple[plb.PosixPath]] = self._load_video_list()
        self.cap_vid_list, self.video2caption = self.make_cap_vid_list()

    def _load_video_list(self) -> List[Tuple]:
        video_feat_list = []
        for vdirs in self.video_feat_dirs:
            video_feat_list.append(plb.Path(vdirs).glob("*.npy"))
        return list(zip(*video_feat_list))

    @abc.abstractmethod
    def make_cap_vid_list(self) -> List[Tuple[str, Tuple]]:
        raise NotImplementedError

    def _getitem_by_caption(self, index) -> Any:
        caption, v_paths = self.cap_vid_list[index]
        v_feats = []
        for v_path in v_paths:
            v_feat = torch.tensor(np.load(str(v_path)), dtype=torch.float32)
            v_feats.append(
                v_feat.transpose(0, 1) if v_feat.shape[0] > v_feat.shape[1] else v_feat
            )
        return v_feats, caption, v_paths[0].stem

    def _getitem_by_video(self, index) -> Any:
        v_paths = self.video_feat_list[index]
        v_feats = []
        for v_path in v_paths:
            v_feat = torch.tensor(np.load(str(v_path)), dtype=torch.float32)
            v_feats.append(
                v_feat.transpose(0, 1) if v_feat.shape[0] > v_feat.shape[1] else v_feat
            )
        return v_feats, "", v_paths[0].stem


class MSRVTT_Dataset(Core_Dataset):
    def __init__(self, video_feat_dirs: List[str], annotation_file: str,
                 split_type="train", mode: str = "by_caption",
                 debug: bool = False, debug_num: int = 400):
        if split_type.lower() in ['val', 'validate']:
            split_type = "validate"
        self.split_type = split_type
        self.mode = mode
        super().__init__(video_feat_dirs, annotation_file)
        if debug is True:
            self.cap_vid_list = self.cap_vid_list[:debug_num]

    def make_cap_vid_list(self) -> Tuple[List[Tuple[Any, Any]], Dict[Any, List]]:
        """
        Provide self.split_type, self.annotation_file and self.video_feat_list
        Out put List of cap-vid pair like ("A man is skiing", ("./data1/video123.npy", "./data2/video123.npy"))
        :return:
        """
        cap_vid_list = []
        video2caption = {}
        # load caption
        if self.split_type == "train" or "validate":
            with open(self.annotation_file, encoding='utf-8') as f:
                annotation = json.load(f)
            video2split = {i["video_id"]: i["split"] for i in annotation["videos"]}
            for cap in tqdm(annotation["sentences"], desc="Loading annotations"):
                if video2split[cap["video_id"]] != self.split_type:
                    continue
                if cap["video_id"] not in video2caption:
                    video2caption[cap["video_id"]] = [cap["caption"]]
                else:
                    video2caption[cap["video_id"]].append(cap["caption"])

            video2path = {i[0].stem: i for i in self.video_feat_list}
            for video, captions in video2caption.items():
                for cap in captions:
                    cap_vid_list.append((cap, video2path[video]))
        return cap_vid_list, video2caption

    def __getitem__(self, index):
        if self.mode == 'by_caption':
            return self._getitem_by_caption(index)
        elif self.mode == 'by_video':
            return self._getitem_by_video(index)
        else:
            raise ValueError

    def __len__(self):
        if self.mode == 'by_caption':
            return len(self.cap_vid_list)
        elif self.mode == 'by_video':
            return len(self.video_feat_list)
        else:
            raise ValueError


class MSVD_Dataset(Core_Dataset):
    def __init__(self, video_feat_dirs: List[str], annotation_file: str,
                 split_type="train", mode: str = "by_caption",
                 debug: bool = False, debug_num: int = 400):
        if split_type.lower() in ['val', 'validate']:
            split_type = "validate"
        self.split_type = split_type
        self.mode = mode
        super().__init__(video_feat_dirs, annotation_file)
        if debug is True:
            self.cap_vid_list = self.cap_vid_list[:debug_num]

    def make_cap_vid_list(self) -> Tuple[List[Tuple[Any, Any]], Dict[Any, List]]:
        """
        Provide self.split_type, self.annotation_file and self.video_feat_list
        Output List of cap-vid pair like ("A man is skiing", ("./data1/video123.npy", "./data2/video123.npy"))
        Output Dict video2caption
        :return:
        """
        cap_vid_list, video2caption = [], {}
        with open(self.annotation_file) as f:
            for line in f.readlines():
                vid = line.split(" ")[0]
                cap = " ".join(line.split(" ")[1:])
                cap = cap.replace("\n", "")
                if vid not in video2caption:
                    video2caption[vid] = [cap]
                else:
                    video2caption[vid].append(cap)
        video2path = {i[0].stem: i for i in self.video_feat_list}
        for video, captions in video2caption.items():
            for cap in captions:
                cap_vid_list.append((cap, video2path[video]))
        return cap_vid_list, video2caption

    def __getitem__(self, index):
        if self.mode == 'by_caption':
            return self._getitem_by_caption(index)
        elif self.mode == 'by_video':
            return self._getitem_by_video(index)
        else:
            raise ValueError

    def __len__(self):
        if self.mode == 'by_caption':
            return len(self.cap_vid_list)
        elif self.mode == 'by_video':
            return len(self.video_feat_list)
        else:
            raise ValueError


def collate_fn(data: List[Tuple[List[Tensor], Tensor, str]]):
    batch_feats, batch_captions, batch_vids = list(zip(*data))
    feat_ts, feat_mask_ts = _make_multi_mask_video(batch_feats)
    return feat_ts, feat_mask_ts, batch_captions, batch_vids


def build_dataloader(data_cfg: dict, multi_gpu: bool):
    dataset_typt = data_cfg.get("dataset", "msrvtt")
    if dataset_typt == "msrvtt":
        data_iter = MSRVTT_Dataset(data_cfg['feat_dir'], data_cfg['annotation_path'],
                                   split_type=data_cfg['split_mode'], mode=data_cfg['mode'],
                                   debug=data_cfg['_debug'], debug_num=data_cfg['_debug_num'])
    else:
        data_iter = MSVD_Dataset(data_cfg['feat_dir'], data_cfg['annotation_path'],
                                 split_type=data_cfg['split_mode'], mode=data_cfg['mode'],
                                 debug=data_cfg['_debug'], debug_num=data_cfg['_debug_num'])
    # Sampler only works in train_dataloader when using multi-GPUs
    data_sampler = DistributedSampler(data_iter, shuffle=True) \
        if (data_cfg['split_mode'] == "train" and multi_gpu) else None

    dataloader = DataLoader(
        data_iter, batch_size=data_cfg['batch_size'],
        collate_fn=collate_fn, sampler=data_sampler,
        shuffle=(data_cfg['split_mode'] == 'train' and not multi_gpu)
    )
    return data_iter, dataloader, data_sampler
