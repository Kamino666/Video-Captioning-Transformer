import mmcv
import numpy as np

from torch.utils.data import Dataset

import json
import pathlib as plb
from tqdm import tqdm
import logging

logger = logging.getLogger("main")


class MSR_VTT_VideoDataset(Dataset):
    def __init__(self, video_dir, annotation_file,
                 frames_num=80, frames_size=224):
        """ Video Dataset
            Args:
                video_dir (str): the path of videos
                annotation_file (str): the path of annotation_file
                frames_num (int): the num of frames to be extracted from each video
                frames_size (int): the size of frame
            """
        # load captions
        with open(annotation_file, encoding='utf-8') as f:
            annotation = json.load(f)
        captions = annotation["senteces"]
        self.video2caption = {}
        for cap in tqdm(captions, desc="Loading annotations"):
            if cap["video_id"] not in self.video2caption:
                self.video2caption[cap["video_id"]] = [cap["caption"]]
            else:
                self.video2caption[cap["video_id"]].append(cap["caption"])
        logger.info("successfully loading {} captions".format(len(captions)))

        # load video paths
        self.video_paths = [i for i in plb.Path(video_dir).glob("*.mp4")]

        # load video frames into memory and resize
        self.video2frames = {}
        for video_path in tqdm(self.video_paths):
            video = mmcv.VideoReader(video_path)
            frame_cnt = video_path.frame_cnt
            samples_ix = np.linspace(0, frame_cnt - 1, frames_num).astype(int)
            frames = map(lambda x: video.get_frame(x), samples_ix)
            resized_frames = list(map(lambda x: mmcv.imresize(x, (frames_size, frames_size)), frames))
            self.video2frames[video_path.stem] = resized_frames
        logger.info("successfully loading {} videos".format(len(self.video2frames)))

    def __getitem__(self, index):
        pass