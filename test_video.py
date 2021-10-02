import mmcv
from mmcv import Config
from mmaction.models import build_model
from mmcv.runner import load_checkpoint
import torch
from transformers import AutoTokenizer

from train import greedy_decode

import numpy as np
import argparse


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
    resized_frames = torch.stack(
        list(map(lambda x: torch.tensor(mmcv.imresize(x, (frames_size, frames_size))), frames)))
    return resized_frames


def swin_extractor(video):
    # """load video"""
    # video = extract_frames(video_path, mode=mode, value=value, frames_size=224).unsqueeze(0)
    """build model"""
    config = './checkpoint/swin_tiny_patch244_window877_kinetics400_1k.py'
    checkpoint = './checkpoint/swin_tiny_patch244_window877_kinetics400_1k.pth'

    cfg = Config.fromfile(config)
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint, map_location=str(device))
    model.eval()

    with torch.no_grad():
        feats = model.extract_feat(video)
        feats = torch.mean(feats, dim=[3, 4]).squeeze()  # T, E
    return feats


def test_single_video(feat, model_path, tokenizer_type):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
    start_id = tokenizer.convert_tokens_to_ids("[CLS]")
    end_id = tokenizer.convert_tokens_to_ids("[SEP]")

    model = torch.load(model_path, map_location=device)

    result = greedy_decode(model, feat, 30, start_id, end_id)
    result_tokens = tokenizer.convert_ids_to_tokens(result)
    result_text = tokenizer.convert_tokens_to_string(result_tokens)
    return result_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", type=str, required=True, help="the path of video")
    parser.add_argument("-m", "--model", type=str, required=True, help="the path of model")
    parser.add_argument("--feat", type=str, default="swin", choices=["I3D", "swin"], help="the feature extractor")
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased", help="the type of tokenizer")
    parser.add_argument("--gpu", action="store_true", help="use gpu or not")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fix", type=int, default=40, help="the fix number of extracted frames")
    group.add_argument("--fps", type=int, default=2, help="the fps of extracted frames")
    args = parser.parse_args()

    device = torch.device("cuda") if args.gpu else torch.device("cpu")
    if args.fix:
        mode = "fix"
    elif args.fps:
        mode = "fps"
    else:
        raise ValueError()
    frames = extract_frames(args.video, mode, )
