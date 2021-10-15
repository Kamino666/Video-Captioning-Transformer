import torch
from transformers import AutoTokenizer
from model.model import VideoTransformer
from utils import generate_square_subsequent_mask
import torch.nn.functional as F

import argparse
from timeit import default_timer as timer


# def extract_frames(video_path, mode="fix", value=40, frames_size=224):
#     video = mmcv.VideoReader(str(video_path))
#     if mode == "fix":
#         samples_ix = np.linspace(0, video.frame_cnt - 1, value).astype(int)
#     elif mode == "fps":
#         time_length = video.frame_cnt / video.fps
#         frame_num = int(time_length * value)
#         samples_ix = np.linspace(0, video.frame_cnt - 1, frame_num).astype(int)
#     else:
#         raise ValueError("模式错误，只能从fix和fps二选一")
#     frames = map(lambda x: video.get_frame(x), samples_ix)
#     resized_frames = torch.stack(
#         list(map(lambda x: torch.tensor(mmcv.imresize(x, (frames_size, frames_size))), frames)))
#     return resized_frames
#
#
# def swin_extractor(video):
#     # """load video"""
#     # video = extract_frames(video_path, mode=mode, value=value, frames_size=224).unsqueeze(0)
#     """build model"""
#     config = './checkpoint/swin_tiny_patch244_window877_kinetics400_1k.py'
#     checkpoint = './checkpoint/swin_tiny_patch244_window877_kinetics400_1k.pth'
#
#     cfg = Config.fromfile(config)
#     model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
#     load_checkpoint(model, checkpoint, map_location=str(device))
#     model.eval()
#
#     with torch.no_grad():
#         feats = model.extract_feat(video)
#         feats = torch.mean(feats, dim=[3, 4]).squeeze()  # T, E
#     return feats
class ExtractConfigs:
    _desc = "一个空的对象，用来保存属性"


def extract_feature(args):
    # build config to extract features
    cfg = ExtractConfigs()
    cfg.on_extraction = "save_numpy"
    cfg.extraction_fps = args.fps
    cfg.video_paths = [args.video]
    cfg.file_with_video_paths = None
    cfg.video_dir = None
    if args.gpu is True:
        cfg.device_ids = 0
    else:
        cfg.cpu = True
    # extract features
    if args.feat == "CLIP":
        cfg.feature_type = "CLIP-ViT-B/32"
        from submodules.video_features.models.CLIP.extract_clip import ExtractCLIP
        extractor = ExtractCLIP(cfg, True)
    else:
        raise ValueError
    feature_dict = extractor(torch.zeros([1], dtype=torch.long))[0]
    feature = feature_dict[cfg.feature_type]
    return feature


def greedy_decode(model, feature, max_len, tokenizer):
    model.eval()
    start_id = tokenizer.convert_tokens_to_ids("[CLS]")
    end_id = tokenizer.convert_tokens_to_ids("[SEP]")

    with torch.no_grad():
        feature = feature.to(device).unsqueeze(0)  # 1, T, e
        memory = model.encode(feature).to(device)  # 1, T, E
        ys = torch.ones(1, 1).fill_(start_id).type(torch.long).to(device)  # 1, 1
        for i in range(max_len - 1):
            tgt_mask = (generate_square_subsequent_mask(ys.shape[1]).type(torch.bool)).to(device)  # t, t
            out = model.decode(ys, memory, tgt_mask)
            prob = model.generator(out[:, -1])  # vocab_size
            prob = F.softmax(prob, dim=-1)
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys, torch.ones(1, 1).type(torch.long).fill_(next_word).to(device)], dim=1)  # 1, t
            if next_word == end_id:
                break
        result = ys.squeeze().tolist()
        # to text
        result_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(result))
        result_text = result_text.replace("[CLS]", "")
        result_text = result_text.replace("[SEP]", "")
    return result_text


def single_video_prediction(args):
    # extract feature
    feature = extract_feature(args)
    feature = torch.from_numpy(feature).to(torch.float).to(device)

    # build model
    if args.feat == "CLIP":
        feat_size = 512
    else:
        raise ValueError
    transformer = VideoTransformer(num_encoder_layers=args.num_encoder_layers,
                                   num_decoder_layers=args.num_decoder_layers,
                                   feat_size=feat_size,
                                   emb_size=args.emb_dim,
                                   nhead=args.num_head,
                                   use_bert=False,
                                   dim_feedforward=args.dim_feedforward,
                                   device=device).to(device)
    transformer.load_state_dict(torch.load(args.model))
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("/data3/lzh_3/video-captioning-swin-transformer/data/tk/")

    # get result
    if args.decode == "greedy":
        text = greedy_decode(transformer, feature, args.maxlen, tokenizer)
    else:
        raise ValueError
    print(f"result: {text}")


if __name__ == "__main__":
    """
    一个可以运行的命令
    python test_video -v ./data/test3.mp4 
    -m ./checkpoint/b64_lr0001_dp03_emb768_e4_d4_hd8_hi2048_MSRVTT&CLIP&SCE_loss_earlystop.pth 
    --feat CLIP --fps 3 --gpu
    """
    start_time = timer()
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument("-v", "--video", type=str, required=True, help="the path of the video")
    parser.add_argument("-m", "--model", type=str, required=True, help="the path of the model")
    parser.add_argument("--feat", type=str, default="CLIP", choices=["CLIP"], help="the feature extractor")
    parser.add_argument("--fps", type=int, default=3, help="the fps of extracted frames")

    # model args
    parser.add_argument("--num_encoder_layers", type=int, default=4, help="num_encoder_layers in transformer")
    parser.add_argument("--num_decoder_layers", type=int, default=4, help="num_decoder_layers in transformer")
    parser.add_argument("--num_head", type=int, default=8, help="the number of head in transformer")
    parser.add_argument("--emb_dim", type=int, default=768, help="the dim of text embedding layer")
    parser.add_argument("--dim_feedforward", type=int, default=2048, help="dim_feedforward in transformer")

    # inference args
    parser.add_argument("--decode", type=str, default="greedy", choices=["greedy", "beam"],
                        help="the type of tokenizer")
    parser.add_argument("--beam_width", type=int, default=3, help="the beam size")
    parser.add_argument("--maxlen", type=int, default=30, help="the max length of generated sentence")
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased", help="the type of tokenizer")

    parser.add_argument("--gpu", action="store_true", help="use gpu or not")

    args_ = parser.parse_args()
    device = torch.device("cuda") if args_.gpu else torch.device("cpu")  # TODO: 好像GPU设置目前还没起作用

    single_video_prediction(args_)
    end_time = timer()
    print(f"用时：{end_time-start_time}")
