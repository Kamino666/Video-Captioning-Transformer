import numpy as np
import torch
from torch import Tensor
from typing import Optional, List
import argparse
import pathlib
import matplotlib.pyplot as plt
import seaborn
import types

from model.MMT4Caption import MMT4Caption
from dataloader import build_dataloader
from utils import configure_hardware, Config
from eval import v2t_batch


def check_validity(args):
    pass


def extract_feat(args):
    # useful attributes
    args.extract_method = args.ext_type
    args.feature_type = args.feat_type[0]
    args.video_paths = [args.video]
    # useless attribute
    args.extraction_fps = None
    args.file_with_video_paths = None
    args.flow_dir = None
    args.flow_paths = None
    args.video_dir = None
    args.on_extraction = None

    from submodules.video_features.models.CLIP.extract_clip import ExtractCLIP
    extractor = ExtractCLIP(args, external_call=True)
    feats_list = extractor(torch.zeros([1], dtype=torch.long))[0][args.feature_type]
    feats_list = torch.from_numpy(feats_list).unsqueeze(0)
    return [feats_list]


# Code from Pytorch TransformerDecoderLayer forward()
# Modified to save attention map
def attn_forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
    r"""Pass the inputs (and mask) through the decoder layer.

    Args:
        tgt: the sequence to the decoder layer (required).
        memory: the sequence from the last layer of the encoder (required).
        tgt_mask: the mask for the tgt sequence (optional).
        memory_mask: the mask for the memory sequence (optional).
        tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
        memory_key_padding_mask: the mask for the memory keys per batch (optional).

    Shape:
        see the docs in Transformer class.
    """
    tgt2, sa = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
    tgt = tgt + self.dropout1(tgt2)
    tgt = self.norm1(tgt)
    tgt2, mha = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)
    tgt = tgt + self.dropout2(tgt2)
    tgt = self.norm2(tgt)
    tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    tgt = tgt + self.dropout3(tgt2)
    tgt = self.norm3(tgt)
    # Extracted attention map:
    # save to 'self'
    # sa : (1, L, L)
    # mha: (1, L, S)
    if hasattr(self, 'mha'):
        self.mha = torch.cat([self.mha, mha[:, -1:, :]], dim=1)
        # print("yes")
    else:
        self.mha = mha
    return tgt


def visualize(layers: List[torch.nn.TransformerDecoderLayer],
              caption: Optional[str] = None, feat_lens: Optional[list] = None):
    """
    :param feat_lens: length of each feature
    :param caption: Caption of video
    :param layers: Modified TransformerDecoderLayer, which contains .sa & .mha attributes
    :return: None
    """
    mha_maps = [i.mha.squeeze(0).cpu() for i in layers]
    avg_map = torch.mean(torch.stack(mha_maps), dim=0)
    # for i, attn_map in enumerate(mha_maps + [avg_map]):
    #     plt.figure(i+1, figsize=(10, 10))
    #     seaborn.heatmap(
    #         attn_map.numpy().T,
    #         xticklabels=caption.split(' ') + ['SEP'],
    #         yticklabels=['G'] + ['m1'] * feat_lens[0] + ['G'] + ['m2'] * feat_lens[1],
    #         annot=True
    #     )
    plt.figure(figsize=(10, 10))
    seaborn.heatmap(
        avg_map.numpy().T,
        xticklabels=caption.split(' ') + ['SEP'],
        # yticklabels=['G'] + ['m1'] * feat_lens[0] + ['G'] + ['m2'] * feat_lens[1],
        annot=True
    )
    plt.show()


def predict(cfg, local_args):
    # Obtain features
    if local_args.video is not None:
        feats = extract_feat(local_args)
    else:
        feats = [torch.tensor(np.load(i), dtype=torch.float32, device=local_args.device).unsqueeze(0)
                 for i in local_args.features]
    # print(type(feats))
    feat_lens = [i.shape[1] for i in feats]
    # Build model
    model = MMT4Caption(cfg['model'], device=local_args.device).to(local_args.device)
    model.mode("caption")
    load_state = model.load_state_dict(
        torch.load(local_args.model, map_location=local_args.device), strict=False
    )
    print(f"Load state: {load_state}")
    layers = []
    for layer in model.cap_decoder.decoder.layers:
        funcType = types.MethodType
        layer.forward = funcType(attn_forward, layer)
        layers.append(layer)
    # Evaluate
    model.eval()
    result = v2t_batch(model, feats, None, max_len=cfg['test']['max_length'], local_args=local_args)[0]
    # Output
    if local_args.features is not None:
        video_id = pathlib.Path(local_args.features[0]).stem
    else:
        video_id = pathlib.Path(local_args.video).stem
    print(f"{video_id}\t:{result}")
    # [Optional] visualize attention maps
    if local_args.vis_attn:
        visualize(layers, caption=result, feat_lens=feat_lens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str, help="The path of '.json' config file")
    parser.add_argument("-m", "--model", required=True, type=str, help="The path of model checkpoint")

    # input argument
    # You can select video as input or features in .npy format as input
    # Note: The shape of numpy array should be (T, C), where T means temporal dimension
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-v", "--video", type=str, help="The path of input video")
    input_group.add_argument("-f", "--features", nargs='+', type=str, help="The paths of input features of a video")
    # if choose -v
    parser.add_argument("--feat_type", nargs='+', type=str, choices=["CLIP", "I3D", "CLIP4CLIP-ViT-B-32"],
                        help="the type of feature extractor(s)")
    parser.add_argument("--ext_type", type=str,
                        help="How to extract video frames.\n Format: [type]_[param]\n Example: fps_2 fix_20 tsn_12")

    # device argument
    device_group = parser.add_mutually_exclusive_group(required=True)  # Multi-GPU not supported
    device_group.add_argument("--cpu", action="store_true", help="use cpu or not")
    device_group.add_argument("--gpu", action="store_true", help="use gpu or not")

    # generation argument
    gen_group = parser.add_mutually_exclusive_group(required=True)
    gen_group.add_argument("--greedy", action="store_true", help="greedy decode")
    gen_group.add_argument("--beam", type=int, help="beam search decode(not support yet)")

    # others
    parser.add_argument("--vis_attn", action="store_true", help="Visualize the attention weights")

    args_ = parser.parse_args()

    # check validity of arguments
    # args_ = check_validity(args_)

    # configure hardware
    args_ = configure_hardware(args_)

    # load config
    cfg_ = Config(args_.config)
    if args_.is_main_rank:
        cfg_.display()

    predict(cfg_.data, args_)
