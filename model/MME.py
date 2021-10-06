import torch
import torch.nn as nn
from torch.nn import Module, Linear, Dropout, AdaptiveMaxPool1d, AdaptiveAvgPool1d
from torch import Tensor
from .Embedding import PositionalEmbedding


class GlobalAggregation(Module):
    def __init__(self, method: str = "maxpooling"):
        """
        进行全局特征提取的方式
        :param method: maxpooling/avgpooling/GRU/biGRU
        """
        super(GlobalAggregation, self).__init__()
        self.method = method
        if method == "maxpooling":
            self.agg = AdaptiveMaxPool1d(1)
        elif method == "avgpooling":
            self.agg = AdaptiveAvgPool1d(1)

    def forward(self, x: Tensor):
        """
        :param x: N,T,E
        :return:
        """
        if "pooling" in self.method:
            return self.agg(x.transpose(1, 2)).transpose(1, 2)


class ExpertEmbedding(Module):
    def __init__(self, expert_num):
        super(ExpertEmbedding, self).__init__()
        self.expert_num = expert_num
        self.emb_value = 0

    def step(self):
        self.emb_value += 1

    def forward(self, x):
        """
        :param x: N, T+1, d_model
        :return:
        """
        return x + self.emb_value


class MultiModalEmbedding(Module):
    def __init__(self, d_model: int, feat_dims: dict, dropout: float, agg_method: str = "avgpooling",
                 device=torch.device('cuda')):
        """
        输入λ种特征，分别进行Input Embedding，再得到Global特征，再将local和Global一起进行Positional Embedding和Expert Embedding
        Global特征可以用GRU、Pooling等方式得到
        :param d_model: Transformer的维度
        :param feat_dims: 输入的K种特征的维度字典
        :param agg_method: 进行全局特征提取的方式 maxpooling/avgpooling/GRU/biGRU
        :param dropout: PositionalEmbedding的Dropout
        """
        super(MultiModalEmbedding, self).__init__()
        self.device = device
        # assert set(feat_dims.keys()).issubset({"scene", "motion"})
        self.input_embeddings = nn.ModuleDict({
            k: Linear(v, d_model) for k, v in feat_dims.items()
        })
        self.aggregates = nn.ModuleDict({
            k: GlobalAggregation(method=agg_method) for k in feat_dims.keys()
        })
        self.positional_embedding = PositionalEmbedding(d_model, dropout)
        self.expert_embedding = ExpertEmbedding(len(feat_dims))

    def forward(self, feats_dict: dict, feats_padding_mask_dict: dict):
        features, lengths, feats_padding_masks = [], [], []
        for feat_name, feats, padding_masks in zip(feats_dict.keys(), feats_dict.values(), feats_padding_mask_dict.values()):
            # B T E -> B T M  # M meas d_model
            feats = self.input_embeddings[feat_name](feats)
            # B 1 M
            global_feat = self.aggregates[feat_name](feats)
            # B T M + B 1 M -> B T+1 M
            feats = torch.cat([global_feat, feats], dim=1)
            # Positional Embedding
            feats = self.positional_embedding(feats)
            # Expert Embedding
            feats = self.expert_embedding(feats)
            self.expert_embedding.step()
            features.append(feats)
            lengths.append(feats.shape[1])
            # B T -> B T+1
            false_ts = (torch.zeros((feats.shape[0], 1), dtype=torch.long) == 1).to(self.device)
            feats_padding_masks.append(
                torch.cat([false_ts, padding_masks], dim=1)
            )
        features = torch.cat(features, dim=1)
        feats_padding_masks = torch.cat(feats_padding_masks, dim=1)
        return features, feats_padding_masks, lengths


