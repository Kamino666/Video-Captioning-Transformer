import torch
from torch.nn import Module, Linear, Dropout, ModuleList, AdaptiveMaxPool1d, AdaptiveAvgPool1d
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


class MultiModalEmbedding(Module):
    def __init__(self, d_model: int, feat_dims: list, dropout: float, agg_method: str = "avgpooling"):
        """
        输入N种feature，输出将这些feature映射到d_model维度的特征以及一个额外的Global特征
        Global特征可以用GRU、Pooling等方式得到
        :param d_model: Transformer的emd_dim
        :param feat_dims: 输入的K种特征的维度列表
        :param agg_method: 进行全局特征提取的方式 maxpooling/avgpooling/GRU/biGRU
        :param dropout: PositionalEmbedding的Dropout
        """
        super(MultiModalEmbedding, self).__init__()
        self.projs = ModuleList([Linear(d, d_model) for d in feat_dims])
        self.agg = GlobalAggregation(method=agg_method)
        self.expert_emb = list(range(10))
        self.pe = PositionalEmbedding(d_model, dropout)

    def forward(self, feats):
        video_embeddings = []
        for expert, feat, proj in zip([self.expert_emb, feats, self.projs]):
            v_feat = proj(feat)  # N T E
            agg_v_feat = self.agg(v_feat)  # N 1 E

            v_feat = v_feat + torch.ones_like(v_feat) * expert
            agg_v_feat = agg_v_feat + torch.ones_like(agg_v_feat) * expert

            video_embeddings.append(
                torch.cat([agg_v_feat, self.pe(v_feat)], dim=1)  # N T+1 E
            )
        return video_embeddings

