import numpy as np
import torch
import torch.nn as nn
from torch.nn import ModuleList
from torch import Tensor

import math
from typing import List, Optional, Any
import copy


class ModalEmbedding(nn.Module):

    def __init__(self, num_modal, d_model=512, modal_different=False, device=torch.device("cuda")):
        super().__init__()
        self.num_modal = num_modal
        self.embed_size = d_model
        self.modal_different = modal_different
        self.device = device
        if modal_different is True:
            # one for every modal and every global
            self.modal_emb = nn.Embedding(num_modal * 2, d_model)
        else:
            self.modal_emb = nn.Embedding(num_modal, d_model)

    def forward(self, modal_feats: List[Tensor]):
        """
        把各个模态的memory加上可学习的Embedding值
        -> Agg1 M1_1 M1_2 M1_3 M1_4 Agg2 M2_1 M2_2
        -> 0    0    0    0    0    1    1    1
        -> [            nn.Embedding             ]
        :param modal_feats: List[N t E]
        :return: Tensor(N T E)
        """
        modal_lens = [i.shape[1] for i in modal_feats]
        batch_size = modal_feats[0].shape[0]
        modal_labels = []  # length: T
        for i, length in enumerate(modal_lens):
            if self.modal_different is True:
                modal_labels += [i + self.num_modal]
            else:
                modal_labels += [i]
            modal_labels += (length - 1) * [i]
        modal_labels = torch.tensor(modal_labels, device=self.device)
        # T -> T E
        modal_embeddings = self.modal_emb(modal_labels)
        # T E -> N T E
        return modal_embeddings.unsqueeze(0).expand(batch_size, -1, -1)


class TemporalEncoding(nn.Module):
    """
    对多模态的特征进行时间编码（和Attention is All you need一样，和MMT论文不同）
    要解决的问题：
        1.单个模态中，不同t的元素之间要有相对关系
        2.单个模态中还要想办法处理Global所在的位置
        3.多个模态的总时长D可能不同，要有对齐的办法
    解决问题的方法：
        用Positional Encoding处理非Global的数据，Global数据加0，
        且以第一个模态（主模态）的长度为准，其它模态的时间步映射到第一个模态的时间步上
        比如： 第一个模态长度为40，第二个模态长度为20，则第二个模态的第2个时间步对应着第一个模态的第3个时间步，
        第二个模态的第3个时间步对应着第一个模态的第5个时间步……
    """

    def __init__(self, d_model=512, max_len=512, separate=False, device=torch.device("cuda")):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.separate = separate
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float().to(device)
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # 1, max_len, d_model
        self.register_buffer('pe', pe)

    def forward(self, modal_feats: List[Tensor]) -> Any:
        """
        输入n种模态，每个模态的第1个都是Global特征，即B,T+1,E
        :param modal_feats:
        :return: sum(T), E
        """
        batch_size = modal_feats[0].shape[0]
        if self.separate is False:
            D = modal_feats[0].shape[1] - 1
            temp_emb = []
            for modal in modal_feats:
                # range(t) 代表遍历_pe赋值（包含agg时要除去第一个）
                # indices  代表遍历下标获取self.pe
                # 结果_pe应该是：位置0为全0，位置1~t+1是pe
                t = modal.shape[1] - 1  # 除了Agg的长度
                indices = np.linspace(0, D - 1, t).astype(np.int32)  # [0, D-1]分成t份，包含头尾
                _pe = torch.zeros([t + 1, self.d_model])
                for i, idx in zip(range(t), indices):
                    _pe[i + 1] = self.pe[:, idx, :].squeeze()
                temp_emb.append(_pe)
            temp_emb = torch.cat(temp_emb, dim=0).to(self.device)  # (包含agg的)所有模态的长度, E
            return temp_emb.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            D = modal_feats[0].shape[1]
            temp_emb = []
            for modal in modal_feats:
                t = modal.shape[1]
                indices = np.linspace(0, D - 1, t).astype(np.int32)  # [0, D-1]分成t份，包含头尾
                _pe = torch.zeros([t, self.d_model])
                for i, idx in zip(range(t), indices):
                    _pe[i] = self.pe[:, idx, :].squeeze()
                temp_emb.append(_pe.unsqueeze(0).expand(batch_size, -1, -1))  # append [B, t, E]
            return temp_emb


class TemporalEmbedding(nn.Module):
    """
    对多模态的特征进行时间嵌入（和MMT论文一样，参数可学习）
    要解决的问题：
        1.单个模态中，不同t的元素之间要有相对关系
        2.单个模态中还要想办法处理Global所在的位置
        3.多个模态的总时长D可能不同，要有对齐的办法
    解决问题的方法：
        用Positional Encoding处理非Global的数据，Global数据加0，
        且以第一个模态（主模态）的长度为准，其它模态的时间步映射到第一个模态的时间步上
        比如： 第一个模态长度为40，第二个模态长度为20，则第二个模态的第2个时间步对应着第一个模态的第3个时间步，
        第二个模态的第3个时间步对应着第一个模态的第5个时间步……
    """

    def __init__(self, d_model=512, max_len=512, separate=False, device=torch.device("cuda")):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.separate = separate

        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, modal_feats: List[Tensor]) -> Any:
        """
        输入n种模态，每个模态的第1个都是Global特征，即B,T+1,E
        [0 1 2 3 4 5 6 7 8 9 10 0 1 3 5 7 9 ]
        [A [      modal1      ] A [ modal2 ]]
        :param modal_feats:
        :return: sum(T), E
        """
        batch_size = modal_feats[0].shape[0]
        if self.separate is False:
            D = modal_feats[0].shape[1] - 1  # 不含Agg的，最长的长度
            temp_emb = []
            for modal in modal_feats:
                t = modal.shape[1] - 1  # 除了Agg的长度
                # [1, D]分成t份，包含头尾
                indices = np.concatenate([np.zeros([1]),
                                          np.linspace(1, D, t).astype(np.int32)])
                temp_emb.append(torch.tensor(indices, dtype=torch.long, device=self.device))
            temp_emb = torch.cat(temp_emb, dim=0).unsqueeze(dim=0).to(self.device)  # 1, (包含agg的)所有模态的长度
            temp_emb = self.embedding(temp_emb)  # 1, 长, E
            return temp_emb.expand(batch_size, -1, -1)
        else:
            D = modal_feats[0].shape[1]  # 长度
            temp_emb = []
            for modal in modal_feats:
                t = modal.shape[1]
                # [0, D-1]分成t份，包含头尾，和上面不同的地方在于不需要留给agg位置了
                indices = np.linspace(0, D - 1, t).astype(np.int32)
                indices = torch.tensor(indices, dtype=torch.long, device=self.device)
                temp_emb.append(self.embedding(indices.unsqueeze(dim=0)))  # list[B, t, E]
            return temp_emb


class GlobalAggregation(nn.Module):
    def __init__(self, method: str = "max", d_model: Optional[int] = None, device=torch.device("cuda")):
        """
        进行全局特征提取的方式
        :param method: max/avg/GRU/biGRU
        """
        super(GlobalAggregation, self).__init__()
        self.method = method
        self.device = device
        if method == "max":
            self.agg = nn.AdaptiveMaxPool1d(1)
        elif method == "avg":
            self.agg = nn.AdaptiveAvgPool1d(1)
        elif method == "GRU" or "biGRU":
            assert d_model is not None
            bidirectional = True if method == "biGRU" else False
            self.agg = nn.GRU(d_model, d_model, batch_first=True, bidirectional=bidirectional)  # 2/1, N, H

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: N,T,E
        :return:  N,1,E
        """
        if self.method == "max" or self.method == "avg":
            return self.agg(x.transpose(1, 2)).transpose(1, 2).to(self.device)
        elif self.method == "GRU":
            return self.agg(x)[1].transpose(0, 1).to(self.device)
        elif self.method == "biGRU":
            return torch.sum(self.agg(x)[1], dim=0, keepdim=True).transpose(0, 1).to(self.device)


# model.video_encoder.type == 'mme' or null
class MultiModalEncoder(nn.Module):
    """
    包括统一特征维度、提取全局特征、时序嵌入、模态嵌入、Transformer Encoder
    """

    def __init__(self, d_feats: List[int], d_model: int, nhead: int,
                 dim_feedforward: int = 2048, num_encoder_layers: int = 4,
                 dropout: float = 0.1, activation: str = "gelu", global_type: str = "avg",
                 modal_different: bool = True, temporal_type: str = "embedding", do_norm: bool = False,
                 device=torch.device("cuda"),
                 ):
        super(MultiModalEncoder, self).__init__()
        self.device = device
        self.num_modal = len(d_feats)
        self.do_norm = do_norm
        # unify the dim of features
        self.unify = ModuleList([
            nn.Linear(d_feat, d_model) for d_feat in d_feats
        ])
        # extract the global info of input features
        self.global_agg = GlobalAggregation(global_type, d_model=d_model, device=device)
        # temporal embedding
        if temporal_type == "embedding":
            self.temp_emb = TemporalEmbedding(d_model, device=device)
        else:
            self.temp_emb = TemporalEncoding(d_model, device=device)
        # modal embedding (disable if one modal)
        if self.num_modal > 1:
            self.modal_emb = ModalEmbedding(self.num_modal, modal_different=modal_different,
                                            d_model=d_model, device=device)
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                   activation=activation, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, nn.LayerNorm(d_model))
        # Normalization & Dropout
        if do_norm:
            self.norm = nn.LayerNorm(d_model)
            self.dp = nn.Dropout(dropout)

    def forward(self, srcs: List[Tensor], src_padding_masks: Optional[List[Tensor]]):
        batch_size = srcs[0].shape[0]
        uni_feats = [self.unify[i](src) for i, src in enumerate(srcs)]

        global_feats = [
            torch.cat([self.global_agg(f), f], dim=1) for f in uni_feats
        ]  # n * [B 1+T E]

        if src_padding_masks is not None:
            # src_padding_masks List[Tensor(B,T)] -> global_masks Tensor(B, n(T+1))
            global_masks = []
            for mask in src_padding_masks:
                new_mask = torch.cat([(torch.ones([batch_size, 1], device=self.device) == 0), mask], dim=1)
                global_masks.append(new_mask)
            # global_masks = torch.cat(global_masks, dim=1)
        else:
            global_masks = None

        temp_embedding = self.temp_emb(global_feats)
        modal_embedding = self.modal_emb(global_feats) if self.num_modal > 1 else None

        global_feats = torch.cat(global_feats, dim=1)
        global_masks = torch.cat(global_masks, dim=1) if global_masks is not None else None

        if self.num_modal > 1:
            mm_src = temp_embedding + modal_embedding + global_feats
        else:
            mm_src = temp_embedding + global_feats
        if self.do_norm:
            mm_src = self.dp(self.norm(mm_src))  # Norm(B,S,E)
        memory = self.transformer_encoder(mm_src, None, global_masks)
        # B, NT, E
        return memory, global_masks, memory[:, 0]


# model.video_encoder.type == 'simple'
class SimpleSepEncoder(nn.Module):
    def __init__(self, d_feats: List[int], d_model: int, nhead: int,
                 dim_feedforward: int, num_encoder_layers: int,
                 dropout: float, activation: str,
                 device=torch.device("cuda"),
                 ):
        super(SimpleSepEncoder, self).__init__()
        self.device = device
        self.num_modal = len(d_feats)
        # unify the dim of features
        self.unify = ModuleList([nn.Linear(d_feat, d_model) for d_feat in d_feats])
        # separate encoders
        base_enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation=activation, batch_first=True)
        base_enc = nn.TransformerEncoder(base_enc_layer, num_encoder_layers, nn.LayerNorm(d_model))
        self.transformer_encoders = _get_clones(base_enc, self.num_modal)
        # temporal embedding
        self.temp_emb = TemporalEncoding(d_model, separate=True, device=device)

    def forward(self, srcs: List[Tensor], src_padding_masks: Optional[List[Tensor]]):
        uni_feats = [self.unify[i](src) for i, src in enumerate(srcs)]
        temp_embeddings = self.temp_emb(uni_feats)
        memories = []
        for i, (uni_feat, temp_emb) in enumerate(zip(uni_feats, temp_embeddings)):
            if src_padding_masks is not None:
                temp_emb = temp_emb.to(self.device)
                memories.append(self.transformer_encoders[i](uni_feat + temp_emb, None, src_padding_masks[i]))
            else:
                temp_emb = temp_emb.to(self.device)
                memories.append(self.transformer_encoders[i](uni_feat + temp_emb))
        return torch.cat(memories, dim=1), None, None


# Hierarchical Multi Modal Encoder
class HMMEncoder(nn.Module):
    def __init__(self, d_feats: List[int], d_model: int, nhead: int,
                 dim_feedforward: int, num_encoder_layers: List[int],
                 dropout: float = 0.1, activation: str = "gelu", global_type: str = "avg",
                 modal_different: bool = True, temporal_type: str = "embedding", do_norm: bool = False,
                 device=torch.device("cuda"),
                 ):
        super(HMMEncoder, self).__init__()
        self.device = device
        self.num_modal = len(d_feats)
        self.do_norm = do_norm
        self.num_encoder_layers = num_encoder_layers
        # unify the dim of features
        self.unify = ModuleList([
            nn.Linear(d_feat, d_model) for d_feat in d_feats
        ])
        # extract the global info of input features
        self.global_agg = GlobalAggregation(global_type, d_model=d_model, device=device)
        # temporal embedding
        if temporal_type == "embedding":
            self.temp_emb = TemporalEmbedding(d_model, device=device)
        else:
            self.temp_emb = TemporalEncoding(d_model, device=device)
        # modal embedding (disable if one modal)
        if self.num_modal > 1:
            self.modal_emb = ModalEmbedding(self.num_modal, modal_different=modal_different,
                                            d_model=d_model, device=device)
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                   activation=activation, batch_first=True)
        self.trans_enc_layers = _get_clones(encoder_layer, max(num_encoder_layers))
        # Normalization & Dropout
        if do_norm:
            self.norm = nn.LayerNorm(d_model)
            self.dp = nn.Dropout(dropout)

    def forward(self, srcs: List[Tensor], src_padding_masks: Optional[List[Tensor]]):
        batch_size = srcs[0].shape[0]
        uni_feats = [self.unify[i](src) for i, src in enumerate(srcs)]
        feat_lens = [i.shape[1]+1 for i in uni_feats]

        global_feats = [
            torch.cat([self.global_agg(f), f], dim=1) for f in uni_feats
        ]  # n * [B 1+T E]

        if src_padding_masks is not None:
            # src_padding_masks List[Tensor(B,T)] -> global_masks Tensor(B, n(T+1))
            global_masks = []
            for mask in src_padding_masks:
                new_mask = torch.cat([(torch.ones([batch_size, 1], device=self.device) == 0), mask], dim=1)
                global_masks.append(new_mask)
            # global_masks = torch.cat(global_masks, dim=1)
        else:
            global_masks = None

        temp_embedding = self.temp_emb(global_feats)
        modal_embedding = self.modal_emb(global_feats) if self.num_modal > 1 else None

        global_feats = torch.cat(global_feats, dim=1)
        global_masks = torch.cat(global_masks, dim=1) if global_masks is not None else None

        if self.num_modal > 1:
            mm_src = temp_embedding + modal_embedding + global_feats
        else:
            mm_src = temp_embedding + global_feats
        if self.do_norm:
            mm_src = self.dp(self.norm(mm_src))  # Norm(B,S,E)

        # Hierarchical Design
        # memory = self.transformer_encoder(mm_src, None, global_masks)

        target_layer = [max(self.num_encoder_layers) - i for i in self.num_encoder_layers]
        ori_input = mm_src.split(feat_lens, dim=1)
        last_outputs = [None] * self.num_modal
        # i-th layer
        for i, mod in enumerate(self.trans_enc_layers):
            inputs = []
            # j-th modal
            for j, last_output in enumerate(last_outputs):
                if target_layer[j] < i:
                    inputs.append(last_output)
                else:
                    inputs.append(ori_input[j])
            last_outputs = mod(torch.cat(inputs, dim=1), src_key_padding_mask=global_masks)
            last_outputs = last_outputs.split(feat_lens, dim=1)  # B, t1+t2+t3, E
        agg_feats = torch.sum(torch.cat([i[:, 0] for i in last_outputs], dim=1), dim=1)
        memory = torch.cat(last_outputs, dim=1)
        # B, NT, E
        return memory, global_masks, agg_feats


# Hierarchical Multi Modal Encoder
# class HMMEncoder_Sep(nn.Module):
#     def __init__(self, d_feats: List[int], d_model: int, nhead: int,
#                  dim_feedforward: int, num_encoder_layers: List[int],
#                  dropout: float = 0.1, activation: str = "gelu", global_type: str = "avg",
#                  modal_different: bool = True, temporal_type: str = "embedding", do_norm: bool = False,
#                  device=torch.device("cuda"),
#                  ):
#         super(HMMEncoder_Sep, self).__init__()
#         self.device = device
#         self.num_modal = len(d_feats)
#         self.do_norm = do_norm
#         self.num_encoder_layers = num_encoder_layers
#         # unify the dim of features
#         self.unify = ModuleList([
#             nn.Linear(d_feat, d_model) for d_feat in d_feats
#         ])
#         # extract the global info of input features
#         self.global_agg = GlobalAggregation(global_type, d_model=d_model, device=device)
#         # temporal embedding
#         if temporal_type == "embedding":
#             self.temp_emb = TemporalEmbedding(d_model, device=device)
#         else:
#             self.temp_emb = TemporalEncoding(d_model, device=device)
#         # modal embedding (disable if one modal)
#         if self.num_modal > 1:
#             self.modal_emb = ModalEmbedding(self.num_modal, modal_different=modal_different,
#                                             d_model=d_model, device=device)
#         # Transformer Encoder
#         encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
#                                                    activation=activation, batch_first=True)
#         self.trans_enc_layers = _get_clones(encoder_layer, max(num_encoder_layers))
#         # Normalization & Dropout
#         if do_norm:
#             self.norm = nn.LayerNorm(d_model)
#             self.dp = nn.Dropout(dropout)
#
#     def forward(self, srcs: List[Tensor], src_padding_masks: Optional[List[Tensor]]):
#         batch_size = srcs[0].shape[0]
#         uni_feats = [self.unify[i](src) for i, src in enumerate(srcs)]
#         feat_lens = [i.shape[1]+1 for i in uni_feats]
#
#         global_feats = [
#             torch.cat([self.global_agg(f), f], dim=1) for f in uni_feats
#         ]  # n * [B 1+T E]
#
#         if src_padding_masks is not None:
#             # src_padding_masks List[Tensor(B,T)] -> global_masks Tensor(B, n(T+1))
#             global_masks = []
#             for mask in src_padding_masks:
#                 new_mask = torch.cat([(torch.ones([batch_size, 1], device=self.device) == 0), mask], dim=1)
#                 global_masks.append(new_mask)
#             # global_masks = torch.cat(global_masks, dim=1)
#         else:
#             global_masks = None
#
#         temp_embedding = self.temp_emb(global_feats)
#         modal_embedding = self.modal_emb(global_feats) if self.num_modal > 1 else None
#
#         global_feats = torch.cat(global_feats, dim=1)
#         global_masks = torch.cat(global_masks, dim=1) if global_masks is not None else None
#
#         if self.num_modal > 1:
#             mm_src = temp_embedding + modal_embedding + global_feats
#         else:
#             mm_src = temp_embedding + global_feats
#         if self.do_norm:
#             mm_src = self.dp(self.norm(mm_src))  # Norm(B,S,E)
#
#         # Hierarchical Design
#         # memory = self.transformer_encoder(mm_src, None, global_masks)
#
#         target_layer = [max(self.num_encoder_layers) - i for i in self.num_encoder_layers]
#         ori_input = mm_src.split(feat_lens, dim=1)
#         last_outputs = [None] * self.num_modal
#         # i-th layer
#         for i, mod in enumerate(self.trans_enc_layers):
#             inputs = []
#             # j-th modal
#             for j, last_output in enumerate(last_outputs):
#                 if target_layer[j] < i:
#                     inputs.append(last_output)
#                 else:
#                     inputs.append(ori_input[j])
#             last_outputs = mod(torch.cat(inputs, dim=1), src_key_padding_mask=global_masks)
#             last_outputs = last_outputs.split(feat_lens, dim=1)  # B, t1+t2+t3, E
#         memory = torch.cat(last_outputs, dim=1)
#         # B, NT, E
#         return memory, global_masks, memory[:, 0]



def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for _ in range(N)])
