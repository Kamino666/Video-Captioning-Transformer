import torch
import torch.nn as nn
from torch import Tensor
from typing import List

from .Embedding import PositionalEmbedding
from .MME import MultiModalEmbedding
from torch.nn import Transformer
from transformers import BertModel, AutoTokenizer


# Seq2Seq Network
# [N,T,C] and [N,S,E] -> [N, len, vocab_size]
class VideoTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 feat_size: int,
                 emb_size: int,
                 nhead: int,
                 bert_type: str = "bert-base-uncased",
                 use_bert: bool = True,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 device: torch.device = torch.device("cuda")
                 ):
        super(VideoTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       activation="gelu",
                                       batch_first=True)
        self.tokenizer = AutoTokenizer.from_pretrained("./data/tk/")
        self.generator = nn.Linear(emb_size, self.tokenizer.vocab_size)
        self.src_to_emb = nn.Linear(feat_size, emb_size)
        self.emb_size = emb_size
        self.device = device
        self.use_bert = use_bert
        if use_bert is True:
            self.tgt_to_emb = BertModel.from_pretrained(bert_type)
        else:
            pad_id = self.tokenizer.convert_tokens_to_ids("[PAD]")
            self.tgt_to_emb = nn.Embedding(self.tokenizer.vocab_size, emb_size, padding_idx=pad_id)
        self.positional_encoding = PositionalEmbedding(emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor = None):
        src_emb = self.positional_encoding(self.src_to_emb(src))  # src: torch.Size([16, 768, 20])
        if self.use_bert is True:
            # tgt: N T
            tgt_emb = torch.zeros([tgt.shape[0], tgt.shape[1], self.emb_size], dtype=torch.float, device=self.device)
            for i in range(tgt.shape[1]):
                tgt_emb[:, i, :] = self.tgt_to_emb(tgt[:, :i + 1, :]).last_hidden_state.to(self.device)[:, i, :]
            tgt_emb = self.positional_encoding(tgt_emb)
        else:
            tgt_emb = self.positional_encoding(self.tgt_to_emb(tgt))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor):
        return self.transformer.encoder(self.positional_encoding(self.src_to_emb(src)))

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        if self.use_bert is True:
            return self.transformer.decoder(
                self.positional_encoding(self.tgt_to_emb(tgt).last_hidden_state.to(self.device)), memory, tgt_mask
            )
        else:
            return self.transformer.decoder(
                self.positional_encoding(self.tgt_to_emb(tgt)), memory, tgt_mask
            )

    def freeze_bert(self):
        assert self.use_bert is True
        for name, v in self.named_parameters():
            if "tgt_to_emb" in name:
                v.requires_grad = False


class MMVideoTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 feat_sizes: List[int],
                 d_model: int,
                 nhead: int,
                 bert_type: str = "bert-base-uncased",
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 device: torch.device = torch.device("cuda"),
                 agg_method: str = "avgpooling",
                 ):
        super(MMVideoTransformer, self).__init__()
        feat_num = len(feat_sizes)
        self.feat_num = feat_num
        self.emb_size = d_model
        self.device = device

        tokenizer = AutoTokenizer.from_pretrained(bert_type)
        pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
        vocab_size = tokenizer.vocab_size
        self.tgt_to_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.positional_encoding = PositionalEmbedding(d_model, dropout=dropout)
        self.mme = MultiModalEmbedding(d_model, feat_sizes, dropout, agg_method=agg_method)

        self.encoders = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True),
                                  num_encoder_layers,
                                  nn.LayerNorm(d_model)) for _ in range(feat_num)
        ])

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.generator = nn.Linear(d_model, self.tokenizer.vocab_size)

    def forward(self,
                srcs: List[Tensor],
                tgt: Tensor,
                src_masks: List[Tensor],
                tgt_mask: Tensor,
                src_padding_masks: List[Tensor],
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor = None):
        memories = []
        for src, src_mask, src_padding_mask in zip(srcs, src_masks, src_padding_masks):
            src = self.positional_encoding(src)  # src: N T E
            memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_padding_mask)  # memory N T E
            memories.append(memory)
        video_embeddings = self.mme(memories)  # List[(N, T+1, E)]
        video_embeddings = torch.cat(video_embeddings, dim=1)  # N K(T+1) E

        tgt_emb = self.positional_encoding(self.tgt_to_emb(tgt))
        output = self.decoder(tgt_emb, video_embeddings, tgt_mask=tgt_mask, memory_mask=None,
                              tgt_key_padding_mask=tgt_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return self.generator(output)  # N S E -> N S vocab_size

    def encode(self, srcs):
        memories = []
        for src in srcs:
            src = self.positional_encoding(src)  # src: N T E
            memory = self.encoder(src)  # memory N T E
            memories.append(memory)
        video_embeddings = self.mme(memories)  # List[(N, T+1, E)]
        video_embeddings = torch.cat(video_embeddings, dim=1)  # N K(T+1) E
