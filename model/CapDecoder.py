import torch
import torch.nn as nn
from torch import Tensor
from .Embedding import PositionalEmbedding
from .loss import SCELoss

from utils import generate_square_subsequent_mask
from typing import List, Tuple, Optional


class CapDecoder(nn.Module):
    def __init__(self, num_layers, embed_dim, nhead, dim_feedforward, dropout,
                 vocab_size, pad_id, sce_loss_alpha: float, custom_decoder_type: Optional[str] = None,
                 activation='gelu', device=torch.device('cuda')):
        super().__init__()
        self.device = device
        if custom_decoder_type is None:
            decoder_layer = nn.TransformerDecoderLayer(embed_dim, nhead, dim_feedforward, dropout,
                                                       activation=activation, batch_first=True)
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers, nn.LayerNorm(embed_dim))
        else:
            decoder_layer = VisTransformerDecoderLayer(embed_dim, nhead, dim_feedforward, dropout,
                                                       activation=activation, batch_first=True)
            self.decoder = VisTransformerDecoder(decoder_layer, num_layers, nn.LayerNorm(embed_dim))
        self.generator = nn.Linear(embed_dim, vocab_size)
        self.tgt_to_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.positional_encoding = PositionalEmbedding(embed_dim, dropout=dropout, maxlen=5000)
        if sce_loss_alpha == 1.0:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id).to(device)
        else:
            self.loss_fn = SCELoss(sce_loss_alpha, 1 - sce_loss_alpha, ignore_index=pad_id, num_classes=vocab_size,
                                   device=device)

    def forward(self, memories: Tensor, tgt: Tensor, tgt_padding_mask: Tensor):
        """
        Use known video features and ground-truth captions to calculate loss
        :param memories: [B, T, E]
        :param tgt: idx of caption [B, S]
        :param tgt_padding_mask: [B, S]
        :return: logits, loss
        """
        # prepare
        tgt_input = tgt[:, :-1]  # N T-1
        tgt_out = tgt[:, 1:]  # N T-1
        tgt_input_padding_mask = tgt_padding_mask[:, :-1]
        tgt_mask = generate_square_subsequent_mask(tgt_input.shape[1]).to(self.device)

        tgt_emb = self.positional_encoding(self.tgt_to_emb(tgt_input))
        outs = self.decoder(
            tgt_emb, memories,
            tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_input_padding_mask,
        )
        if type(outs) is tuple:
            outs, self.attn_weights = outs
        logits = self.generator(outs)
        loss = self.loss_fn(
            logits.reshape(-1, logits.shape[-1]),
            tgt_out.reshape(-1)
        )
        return logits, loss

    def decode_word(self, memories: Tensor, tgt: Tensor, tgt_padding_mask: Optional[Tensor]):
        """
        Use known video features and text to predict the next word
        :param memories: [B, T, E]
        :param tgt: idx of caption [B, S]
        :param tgt_padding_mask: [B, S]
        :return:
        """
        tgt_emb = self.positional_encoding(self.tgt_to_emb(tgt))
        tgt_mask = generate_square_subsequent_mask(tgt_emb.shape[1]).to(torch.bool).to(self.device)
        outs = self.decoder(
            tgt_emb, memories,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        logits = self.generator(outs[:, -1])
        # _, word_idx = torch.max(logits, dim=1)  # N
        return logits


# code is modified from PyTorch
class VisTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False, device=None, dtype=None) -> None:
        super(VisTransformerDecoderLayer, self).__init__(d_model, nhead, dim_feedforward=dim_feedforward,
                                                         dropout=dropout,
                                                         activation=activation, layer_norm_eps=layer_norm_eps,
                                                         batch_first=batch_first, device=device, dtype=dtype)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
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
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn_output_weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                                        key_padding_mask=memory_key_padding_mask)
        # B, tgt_length, memory_length
        # print(attn_output_weights.shape)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn_output_weights


class VisTransformerDecoder(nn.TransformerDecoder):
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, List]:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

                Args:
                    tgt: the sequence to the decoder (required).
                    memory: the sequence from the last layer of the encoder (required).
                    tgt_mask: the mask for the tgt sequence (optional).
                    memory_mask: the mask for the memory sequence (optional).
                    tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
                    memory_key_padding_mask: the mask for the memory keys per batch (optional).

                Shape:
                    see the docs in Transformer class.
                """
        output = tgt

        attn_output_weights = []
        for mod in self.layers:
            output, attn = mod(output, memory, tgt_mask=tgt_mask,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask)
            attn_output_weights.append(attn)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_output_weights
