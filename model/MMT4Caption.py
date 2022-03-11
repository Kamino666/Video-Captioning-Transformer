import torch
import torch.nn as nn
from torch import Tensor

from .CapDecoder import CapDecoder
from .CapPreprocessor import CapPreprocessor
from .TextEncoder import TextEncoder
from .MMEncoder import MultiModalEncoder, SimpleSepEncoder, HMMEncoder
from .Matching import Matching

from typing import Dict, List, Tuple, Optional


# Multi-modal Multi-task Transformer for Captioning
class MMT4Caption(nn.Module):
    def __init__(self, model_config: dict, device=torch.device('cuda')):
        """
        Multi-modal Multi-task Transformer for Captioning
        Main model class of this repo.
        :param model_config: The "model" main field of the configuration file in json format
        :param device: torch.device
        """
        super().__init__()
        self.device = device
        self.model_config = model_config
        self.loss_beta = model_config['loss_beta']
        self.f_type = None

        self.cap_preprocessor = CapPreprocessor(model_config['tokenizer'], device=device)
        self.text_encoder = TextEncoder(model_config['text_enc_type'], device=device)
        self.cap_decoder = CapDecoder(
            num_layers=model_config['caption_decoder']['layer'],
            embed_dim=model_config['embed_dim'],
            nhead=model_config['caption_decoder']['nhead'],
            dim_feedforward=model_config['caption_decoder']['feedforward'],
            dropout=model_config['dropout'],
            vocab_size=self.cap_preprocessor.tokenizer.vocab_size,
            pad_id=self.cap_preprocessor.pad_id,
            sce_loss_alpha=model_config['caption_decoder']['sce_loss_alpha'],
            custom_decoder_type=model_config['caption_decoder'].get('layer_type', None),
            activation=model_config['activation'],
            device=device
        )
        vid_enc_type = model_config['video_encoder'].get('type', 'mme')
        if vid_enc_type == "simple":
            self.video_encoder = SimpleSepEncoder(
                d_feats=model_config['modal_shape'],
                d_model=model_config['embed_dim'],
                nhead=model_config['video_encoder']['nhead'],
                dim_feedforward=model_config['video_encoder']['feedforward'],
                num_encoder_layers=model_config['video_encoder']['layer'],
                dropout=model_config['dropout'],
                activation=model_config['activation'],
                device=device
            )
        elif vid_enc_type == 'hmme':
            self.video_encoder = HMMEncoder(
                d_feats=model_config['modal_shape'],
                d_model=model_config['embed_dim'],
                nhead=model_config['video_encoder']['nhead'],
                dim_feedforward=model_config['video_encoder']['feedforward'],
                num_encoder_layers=model_config['video_encoder']['layer'],
                dropout=model_config['dropout'],
                activation=model_config['activation'],
                global_type=model_config['video_encoder']['mme']['aggregation'],
                modal_different=model_config['video_encoder']['mme'].get('modal_different', True),
                temporal_type=model_config['video_encoder']['mme'].get('temporal', 'encoding'),
                do_norm=model_config['video_encoder']['mme'].get('do_norm', False),
                device=device
            )
        else:
            self.video_encoder = MultiModalEncoder(
                d_feats=model_config['modal_shape'],
                d_model=model_config['embed_dim'],
                nhead=model_config['video_encoder']['nhead'],
                dim_feedforward=model_config['video_encoder']['feedforward'],
                num_encoder_layers=model_config['video_encoder']['layer'],
                dropout=model_config['dropout'],
                activation=model_config['activation'],
                global_type=model_config['video_encoder']['mme']['aggregation'],
                modal_different=model_config['video_encoder']['mme'].get('modal_different', True),
                temporal_type=model_config['video_encoder']['mme'].get('temporal', 'encoding'),
                do_norm=model_config['video_encoder']['mme'].get('do_norm', False),
                device=device
            )
        if model_config.get('matching', None) is not None:
            self.matching = Matching((model_config['embed_dim'], self.text_encoder.dim),
                                     enable_tem=model_config['matching']['enable_tem'],
                                     loss=model_config['matching']['matching_loss'],
                                     loss_tem=model_config['matching'].get("temperature", None),
                                     device=device)

        # if model_config['pretrained_model'] is not None:
        #     self.load_state_dict(torch.load(model_config['pretrained_model'], map_location=self.device), strict=False)

    def forward(self, video_feats: List[Tensor], video_masks: List[Tensor], captions: List[str]):
        """
        Perform a forward pass based on f_type
        :param video_feats: length of this list is equal to the number of modal.
                            every element of the list has a shape like [B, T, E].(T varies)
        :param video_masks: same as video_feats, but shape is like [B, T]
        :param captions: list of raw caption strings.
        :return:
        """
        if self.f_type == "caption":
            return self.caption_forward(video_feats, video_masks, captions)
        elif self.f_type == "match":
            return self.match_forward(video_feats, video_masks, captions)
        elif self.f_type == "cross":
            return self.cross_forward(video_feats, video_masks, captions)
        else:
            raise ValueError

    def caption_forward(self, video_feats: List[Tensor], video_masks: List[Tensor], captions: List[str]):
        """
        Forward propagation of video captioning task.
        """
        text_ts, text_mask_ts = self.cap_preprocessor(captions)
        memory, _, _ = self.video_encoder(video_feats, video_masks)
        logits, loss = self.cap_decoder(memory, text_ts, text_mask_ts)
        return loss

    def match_forward(self, video_feats: List[Tensor], video_masks: List[Tensor], captions: List[str]):
        """
        Forward propagation of video-text match task.
        """
        text_feat = self.text_encoder(captions)
        _, _, agg_feat = self.video_encoder(video_feats, video_masks)
        loss = self.matching(text_feat, agg_feat)
        return loss

    def cross_forward(self, video_feats: List[Tensor], video_masks: List[Tensor], captions: List[str]):
        """
        Forward propagation of both task.
        """
        text_ts, text_mask_ts = self.cap_preprocessor(captions)
        text_feat = self.text_encoder(captions)
        memory, memory_masks, agg_feat = self.video_encoder(video_feats, video_masks)

        logits, cap_loss = self.cap_decoder(memory, text_ts, text_mask_ts)
        match_loss = self.matching(text_feat, agg_feat)

        loss = self.loss_beta * cap_loss + (1 - self.loss_beta) * match_loss
        return loss, cap_loss, match_loss

    def greedy_decode(self, video_feat: List[Tensor],
                      video_masks: Optional[List[Tensor]] = None,
                      max_len: int = 30) -> List[str]:
        """
        Use greedy algorithm for video captioning.
        :param video_feat: feature of different modal. The shape of every tensor is [B, T, E]
        :param video_masks: same as video_feats, but shape is like [B, T]
        :param max_len: The maximum length of a predicted caption
        :return:
        """
        batch_size = video_feat[0].shape[0]
        start_id = self.cap_preprocessor.start_id
        end_id = self.cap_preprocessor.end_id
        memory, _, _ = self.video_encoder(video_feat, video_masks)
        # predict
        ys = torch.ones(batch_size, 1).fill_(start_id).type(torch.long).to(self.device)  # N, 1
        end_flag = [0] * batch_size
        for i in range(max_len - 1):
            prob = self.cap_decoder.decode_word(memory, ys, None)
            _, next_word = torch.max(prob, dim=1)  # N
            ys = torch.cat([ys, next_word.unsqueeze(1).type(torch.long)], dim=1)  # N, t
            # break when all reach 'end_id'
            for k, flag in enumerate((next_word == end_id).tolist()):
                if flag is True:
                    end_flag[k] = 1
            if sum(end_flag) >= batch_size:
                break
        # to string
        result = []
        for idx_cap in ys.tolist():
            end_count = -1
            for i, idx in enumerate(idx_cap):
                if idx == end_id:
                    end_count = i
                    break
            idx_cap = idx_cap[1:end_count]
            token_cap = self.cap_preprocessor.tokenizer.convert_ids_to_tokens(idx_cap)
            result.append(self.cap_preprocessor.tokenizer.convert_tokens_to_string(token_cap))
        return result

    def beam_decode(self):
        pass

    def mode(self, forward_type="caption") -> None:
        """
        Change mode of model.
        :param forward_type: "caption", "match" or "cross"
        """
        self.f_type = forward_type
        if forward_type == "caption":
            for param in self.cap_decoder.parameters():
                param.requires_grad = True
            for param in self.matching.parameters():
                param.requires_grad = False
        elif forward_type == "match":
            for param in self.cap_decoder.parameters():
                param.requires_grad = False
            for param in self.matching.parameters():
                param.requires_grad = True
        elif forward_type == "cross":
            for param in self.cap_decoder.parameters():
                param.requires_grad = True
            for param in self.matching.parameters():
                param.requires_grad = True
        else:
            raise ValueError

    def load_embedding_from_bert(self):
        from transformers import BertModel
        bert = BertModel.from_pretrained("bert-base-uncased")
        for k, v in bert.named_parameters():
            if k == "embeddings.word_embeddings.weight":
                self.cap_decoder.tgt_to_emb.weight = v
            elif k == "embeddings.position_embeddings.weight":
                self.cap_decoder.positional_encoding.pos_embedding = v

    def load_cap_decoder_from_univl(self, path):
        univl: dict = torch.load(path)
        cap_params_dict: Dict[str, Tensor] = {}

        for l in range(3):
            for wb in ['weight', 'bias']:
                # self_attn | slf_attn
                cap_params_dict[f'decoder.layers.{l}.self_attn.in_proj_{wb}'] = torch.cat([
                    univl[f'decoder.decoder.layer.{l}.slf_attn.att.query.{wb}'],
                    univl[f'decoder.decoder.layer.{l}.slf_attn.att.key.{wb}'],
                    univl[f'decoder.decoder.layer.{l}.slf_attn.att.value.{wb}'],
                ], dim=0)  # torch.Size([2304, 768]) or torch.Size([2304])
                cap_params_dict[f'decoder.layers.{l}.self_attn.out_proj.{wb}'] = \
                    univl[f'decoder.decoder.layer.{l}.slf_attn.output.dense.{wb}']
                # torch.Size([768, 768]) or torch.Size([768])
                cap_params_dict[f'decoder.layers.{l}.norm1.{wb}'] = \
                    univl[f'decoder.decoder.layer.{l}.slf_attn.output.LayerNorm.{wb}']
                # torch.Size([768])

                # multihead_attn | enc_attn
                cap_params_dict[f'decoder.layers.{l}.multihead_attn.in_proj_{wb}'] = torch.cat([
                    univl[f'decoder.decoder.layer.{l}.enc_attn.att.query.{wb}'],
                    univl[f'decoder.decoder.layer.{l}.enc_attn.att.key.{wb}'],
                    univl[f'decoder.decoder.layer.{l}.enc_attn.att.value.{wb}'],
                ], dim=0)  # torch.Size([2304, 768]) or torch.Size([2304])
                cap_params_dict[f'decoder.layers.{l}.multihead_attn.out_proj.{wb}'] = \
                    univl[f'decoder.decoder.layer.{l}.enc_attn.output.dense.{wb}']
                # torch.Size([768, 768]) or torch.Size([768])
                cap_params_dict[f'decoder.layers.{l}.norm2.{wb}'] = \
                    univl[f'decoder.decoder.layer.{l}.enc_attn.output.LayerNorm.{wb}']
                # torch.Size([768])

                # linear1 | intermediate
                cap_params_dict[f'decoder.layers.{l}.linear1.{wb}'] = \
                    univl[f'decoder.decoder.layer.{l}.intermediate.dense.{wb}']
                # torch.Size([3072, 768]) or torch.Size([3072])

                # linear2 | output
                cap_params_dict[f'decoder.layers.{l}.linear2.{wb}'] = \
                    univl[f'decoder.decoder.layer.{l}.output.dense.{wb}']
                # torch.Size([768, 3072]) or torch.Size([768])

                # norm3 | output.LayerNorm
                cap_params_dict[f'decoder.layers.{l}.norm3.{wb}'] = \
                    univl[f'decoder.decoder.layer.{l}.output.LayerNorm.{wb}']
                # torch.Size([768])

        for wb in ['weight', 'bias']:
            # decoder.norm | decoder.embeddings.LayerNorm torch.Size([768])
            cap_params_dict[f'decoder.norm.{wb}'] = univl[f'decoder.embeddings.LayerNorm.{wb}']

        # generator | classifier.cls
        cap_params_dict[f'generator.weight'] = univl[f'decoder.classifier.cls.predictions.decoder.weight']
        cap_params_dict[f'generator.bias'] = univl[f'decoder.classifier.cls.predictions.bias']
        # torch.Size([30522, 768]) or torch.Size([30522])

        # tgt_to_emb | word_embeddings torch.Size([30522, 768])
        cap_params_dict[f'tgt_to_emb.weight'] = univl[f'decoder.embeddings.word_embeddings.weight']
        # positional_encoding | position_embeddings torch.Size([512, 768])
        cap_params_dict[f'positional_encoding.pos_embedding'] = univl[f'decoder.embeddings.position_embeddings.weight']

        self.cap_decoder.load_state_dict(cap_params_dict)


