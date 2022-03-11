import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional, Tuple


class TextEncoder:
    def __init__(self, text_enc_type: str, device=torch.device('cuda')):
        self.text_enc_type = text_enc_type
        self.device = device

        if "CLIP" in text_enc_type:
            import clip
            self.text_enc, _ = clip.load("ViT-B/32", device=device)
            self.text_enc.eval()
            self.dim = 512
        elif "bert" in text_enc_type:
            from transformers import AutoTokenizer, BertModel
            # self.tokenizer = AutoTokenizer.from_pretrained(text_enc_type)
            self.tokenizer = AutoTokenizer.from_pretrained("./data/tk")
            self.text_enc = BertModel.from_pretrained(text_enc_type).to(self.device)
            self.dim = 768
        else:
            raise ValueError

    def __call__(self, captions: List[str]) -> Tensor:
        """
        Use Clip or Bert to encode captions, the result is a n-dim tensor for each caption
        :param captions: list of raw caption strings.
        :return:
        """
        if "CLIP" in self.text_enc_type:
            import clip
            tokens = clip.tokenize(captions).to(self.device)
            text_feat: Tensor = self.text_enc.encode_text(tokens)
            text_feat = text_feat.to(torch.float32).detach()
        elif "bert" in self.text_enc_type:
            pad_id = self.tokenizer.convert_tokens_to_ids("[PAD]")
            # 1-pass
            batch_size = len(captions)
            tokens = []
            for i in range(batch_size):
                tokens.append(self.tokenizer.encode(captions[i], return_tensors="pt").squeeze())
            # 2-pass
            text_len = [len(i) for i in tokens]
            max_len = max(text_len)
            text_ts = torch.ones([batch_size, max_len], dtype=torch.long).to(self.device) * pad_id
            for i in range(batch_size):
                text_ts[i, :len(tokens[i])] = tokens[i]
            text_mask_ts = (text_ts == pad_id).to(self.device)

            text_feat = self.text_enc(text_ts, text_mask_ts).last_hidden_state[:, 0]  # B, 1
        else:
            raise ValueError
        return text_feat
