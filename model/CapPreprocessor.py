import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from transformers import AutoTokenizer


class CapPreprocessor:
    def __init__(self, tokenizer_type, device=torch.device('cuda')):
        self.tokenizer_type = tokenizer_type
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
        self.pad_id = self.tokenizer.convert_tokens_to_ids("[PAD]")
        self.start_id = self.tokenizer.convert_tokens_to_ids("[CLS]")
        self.end_id = self.tokenizer.convert_tokens_to_ids("[SEP]")

    def __call__(self, captions: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Turn raw text captions to tensor by Hugging Face tokenizer
        text -> id -> batching -> masking
        :param captions: list of raw caption strings.
        :return: batched text tensor and mask tensor (True for valid position).
        """
        # 1-pass
        batch_size = len(captions)
        tokens = []
        for i in range(batch_size):
            tokens.append(self.tokenizer.encode(captions[i], return_tensors="pt").squeeze().to(self.device))
        # 2-pass
        text_len = [len(i) for i in tokens]
        max_len = max(text_len)
        text_ts = torch.ones([batch_size, max_len], dtype=torch.long).to(self.device) * self.pad_id
        for i in range(batch_size):
            text_ts[i, :len(tokens[i])] = tokens[i]
        text_mask_ts = (text_ts == self.pad_id).to(self.device)
        return text_ts, text_mask_ts




