import torch
import torch.nn as nn
from torch.nn import Transformer
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, AutoTokenizer

import math
import pathlib as plb
import random
import json
from tqdm import tqdm
import numpy as np
from timeit import default_timer as timer

device = torch.device("cuda")


class Opt:
    batch_size = 64
    bert_type = "bert-base-uncased"
    enc_layer_num = 8
    dec_layer_num = 8
    head_num = 8
    emb_dim = 768
    hid_dim = 1024
    dropout = 0.1
    lr = 1e-4
    epoch_num = 50


class MSRVTT(Dataset):
    def __init__(self, video_feat_dir: str, annotation_file: str, tokenizer,
                 random_seed: int = 1234, mode: str = "train"):
        super(MSRVTT, self).__init__()
        random.seed(random_seed)
        self.tokenizer = tokenizer
        # load video list
        video_feat_dir = plb.Path(video_feat_dir)
        self.video_feat_list = list(video_feat_dir.glob("*.npy"))
        self.mode = mode

        # load caption
        if mode == "train" or "validate":
            self.video2caption = {}
            with open(annotation_file, encoding='utf-8') as f:
                annotation = json.load(f)
            self.video2split = {i["video_id"]: i["split"] for i in annotation["videos"]}
            for cap in tqdm(annotation["sentences"], desc="Loading annotations"):
                if self.video2split[cap["video_id"]] != mode:
                    continue
                if cap["video_id"] not in self.video2caption:
                    self.video2caption[cap["video_id"]] = [cap["caption"]]
                else:
                    self.video2caption[cap["video_id"]].append(cap["caption"])

    def __getitem__(self, index):
        video_path = self.video_feat_list[index]
        vid = video_path.stem
        v_feat = torch.tensor(np.load(str(video_path)), dtype=torch.float).transpose(0, 1)
        if self.mode == "train" or "val" or "validate":
            caption = random.choice(self.video2caption[vid])
            caption = self.tokenizer.encode(caption, return_tensors="pt").squeeze()
            return v_feat, caption #caption["input_ids"], caption["attention_mask"]
        return v_feat

    def __len__(self):
        return len(self.video_feat_list)


def collate_fn(data):
    """
    :param data:
    :return tuple(N T E, N T):
    """
    if type(data[0]) is tuple:
        batch_size = len(data)
        feat_ts = torch.stack([i[0] for i in data])

        text_data = [i[1] for i in data]
        text_len = [len(i) for i in text_data]
        max_len = max(text_len)

        text_ts = torch.ones([batch_size, max_len], dtype=torch.long) * opt.pad_id
        for i in range(batch_size):
            text_ts[i, :text_len[i]] = text_data[i]

        mask_ts = (text_ts == opt.pad_id)

        return feat_ts, text_ts, mask_ts
    else:
        return torch.stack(data)


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        #pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        # 广播机制加到每一个batch上
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.shape[1], :])


# Seq2Seq Network
# [N,T,E] and [N,S,E] -> [N, len, vocab_size]
class VideoTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 bert_type: str,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(VideoTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       batch_first=True)
        self.tokenizer = AutoTokenizer.from_pretrained("./data/tk/")
        self.generator = nn.Linear(emb_size, self.tokenizer.vocab_size)
        self.tgt_tok_emb = BertModel.from_pretrained(bert_type)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor = None):
        src_emb = self.positional_encoding(src)  # src: torch.Size([16, 768, 20])
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt).last_hidden_state.to(device))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(src), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)),
            memory,
            tgt_mask)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def train_epoch(model, optimizer, train_dataloader):
    model.train()
    losses = 0

    for src, tgt, tgt_padding_mask in tqdm(train_dataloader):
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_padding_mask = tgt_padding_mask.to(device)[:, :-1]

        tgt_input = tgt[:, :-1]  # N T-1
        tgt_mask = generate_square_subsequent_mask(tgt_input.shape[1])

        logits = model(src, tgt_input,
                       tgt_mask=tgt_mask, tgt_padding_mask=tgt_padding_mask,
                       src_mask=None, src_padding_mask=None)  # N T-1 vocab_szie

        optimizer.zero_grad()

        tgt_out = tgt[:, 1:]  # N T-1
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)


def evaluate(model, val_dataloader):
    model.eval()
    losses = 0

    for src, tgt, tgt_padding_mask in val_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_padding_mask = tgt_padding_mask.to(device)[:, :-1]

        tgt_input = tgt[:, :-1]  # N T-1
        tgt_mask = generate_square_subsequent_mask(tgt_input.shape[1])
        with torch.no_grad():
            logits = model(src, tgt_input,
                           tgt_mask=tgt_mask, tgt_padding_mask=tgt_padding_mask,
                           src_mask=None, src_padding_mask=None)  # N T-1 vocab_szie

            tgt_out = tgt[:, 1:]  # N T-1
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)


if __name__ == "__main__":
    opt = Opt()
    transformer = VideoTransformer(num_encoder_layers=opt.enc_layer_num,
                                   num_decoder_layers=opt.dec_layer_num,
                                   emb_size=opt.emb_dim,
                                   nhead=opt.head_num,
                                   bert_type=opt.bert_type,
                                   dropout=opt.dropout,
                                   dim_feedforward=opt.hid_dim)
    tokenizer = AutoTokenizer.from_pretrained("./data/tk/")
    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
    opt.pad_id = pad_id

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=opt.lr)
    
    train_iter = MSRVTT(r"./data/msrvtt-train-feats",
                        r"./data/MSRVTT-annotations/train_val_videodatainfo.json",
                        tokenizer=tokenizer)
    train_dataloader = DataLoader(train_iter, batch_size=opt.batch_size, collate_fn=collate_fn, shuffle=True)
    val_iter = MSRVTT(r"./data/msrvtt-validate-feats",
                      r"./data/MSRVTT-annotations/train_val_videodatainfo.json",
                      tokenizer=tokenizer, mode="validate")
    val_dataloader = DataLoader(val_iter, batch_size=opt.batch_size, collate_fn=collate_fn)

    for epoch in range(1, opt.epoch_num + 1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer, train_dataloader)
        end_time = timer()
        val_loss = evaluate(transformer, val_dataloader)
        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f},"
              f" Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s")
