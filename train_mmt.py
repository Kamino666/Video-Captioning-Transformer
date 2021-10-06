import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoTokenizer
from visualdl import LogWriter

from dataloader import MultiModalMSRVTT
from utils import generate_square_subsequent_mask
from model.model import MMVideoTransformer

from tqdm import tqdm
from timeit import default_timer as timer
import re
from utils import EarlyStopping
import os

# device = torch.device("cuda")
local_rank = int(os.environ['LOCAL_RANK'])  # int 0/1/2/3
# 新增：DDP backend初始化
dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端
# 构造模型
device = torch.device("cuda", local_rank)


class Opt:
    # data
    train_video_dirs = {"resnet152": "data/msrvtt_resnet152_fps3_feats/train",
                        "CLIP": "data/MSRVTT-CLIP-FEATURES/train_feats",
                        "swin": "data/msrvtt_swin_base/train"}
    train_annotation_path = r"./data/MSRVTT-annotations/train_val_videodatainfo.json"
    val_video_dirs = {"resnet152": "data/msrvtt_resnet152_fps3_feats/val",
                      "CLIP": "data/MSRVTT-CLIP-FEATURES/val_feats",
                      "swin": "data/msrvtt_swin_base/val"}
    val_annotation_path = r"./data/MSRVTT-annotations/train_val_videodatainfo.json"
    raw_video_dir = None  # r"data/MSRVTT_trainval"
    # train
    batch_size = 8
    lr = 1e-4
    max_len = 30
    learning_rate_patience = 5
    early_stopping_patience = 10
    # model
    bert_type = "bert-base-uncased"
    enc_layer_num = 4
    dec_layer_num = 4
    head_num = 8
    feat_dims = {"resnet152": 2048, "CLIP": 512, "swin": 768}
    d_model = 512
    dim_feedforward = 2048
    dropout = 0.3
    agg_method = "avg"
    epoch_num = 30
    # save & load
    save_freq = 50
    load_model = None
    model_save_dir = "./checkpoint"
    log_subdir = "MMT"
    _extra_msg = "MSRVTT&3feat"  # Dataset|Bert|pretrained
    training_name = f"b{batch_size}_lr{str(lr)[2:]}_dp{str(dropout).replace('.', '')}_dmodel{d_model}_e{enc_layer_num}" \
                    f"_d{dec_layer_num}_dhead{head_num}_dfeed{dim_feedforward}_{_extra_msg}"


def train_epoch(model, optimizer, train_dataloader):
    model.train()
    losses = 0

    for feats_dict, text_ts, feat_mask_dict, text_mask_ts in tqdm(train_dataloader):
        tgt = text_ts.to(device)
        tgt_padding_mask = text_mask_ts.to(device)[:, :-1]
        feats_dict = {k: v.to(device) for k, v in feats_dict.items()}
        feat_mask_dict = {k: v.to(device) for k, v in feat_mask_dict.items()}

        tgt_input = tgt[:, :-1]  # N T-1
        tgt_mask = generate_square_subsequent_mask(tgt_input.shape[1]).to(device)
        
        
        logits = model(feats_dict=feats_dict, feats_padding_mask_dict=feat_mask_dict,
                       tgt=tgt_input, tgt_mask=tgt_mask, tgt_padding_mask=tgt_padding_mask,)  # N T-1 vocab_size

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

    for feats_dict, text_ts, feat_mask_dict, text_mask_ts in tqdm(train_dataloader):
        tgt = text_ts.to(device)
        tgt_padding_mask = text_mask_ts.to(device)[:, :-1]
        feats_dict = {k: v.to(device) for k, v in feats_dict.items()}
        feat_mask_dict = {k: v.to(device) for k, v in feat_mask_dict.items()}

        tgt_input = tgt[:, :-1]  # N T-1
        tgt_mask = generate_square_subsequent_mask(tgt_input.shape[1]).to(device)
        with torch.no_grad():
            logits = model(feats_dict=feats_dict, feats_padding_mask_dict=feat_mask_dict,
                           tgt=tgt_input, tgt_mask=tgt_mask, tgt_padding_mask=tgt_padding_mask, )  # N T-1 vocab_size

            tgt_out = tgt[:, 1:]  # N T-1
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, max_len, start_symbol, end_symbol):
    src = src.to(device).unsqueeze(0).transpose(1, 2)  # 1, T, e
    memory = model.encode(src).to(device)  # 1, T, E

    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)  # 1, 1
    for i in range(max_len - 1):
        tgt_mask = (generate_square_subsequent_mask(ys.shape[1]).type(torch.bool)).to(device)  # t, t
        out = model.decode(ys, memory, tgt_mask)
        prob = model.generator(out[:, -1])  # vocab_size
        prob = F.softmax(prob)
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type(torch.long).fill_(next_word).to(device)], dim=1)  # 1, t
        if next_word == end_symbol:
            break
    return ys.squeeze().tolist()


def translate(model, src, max_len, start_symbol, end_symbol, tokenizer):
    src = src.to(device)
    # inference
    result = greedy_decode(model, src, max_len, start_symbol, end_symbol)  # list T
    # to text
    result_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(result))
    result_text = result_text.replace("[CLS]", "")
    result_text = result_text.replace("[SEP]", "")
    return result_text


def collate_fn(data):
    """
    :param data: List of (feats_dict, caption, vid)
    :return dict(B T E)
    """
    batch_size = len(data)
    feat_dicts, captions, vids = zip(*data)

    # video feature
    # 先获取信息
    feat_names = set()  # {"resnet152", "CLIP"}
    feat_len_dict = {}  # {"resnet152": [5, 4, 6], "CLIP": [6, 3, 6]}
    feat_dim_dict = {}  # {"resnet152": 2048, "CLIP": 512}
    feat_unpad_dict = {}  # {"resnet152": [Tensor, Tensor], "CLIP": [Tensor, Tensor]}
    for feat_dict in feat_dicts:
        for name, ts in feat_dict.items():
            feat_names.add(name)
            if name in feat_len_dict:
                feat_len_dict[name].append(ts.shape[0])
            else:
                feat_len_dict[name] = [ts.shape[0]]
            feat_dim_dict[name] = ts.shape[1]
            if name in feat_unpad_dict:
                feat_unpad_dict[name].append(ts)
            else:
                feat_unpad_dict[name] = [ts]
    feat_maxlen_dict = {k: max(v) for k, v in feat_len_dict.items()}  # {"resnet152": 6, "CLIP": 6}
    # 再pad
    feat_padded_dict = {}  # {"resnet152": [Tensor, Tensor], "CLIP": [Tensor, Tensor]}
    feat_mask_dict = {}  # {"resnet152": [Tensor, Tensor], "CLIP": [Tensor, Tensor]}
    for feat_name in feat_names:
        feat_ts = torch.zeros([batch_size, feat_maxlen_dict[feat_name], feat_dim_dict[feat_name]], dtype=torch.float)
        feat_mask_ts = torch.ones([batch_size, feat_maxlen_dict[feat_name]], dtype=torch.long)
        for i in range(batch_size):
            feat_ts[i, :feat_len_dict[feat_name][i]] = feat_unpad_dict[feat_name][i]
            feat_mask_ts[i, :feat_len_dict[feat_name][i]] = 0
        feat_mask_ts = (feat_mask_ts == 1)
        # 填充进dict
        feat_padded_dict[feat_name] = feat_ts
        feat_mask_dict[feat_name] = feat_mask_ts

    # text
    text_len = [len(i) for i in captions]
    max_len = max(text_len)
    text_ts = torch.ones([batch_size, max_len], dtype=torch.long) * opt.pad_id
    for i in range(batch_size):
        text_ts[i, :text_len[i]] = captions[i]
    text_mask_ts = (text_ts == opt.pad_id)
    return feat_padded_dict, text_ts, feat_mask_dict, text_mask_ts


if __name__ == "__main__":
    opt = Opt()
    transformer = MMVideoTransformer(num_encoder_layers=opt.enc_layer_num,
                                     num_decoder_layers=opt.dec_layer_num,
                                     feat_dims=opt.feat_dims,
                                     d_model=opt.d_model,
                                     nhead=opt.head_num,
                                     bert_type=opt.bert_type,
                                     dropout=opt.dropout,
                                     dim_feedforward=opt.dim_feedforward,
                                     agg_method=opt.agg_method,
                                     device=device)
    if opt.load_model is None:
        st_epoch = 0
    else:
        transformer.load_state_dict(torch.load(opt.load_model))
        st_epoch = int(re.findall("epoch([0-9]+)", opt.load_model)[0])

    transformer = transformer.to(device)
    # multi GPU (DDP)
    transformer = DDP(transformer, device_ids=[local_rank], output_device=local_rank)

    # transformer.freeze_bert()
    tokenizer = AutoTokenizer.from_pretrained("./data/tk/")
    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
    start_id = tokenizer.convert_tokens_to_ids("[CLS]")
    end_id = tokenizer.convert_tokens_to_ids("[SEP]")
    opt.pad_id = pad_id

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, transformer.parameters()), lr=opt.lr)
    # dynamic learning rate
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, verbose=True, patience=opt.learning_rate_patience
    )
    # early stop
    early_stopping = EarlyStopping(patience=opt.early_stopping_patience,
                                   verbose=True,
                                   path=os.path.join(opt.model_save_dir, f"{opt.training_name}_earlystop.pth"))
    # visulize & log
    writer = LogWriter(f"./log/{opt.log_subdir}/{opt.training_name}")

    # dataloader
    train_iter = MultiModalMSRVTT(opt.train_video_dirs, opt.train_annotation_path, tokenizer=tokenizer)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_iter, shuffle=True)
    train_dataloader = DataLoader(train_iter, batch_size=opt.batch_size, collate_fn=collate_fn,
                                  sampler=train_sampler)
    val_iter = MultiModalMSRVTT(opt.val_video_dirs, opt.val_annotation_path, tokenizer=tokenizer, mode="validate")
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_iter)
    val_dataloader = DataLoader(val_iter, batch_size=opt.batch_size, collate_fn=collate_fn, sampler=val_sampler)

    # train
    for epoch in range(1 + st_epoch, opt.epoch_num + 1 + st_epoch):
        train_dataloader.sampler.set_epoch(epoch)
        val_dataloader.sampler.set_epoch(epoch)

        start_time = timer()
        train_loss = train_epoch(transformer, optimizer, train_dataloader)
        end_time = timer()

        val_loss = evaluate(transformer, val_dataloader)

        # sample = val_iter.get_a_sample(ori_video_dir=opt.raw_video_dir)
        # result_text = translate(transformer, sample["v_feat"], opt.max_len, start_id, end_id, tokenizer)
        # sample_text = sample["raw_caption"]

        # logging
        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f},"
              f" Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s")
        # print(f"target:{sample_text} \nresult:{result_text}")
        writer.add_scalar("train_loss", train_loss, step=epoch)
        writer.add_scalar("val_loss", val_loss, step=epoch)
        writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], step=epoch)
        # writer.add_text("text/sample", sample_text, step=epoch)
        # writer.add_text("text/result", result_text, step=epoch)
        lr_scheduler.step(val_loss)

        if dist.get_rank() == 0:
            # early stopping
            early_stopping(val_loss, transformer.module)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if epoch % opt.save_freq == 0:
                print("Saving checkpoint...")
                torch.save(transformer.module.state_dict(),
                           os.path.join(opt.model_save_dir, f"{opt.training_name}_epoch{epoch}.pth"))
