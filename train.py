import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, BertModel
from tensorboardX import SummaryWriter

from dataloader import MSRVTT, build_collate_fn
from utils import generate_square_subsequent_mask
from utils import SCELoss
from model.model import VideoTransformer
from eval_batch_video import coco_eval

from tqdm import tqdm
from timeit import default_timer as timer
import re
from utils import EarlyStopping
import os

device = torch.device("cuda")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Opt:
    # data
    train_feat_dir = r"/data3/lzh_3/video-captioning-swin-transformer/data/msrvtt_CLIP_fps3_feats/train"
    train_annotation_path = r"/data3/lzh_3/video-captioning-swin-transformer/data/MSRVTT-annotations-clean/train_val_videodatainfo.json"
    val_feat_dir = r"/data3/lzh_3/video-captioning-swin-transformer/data/msrvtt_CLIP_fps3_feats/val"
    val_annotation_path = r"/data3/lzh_3/video-captioning-swin-transformer/data/MSRVTT-annotations-clean/train_val_videodatainfo.json"
    raw_video_dir = None
    # train
    batch_size = 64
    lr = 1e-4
    max_len = 30
    observe_metric = False
    # learning_rate_patience = 5
    early_stopping_patience = 10
    # model
    bert_type = "bert-base-uncased"
    enc_layer_num = 4
    dec_layer_num = 4
    head_num = 8
    feat_size = 512
    emb_dim = 768
    hid_dim = 2048
    dropout = 0.3
    epoch_num = 30
    use_bert = False
    # save & load
    save_freq = 10
    load_model = None
    model_save_dir = "./checkpoint"
    _extra_msg = "MSRVTT&CLIP3&SCE_loss"  # Dataset|Bert|pretrained
    training_name = f"b{batch_size}_lr{str(lr)[2:]}_dp{str(dropout).replace('.', '')}_emb{emb_dim}_e{enc_layer_num}" \
                    f"_d{dec_layer_num}_hd{head_num}_hi{hid_dim}_{_extra_msg}"
    log_subdir = training_name


def train_epoch(model, optimizer, train_dataloader):
    model.train()
    losses = 0

    for src, tgt, src_padding_mask, tgt_padding_mask in tqdm(train_dataloader):
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_padding_mask = tgt_padding_mask.to(device)[:, :-1]
        src_padding_mask = src_padding_mask.to(device)

        tgt_input = tgt[:, :-1]  # N T-1
        tgt_mask = generate_square_subsequent_mask(tgt_input.shape[1]).to(device)

        logits = model(src, tgt_input,
                       tgt_mask=tgt_mask, tgt_padding_mask=tgt_padding_mask,
                       src_mask=None, src_padding_mask=src_padding_mask)  # N T-1 vocab_szie

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

    for src, tgt, src_padding_mask, tgt_padding_mask in val_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_padding_mask = tgt_padding_mask.to(device)[:, :-1]
        src_padding_mask = src_padding_mask.to(device)

        tgt_input = tgt[:, :-1]  # N T-1
        tgt_mask = generate_square_subsequent_mask(tgt_input.shape[1]).to(device)
        with torch.no_grad():
            logits = model(src, tgt_input,
                           tgt_mask=tgt_mask, tgt_padding_mask=tgt_padding_mask,
                           src_mask=None, src_padding_mask=src_padding_mask)  # N T-1 vocab_szie

            tgt_out = tgt[:, 1:]  # N T-1
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)


def eval_by_metric(model, val_dataloader, val_iter, opt, tokenizer):
    """
    {'Bleu_1': 0.7350285171100914, 'Bleu_2': 0.5488721869312329,
     'Bleu_3': 0.39126032508403474, 'Bleu_4': 0.26597046542549374,
      'METEOR': 0.24072781402828644, 'ROUGE_L': 0.529469363492036,
       'CIDEr': 0.2837123774801633}
    """
    scorer = coco_eval(model, val_dataloader, val_iter, opt, tokenizer, verbose=False)
    score = {k: round(v, 4) for k, v in scorer.eval.items()}
    return score


# def train_epoch_gda(model, optimizer, train_dataloader):
#     assert train_dataloader.dataset.return_all_captions is True
#     model.train()
#     losses = 0
#
#     for src, tgts, src_padding_mask, tgt_padding_masks in tqdm(train_dataloader):
#         optimizer.zero_grad()
#         src = src.to(device)
#         src_padding_mask = src_padding_mask.to(device)
#         tgts = [tgt.to(device) for tgt in tgts]
#         tgt_padding_masks = [tgt_padding_mask.to(device)[:, :-1] for tgt_padding_mask in tgt_padding_masks]
#
#         tgt_inputs = [tgt[:, :-1] for tgt in tgts]  # N T-1
#         tgt_masks = [generate_square_subsequent_mask(tgt_input.shape[1]).to(device)
#                      for tgt_input in tgt_inputs]
#
#         logits_list = model(src, tgt_inputs,
#                             tgt_mask=tgt_masks, tgt_padding_mask=tgt_padding_masks,
#                             src_mask=None, src_padding_mask=src_padding_mask)  # N T-1 vocab_szie
#
#         tgt_outs = [tgt[:, 1:] for tgt in tgts]  # N T-1
#
#         # 计算loss
#         local_losses = 0
#         for logit, tgt_out in zip(logits_list, tgt_outs):
#             loss = loss_fn(logit.reshape(-1, logit.shape[-1]), tgt_out.reshape(-1))
#             loss.backward(retain_graph=True)
#             local_losses += loss.item()
#
#         optimizer.step()
#         losses += (local_losses / len(tgt_outs))
#
#     return losses / len(train_dataloader)


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, max_len, start_symbol, end_symbol):
    src = src.to(device).unsqueeze(0).transpose(1, 2)  # 1, T, e
    memory = model.encode(src).to(device)  # 1, T, E

    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)  # 1, 1
    for i in range(max_len - 1):
        tgt_mask = (generate_square_subsequent_mask(ys.shape[1]).type(torch.bool)).to(device)  # t, t
        out = model.decode(ys, memory, tgt_mask)
        prob = model.generator(out[:, -1])  # vocab_size
        prob = F.softmax(prob, dim=-1)
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


def get_embedding_from_bert(bert_type="bert-base-uncased"):
    print("loading BERT: {}".format(bert_type))
    bert = BertModel.from_pretrained(bert_type)
    for k, v in bert.named_parameters():
        if k == "embeddings.word_embeddings.weight":
            return v


# def collate_fn(data):
#     """
#     :param data:
#     :return tuple(N T E, N T):
#     """
#     batch_size = len(data)
#
#     # video feature
#     feat_dim = data[0][0].shape[1]
#     feat_data = [i[0] for i in data]
#     feat_len = [len(i) for i in feat_data]
#     max_len = max(feat_len)
#     feat_ts = torch.zeros([batch_size, max_len, feat_dim], dtype=torch.float)
#     feat_mask_ts = torch.ones([batch_size, max_len], dtype=torch.long)
#     for i in range(batch_size):
#         feat_ts[i, :feat_len[i]] = feat_data[i]
#         feat_mask_ts[i, :feat_len[i]] = 0
#     feat_mask_ts = (feat_mask_ts == 1)
#
#     # text
#     text_data = [i[1] for i in data]
#     text_len = [len(i) for i in text_data]
#     max_len = max(text_len)
#     text_ts = torch.ones([batch_size, max_len], dtype=torch.long) * opt.pad_id
#     for i in range(batch_size):
#         text_ts[i, :text_len[i]] = text_data[i]
#     text_mask_ts = (text_ts == opt.pad_id)
#     return feat_ts, text_ts, feat_mask_ts, text_mask_ts
#
#
# def multi_cap_collate_fn(data):
#     batch_size = len(data)
#
#     # video feature
#     feat_dim = data[0][0].shape[1]
#     feat_data = [i[0] for i in data]
#     feat_len = [len(i) for i in feat_data]
#     max_len = max(feat_len)
#     feat_ts = torch.zeros([batch_size, max_len, feat_dim], dtype=torch.float)
#     feat_mask_ts = torch.ones([batch_size, max_len], dtype=torch.long)
#     for i in range(batch_size):
#         feat_ts[i, :feat_len[i]] = feat_data[i]
#         feat_mask_ts[i, :feat_len[i]] = 0
#     feat_mask_ts = (feat_mask_ts == 1)
#
#     # text
#     text_ts_list = []
#     text_mask_ts_list = []
#     for data_of_a_video in data:
#         captions_of_a_video = data_of_a_video[1]
#         lengths = [len(i) for i in captions_of_a_video]
#         max_len = max(lengths)
#         text_ts = torch.ones([batch_size, max_len], dtype=torch.long) * opt.pad_id
#         for i in range(batch_size):
#             text_ts[i, :lengths[i]] = captions_of_a_video[i]
#         text_mask_ts = (text_ts == opt.pad_id)
#         text_ts_list.append(text_ts)
#         text_mask_ts_list.append(text_mask_ts)
#     return feat_ts, text_ts_list, feat_mask_ts, text_mask_ts_list
#
#

if __name__ == "__main__":
    opt = Opt()
    transformer = VideoTransformer(num_encoder_layers=opt.enc_layer_num,
                                   num_decoder_layers=opt.dec_layer_num,
                                   feat_size=opt.feat_size,
                                   emb_size=opt.emb_dim,
                                   nhead=opt.head_num,
                                   bert_type=opt.bert_type,
                                   dropout=opt.dropout,
                                   use_bert=opt.use_bert,
                                   dim_feedforward=opt.hid_dim,
                                   device=device)
    if opt.load_model is None:
        st_epoch = 0
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    else:
        transformer.load_state_dict(torch.load(opt.load_model))
        st_epoch = int(re.findall("epoch([0-9]+)", opt.load_model)[0])

    # transformer.load_embedding_weights(get_embedding_from_bert("bert-base-uncased"))
    transformer = transformer.to(device)
    # transformer.freeze_bert()

    tokenizer = AutoTokenizer.from_pretrained("/data3/lzh_3/video-captioning-swin-transformer/data/tk/")
    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
    start_id = tokenizer.convert_tokens_to_ids("[CLS]")
    end_id = tokenizer.convert_tokens_to_ids("[SEP]")
    opt.pad_id = pad_id
    opt.start_id = start_id
    opt.end_id = end_id

    # loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_id)
    # loss_fn = SCELoss(0.5, 0.5, pad_id, tokenizer.vocab_size)
    loss_fn = SCELoss(0.5, 0.5, pad_id, tokenizer.vocab_size)

    optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, transformer.parameters()), lr=opt.lr)

    # dynamic learning rate
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, verbose=True, patience=opt.learning_rate_patience
    # )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=8, eta_min=1e-5, verbose=True
    )

    # early stop
    early_stopping = EarlyStopping(patience=opt.early_stopping_patience,
                                   verbose=True,
                                   path=os.path.join(opt.model_save_dir, f"{opt.training_name}_earlystop.pth"))
    # visulize & log
    writer = SummaryWriter(os.path.join("./log", opt.log_subdir))

    # dataloader
    train_iter = MSRVTT(opt.train_feat_dir, opt.train_annotation_path, tokenizer=tokenizer)
    train_dataloader = DataLoader(train_iter, batch_size=opt.batch_size, collate_fn=build_collate_fn(opt.pad_id, False),
                                  shuffle=True)
    val_iter = MSRVTT(opt.val_feat_dir, opt.val_annotation_path, tokenizer=tokenizer, mode="validate", include_id=True)
    val_dataloader = DataLoader(val_iter, batch_size=opt.batch_size, collate_fn=build_collate_fn(opt.pad_id, False))
    if opt.observe_metric is True:
        metric_val_dataloader = DataLoader(val_iter, batch_size=opt.batch_size,
                                           collate_fn=build_collate_fn(opt.pad_id, True))

    # train
    for epoch in range(1 + st_epoch, opt.epoch_num + 1 + st_epoch):
        # 训练一个epoch
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer, train_dataloader)
        end_time = timer()
        # 验证一个epoch
        val_loss = evaluate(transformer, val_dataloader)
        # 验证METEOR指标
        if opt.observe_metric is True:
            score_dict = eval_by_metric(transformer, metric_val_dataloader, val_iter, opt, tokenizer)
        # 取样检查效果
        sample = val_iter.get_a_sample(ori_video_dir=opt.raw_video_dir)
        result_text = sample['v_id'] + " " + translate(transformer, sample["v_feat"], opt.max_len, start_id, end_id,
                                                       tokenizer)
        sample_text = sample['v_id'] + " " + sample["raw_caption"]
        # lr_scheduler.step(val_loss)
        lr_scheduler.step()

        # logging
        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f},"
              f" Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s")
        print(f"target:{sample_text} \nresult:{result_text}")
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("val_loss", val_loss, epoch)
        writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
        writer.add_text("text/sample", sample_text, epoch)
        writer.add_text("text/result", result_text, epoch)
        if opt.observe_metric is True:
            print(f"METEOR:{score_dict['METEOR']} B@4:{score_dict['Bleu_4']}")
            writer.add_scalar("METEOR", score_dict['METEOR'], epoch)
            writer.add_scalar("Bleu@4", score_dict['Bleu_4'], epoch)

        # early stopping
        early_stopping(val_loss, transformer)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        if epoch % opt.save_freq == 0:
            print("Saving checkpoint...")
            torch.save(transformer.state_dict(),
                       os.path.join(opt.model_save_dir, f"{opt.training_name}_epoch{epoch}.pth"))

    # over log
    writer.add_hparams(hparam_dict=dict(vars(Opt)), metric_dict={"loss": early_stopping.best_score})

# TODO: 多GPU训练
# TODO: 开发MMT，注意同步与统一，注意输入检测，这是个大作业
# TODO: 搞好test_video.py
# TODO: embedding学习率调整         后面再来细调
# TODO: 改成KL LabelSmoothing      算了吧
