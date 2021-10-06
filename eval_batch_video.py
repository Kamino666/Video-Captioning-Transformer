from datasets import load_metric
from train import VideoTransformer, MSRVTT, VATEX
from train import generate_square_subsequent_mask
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from utils import Meter

device = torch.device("cuda")


class EvalOpt:
    # eval
    video_feat_dir = r"data/MSRVTT-CLIP-FEATURES/val_feats"
    annotation_file = r"./data/MSRVTT-annotations/train_val_videodatainfo.json"
    tokenizer_type = "bert-base-uncased"
    model_path = r"./checkpoint/b32_msrvttCLIP_enc1_dec1_head4_emb512_hid1024_epoch100.pth"
    max_len = 30
    batch_size = 16
    # model
    bert_type = "bert-base-uncased"
    enc_layer_num = 1
    dec_layer_num = 1
    head_num = 4
    feat_size = 512
    emb_dim = 768
    hid_dim = 1024
    dropout = 0.1
    use_bert = False


def collate_fn(data):
    """
    :param data:
    :return tuple(N T E, N T):
    """
    batch_size = len(data)
    # video id
    id_data = [i[2] for i in data]

    # video feature
    feat_dim = data[0][0].shape[1]
    feat_data = [i[0] for i in data]
    feat_len = [len(i) for i in feat_data]
    max_len = max(feat_len)
    feat_ts = torch.zeros([batch_size, max_len, feat_dim], dtype=torch.float)
    feat_mask_ts = torch.ones([batch_size, max_len], dtype=torch.long)
    for i in range(batch_size):
        feat_ts[i, :feat_len[i]] = feat_data[i]
        feat_mask_ts[i, :feat_len[i]] = 0
    feat_mask_ts = (feat_mask_ts == 1)

    # text
    text_data = [i[1] for i in data]
    text_len = [len(i) for i in text_data]
    max_len = max(text_len)
    text_ts = torch.ones([batch_size, max_len], dtype=torch.long) * opt.pad_id
    for i in range(batch_size):
        text_ts[i, :text_len[i]] = text_data[i]
    text_mask_ts = (text_ts == opt.pad_id)
    return feat_ts, text_ts, feat_mask_ts, text_mask_ts, id_data


def greedy_decode_dataset(model, test_loader):
    vid2result = {}
    for src, tgt, src_padding_mask, tgt_padding_mask, video_ids in tqdm(test_loader):
        src = src.to(device)  # N, T
        memory = model.encode(src).to(device)  # N, T, E
        batch_size = src.shape[0]

        ys = torch.ones(batch_size, 1).fill_(opt.start_id).type(torch.long).to(device)  # N, 1
        end_flag = [0 for i in range(batch_size)]
        for i in range(opt.max_len - 1):
            tgt_mask = (generate_square_subsequent_mask(ys.shape[1]).type(torch.bool)).to(device)  # t, t
            out = model.decode(ys, memory, tgt_mask)
            prob = model.generator(out[:, -1])  # N, vocab_size
            _, next_word = torch.max(prob, dim=1)  # N

            ys = torch.cat([ys, next_word.unsqueeze(1).type(torch.long)], dim=1)  # N, t

            # break flag
            for k, flag in enumerate((next_word == opt.end_id).tolist()):
                if flag is True:
                    end_flag[k] = 1
            if sum(end_flag) >= batch_size:
                break
        for k, v in zip(video_ids, ys.tolist()):
            v = tokenizer.convert_ids_to_tokens(v)
            end_count = -1
            for i, token in enumerate(v):
                if token == "[SEP]":
                    end_count = i
                    break
            v = v[1:end_count]
            v = tokenizer.convert_tokens_to_string(v)
            vid2result[k] = v
    return vid2result


def metric_eval(model, test_loader, test_iter, metrics=None):
    def bleu_metric(m):
        import re
        bleu_pred = []
        for k, v in vid2result.items():
            # tokenize
            v = v.lower()
            v = re.sub(r'[~\\/():;]', ' ', v)  # .!,?
            v = v.replace(".", " . ").replace("!", " ! ").replace("?", " ? ").replace(",", " , ")
            bleu_pred.append(v.split())
        bleu_ref = []
        for k, vs in video2caption.items():
            bleu_ref_vs = []
            for v in vs:
                # tokenize
                v = v.lower()
                v = re.sub(r'[~\\/():;]', ' ', v)  # .!,?
                v = v.replace(".", " . ").replace("!", " ! ").replace("?", " ? ").replace(",", " , ")
                bleu_ref_vs.append(v.split())
            bleu_ref.append(bleu_ref_vs)
        bleu_1 = m.compute(predictions=bleu_pred, references=bleu_ref, max_order=1)
        bleu_2 = m.compute(predictions=bleu_pred, references=bleu_ref, max_order=2)
        bleu_3 = m.compute(predictions=bleu_pred, references=bleu_ref, max_order=3)
        bleu_4 = m.compute(predictions=bleu_pred, references=bleu_ref, max_order=4)
        return bleu_1["bleu"], bleu_2["bleu"], bleu_3["bleu"], bleu_4["bleu"]

    def rouge_l_metric(m):
        rouge_ref = [i for i in video2caption.values()]
        rouge_pred = [i for i in vid2result.values()]
        for i in range(len(rouge_ref)):
            for j in range(len(rouge_ref[i])):
                m.add(prediction=rouge_pred[i], reference=rouge_ref[i][j])
        meteor = m.compute()
        return meteor['rougeL'].mid.recall

    def meteor_metric(m):
        meteor_ref = [i for i in video2caption.values()]
        meteor_pred = [i for i in vid2result.values()]
        avg_meter = Meter(mode="avg")
        for i in range(len(meteor_ref)):
            max_meter = Meter(mode="max")
            for j in range(len(meteor_ref[i])):
                m.add(prediction=meteor_pred[i], reference=meteor_ref[i][j])
                max_meter.add(m.compute()["meteor"])
            avg_meter.add(max_meter.get())
        # meteor = m.compute()
        return avg_meter.get()

    if metrics is None:
        metrics = ["bleu", "meteor", "rouge"]
    video2caption = test_iter.video2caption
    vid2result = greedy_decode_dataset(model, test_loader)
    if "bleu" in metrics:
        print(bleu_metric(load_metric("./metric_config/bleu.py")))
    if "meteor" in metrics:
        print(meteor_metric(load_metric("./metric_config/meteor.py")))
    if "rouge" in metrics:
        print(rouge_l_metric(load_metric("./metric_config/rouge.py")))


if __name__ == "__main__":
    # load configs
    opt = EvalOpt()
    # load model
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
    transformer.load_state_dict(torch.load(opt.model_path))
    transformer.eval()
    transformer = transformer.to(device)
    # load data
    tokenizer = AutoTokenizer.from_pretrained("./data/tk/")
    opt.start_id = tokenizer.convert_tokens_to_ids("[CLS]")
    opt.end_id = tokenizer.convert_tokens_to_ids("[SEP]")
    opt.pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
    test_iter = MSRVTT(opt.video_feat_dir, opt.annotation_file, tokenizer=tokenizer, mode="validate")
    test_dataloader = DataLoader(test_iter, batch_size=opt.batch_size, collate_fn=collate_fn)
    metric_eval(transformer, test_dataloader, test_iter, metrics=["meteor", "bleu"])
