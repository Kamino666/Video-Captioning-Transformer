from model.model import VideoTransformer
from dataloader import MSRVTT, build_collate_fn
from utils import generate_square_subsequent_mask
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from utils import Meter
import os

from coco_caption.pycocoevalcap.bleu.bleu import Bleu
from coco_caption.pycocoevalcap.rouge.rouge import Rouge
from coco_caption.pycocoevalcap.cider.cider import Cider
from coco_caption.pycocoevalcap.meteor.meteor import Meteor
from coco_caption.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

device = torch.device("cuda")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class EvalOpt:
    # eval
    video_feat_dir = r"/data3/lzh_3/video-captioning-swin-transformer/data/msrvtt_CLIP_fps3_feats/val"
    annotation_file = r"/data3/lzh_3/video-captioning-swin-transformer/data/MSRVTT-annotations/train_val_videodatainfo.json"
    tokenizer_type = "bert-base-uncased"
    model_path = r"./checkpoint/b64_lr0001_dp03_emb768_e4_d4_hd8_hi2048_MSRVTT&CLIP3&SCE_los&clean_earlystop.pth"
    max_len = 30
    batch_size = 32
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


def greedy_decode_dataset(model, test_loader, opt, tokenizer):
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
    from nltk.translate.meteor_score import meteor_score
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    from nltk.tokenize import word_tokenize
    from rouge_score import rouge_scorer
    def bleu_metric_nltk():
        # tokenize
        bleu_pred = []
        for k, v in vid2result.items():
            bleu_pred.append(word_tokenize(v))
        bleu_ref = []
        for k, vs in video2caption.items():
            bleu_ref_vs = []
            for v in vs:
                bleu_ref_vs.append(word_tokenize(v))
            bleu_ref.append(bleu_ref_vs)
        return round(
            corpus_bleu(bleu_ref, bleu_pred, weights=(0, 0, 0, 1), smoothing_function=SmoothingFunction().method2), 4
        )

    def rouge_l_metric_rs():
        m = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_ref = [i for i in video2caption.values()]
        rouge_pred = [i for i in vid2result.values()]
        avg_meter = Meter(mode="avg")
        for i in range(len(rouge_ref)):
            max_meter = Meter(mode="max")
            for j in range(len(rouge_ref[i])):
                max_meter.add(
                    m.score(rouge_ref[i][j], rouge_pred[i])["rougeL"].recall
                )
            avg_meter.add(max_meter.pop())
        return round(avg_meter.pop(), 4)

    def meteor_metric_nltk():
        meteor_ref = [i for i in video2caption.values()]
        meteor_pred = [i for i in vid2result.values()]
        avg_meter = Meter(mode="avg")
        for i in range(len(meteor_ref)):
            res = meteor_score(meteor_ref[i], meteor_pred[i])
            avg_meter.add(res)
        return round(avg_meter.get(), 4)

    if metrics is None:
        metrics = ["bleu", "meteor", "rouge"]
    video2caption = test_iter.video2caption
    vid2result = greedy_decode_dataset(model, test_loader)
    if "bleu" in metrics:
        print("Bleu score: ", end="")
        print(bleu_metric_nltk())
    if "meteor" in metrics:
        print("METEOR score: ", end="")
        print(meteor_metric_nltk())
    if "rouge" in metrics:
        print("ROUGE score: ", end="")
        print(rouge_l_metric_rs())


def coco_eval(model, test_loader, test_iter, opt, tokenizer, verbose=True):
    video2caption = test_iter.video2caption
    global vid2result
    vid2result = greedy_decode_dataset(model, test_loader, opt, tokenizer)
    gts, samples, IDs = make_coco_sample(vid2result, video2caption)

    scorer = COCOScorer(verbose=verbose)
    scorer.score(gts, samples, IDs)
    return scorer
    # print("***********************")
    # print(scorer.eval)
    # print("***********************")
    # print(scorer.imgToEval)


def make_coco_sample(prediction_dict, ground_truth_dict):
    # samples = {
    #   '184321': [{u'image_id': '184321', u'caption': u'train traveling down a track in front of a road'}],
    #   '81922': [{u'image_id': '81922', u'caption': u'plane is flying through the sky'}],
    # }
    # gts = {
    #   '184321': [{u'image_id': '184321', u'caption': u'train traveling down a track in front of a road'}],
    #   '81922': [{u'image_id': '81922', u'caption': u'plane is flying through the sky'}],
    # }
    samples = {}
    IDs = []
    gts = {}
    for vid, cap in prediction_dict.items():
        IDs.append(vid)
        samples[vid] = [{u'image_id': vid, u'caption': cap}]
    for vid, caps in ground_truth_dict.items():
        gts[vid] = []
        for cap in caps:
            gts[vid].append({u'image_id': vid, u'caption': cap})
    return gts, samples, IDs


class COCOScorer(object):
    """
    codes from https://github.com/tylin/coco-caption
    Microsoft COCO Caption Evaluation
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        if self.verbose:
            print('init COCO-EVAL scorer')

    def score(self, GT, RES, IDs):
        """
        GT:
        RES: {
              '184321': [{u'image_id': '184321', u'caption': u'train traveling down a track in front of a road'}],
              '81922': [{u'image_id': '81922', u'caption': u'plane is flying through the sky'}],
              }
        IDs: video id的列表
        """
        self.eval = {}
        self.imgToEval = {}
        gts = {}
        res = {}
        for ID in IDs:
            #            print ID
            gts[ID] = GT[ID]
            res[ID] = RES[ID]
        # get token
        if self.verbose:
            print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        if self.verbose:
            print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        eval = {}
        for scorer, method in scorers:
            if self.verbose:
                print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, IDs, m)
                    if self.verbose:
                        print("%s: %0.3f" % (m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, IDs, method)
                if self.verbose:
                    print("%s: %0.3f" % (method, score))

        # for metric, score in self.eval.items():
        #    print '%s: %.3f'%(metric, score)
        return self.eval

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if imgId not in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score


if __name__ == "__main__":
    vid2result = None
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
    tokenizer = AutoTokenizer.from_pretrained("/data3/lzh_3/video-captioning-swin-transformer/data/tk/")
    opt.start_id = tokenizer.convert_tokens_to_ids("[CLS]")
    opt.end_id = tokenizer.convert_tokens_to_ids("[SEP]")
    opt.pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
    test_iter = MSRVTT(opt.video_feat_dir, opt.annotation_file,
                       tokenizer=tokenizer, mode="validate",
                       by_caption=True, include_id=True)
    test_dataloader = DataLoader(test_iter, batch_size=opt.batch_size, collate_fn=build_collate_fn(opt.pad_id, True))
    # metric_eval(transformer, test_dataloader, test_iter, metrics=["meteor", "bleu", "rouge"])
    scorer = coco_eval(transformer, test_dataloader, test_iter, opt, tokenizer)
    print("***********************")
    print(scorer.eval)
