import os
from utils import Config, setup_seed, configure_hardware
import argparse
from tqdm import tqdm
from typing import Any, Optional

from model.MMT4Caption import MMT4Caption
from dataloader import build_dataloader
import torch

from submodules.pycocoevalcap.bleu.bleu import Bleu
from submodules.pycocoevalcap.rouge.rouge import Rouge
from submodules.pycocoevalcap.cider.cider import Cider
from submodules.pycocoevalcap.meteor.meteor import Meteor
from submodules.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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


@torch.no_grad()
def v2t_batch(model: MMT4Caption, video_feats: list, video_masks: Optional[list],
              local_args, max_len: int = 30):
    """
    :param model: Model of MMT4Caption
    :param video_feats: List of video features
    :param video_masks: List of video masks
    :param max_len: Max length of generated sentences
    :param local_args: Some arguments of GPU settings
    :return: List of result strings
    """
    model.eval()
    video_feats = [i.to(local_args.device) for i in video_feats]
    video_masks = [i.to(local_args.device) for i in video_masks] if video_masks is not None else None

    results = model.greedy_decode(video_feats, video_masks, max_len=max_len)
    results = [r.replace("[CLS]", "").replace("[SEP]", "") for r in results]
    return results


def evaluate(cfg: dict, local_args):
    # build model
    model = MMT4Caption(cfg['model'], device=local_args.device).to(local_args.device)
    model.mode("caption")
    load_state = model.load_state_dict(
        torch.load(local_args.model, map_location=local_args.device), strict=False
    )
    print(f"Load state: {load_state}")
    # build dataloader
    val_iter, val_dataloader, _ = build_dataloader(cfg['data']['eval'], multi_gpu=False)
    # evaluate
    model.eval()
    vid2result, video2caption = {}, val_iter.video2caption
    for v_feats, v_masks, _, vids in tqdm(val_dataloader):
        pred_captions = v2t_batch(model, v_feats, v_masks, max_len=cfg['test']['max_length'], local_args=local_args)
        vid2result.update(list(zip(vids, pred_captions)))
    # Coco eval
    gts, samples, IDs = make_coco_sample(vid2result, video2caption)
    scorer = COCOScorer(verbose=True)
    scorer.score(gts, samples, IDs)
    print("***********************")
    print(scorer.eval)
    print("***********************")
    return scorer, vid2result


if __name__ == "__main__":
    setup_seed(666)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str, help="The path of '.json' config file")
    parser.add_argument("-m", "--model", required=True, type=str, help="The path of model checkpoint")
    # Multi-GPU not supported
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cpu", action="store_true", help="use cpu or not")
    group.add_argument("--gpu", action="store_true", help="use gpu or not")
    args_ = parser.parse_args()

    # configure hardware
    args_ = configure_hardware(args_)

    # load config
    cfg_ = Config(args_.config)
    cfg_.data['model']['pretrained_model'] = None
    if args_.is_main_rank:
        cfg_.display()

    scorer, vid2result = evaluate(cfg_.data, args_)
