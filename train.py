import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dataloader import build_dataloader
from utils import Config, setup_seed, configure_hardware
from model.MMT4Caption import MMT4Caption
from eval import v2t_batch, make_coco_sample, COCOScorer

from tqdm import tqdm
from utils import EarlyStopping
import os
import argparse
import random
from tensorboardX import SummaryWriter

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def build_stuffs(train_cfg: dict, model, local_args):
    # optimizer
    if train_cfg['optimizer']['name'] == 'adam':
        if train_cfg['optimizer']['weight_decay'] == 0:
            optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()),
                                         lr=train_cfg['optimizer']['learning_rate'],
                                         betas=train_cfg['optimizer']['beta'])
        else:
            optimizer = torch.optim.AdamW(filter(lambda param: param.requires_grad, model.parameters()),
                                          lr=train_cfg['optimizer']['learning_rate'],
                                          betas=train_cfg['optimizer']['beta'],
                                          weight_decay=train_cfg['optimizer']['weight_decay'])
    elif train_cfg['optimizer']['name'] == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda param: param.requires_grad, model.parameters()),
                                    lr=train_cfg['optimizer']['learning_rate'],
                                    momentum=train_cfg['optimizer']['momentum'])
    else:
        raise ValueError("Do not support optimizer: {}".format(train_cfg['optimizer']['name']))
    # lr_scheduler
    sche_cfg = train_cfg['optimizer']['lr_scheduler']
    if sche_cfg['name'] == 'CosineAnnealingLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=sche_cfg['T_max'], eta_min=sche_cfg['eta_min']
        )
    elif sche_cfg['name'] == 'ReduceLROnPlateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, verbose=True, patience=sche_cfg['patience']
        )
    else:
        raise ValueError("Do not support lr_scheduler: {}".format(sche_cfg['name']))
    # early stop
    early_stopping = EarlyStopping(
        patience=train_cfg['earlystop'],
        verbose=True,
        # path=os.path.join(train_cfg['save_dir'], train_cfg['tag'] + str(local_args.local_rank) + "_earlystop.pth"),
        path=os.path.join(train_cfg['save_dir'], train_cfg['tag'] + "_earlystop.pth"),
    )
    # writer
    writer = None
    if local_args.is_main_rank:
        writer = SummaryWriter(os.path.join(train_cfg['log_dir'], train_cfg['tag']))
    return optimizer, lr_scheduler, early_stopping, writer


def logging(writer, epoch, task, train_loss, val_loss, **kwargs):
    def _log_metric():
        print(f"Bleu@4: {round(kwargs['metrics'][0] * 100, 1)}", end='\t')
        print(f"METEOR: {round(kwargs['metrics'][1] * 100, 1)}", end='\t')
        print(f"ROUGE_L: {round(kwargs['metrics'][2] * 100, 1)}", end='\t')
        print(f"CIDEr: {round(kwargs['metrics'][3] * 100, 1)}")
        writer.add_scalar("Bleu@4", kwargs['metrics'][0] * 100, epoch)
        writer.add_scalar("METEOR", kwargs['metrics'][1] * 100, epoch)
        writer.add_scalar("ROUGE_L", kwargs['metrics'][2] * 100, epoch)
        writer.add_scalar("CIDEr", kwargs['metrics'][3] * 100, epoch)
    if writer is None:
        return
    print(f"Epoch: {epoch}")
    if task == "cross":
        print(f" Train: train loss: {train_loss[0]:.3f}\t"
              f" train_cap_loss: {train_loss[1]:.3f}\t"
              f" train_match_loss: {train_loss[2]:.3f}")
        print(f" Val: val loss: {val_loss[0]:.3f}\t"
              f" val_cap_loss: {val_loss[1]:.3f}\t"
              f" val_match_loss: {val_loss[2]:.3f}")
        if kwargs.get('metrics', None) is not None:
            _log_metric()
        writer.add_scalar("train_loss", train_loss[0], epoch)
        writer.add_scalar("train_cap_loss", train_loss[1], epoch)
        writer.add_scalar("train_match_loss", train_loss[2], epoch)
        writer.add_scalar("val_loss", val_loss[0], epoch)
        writer.add_scalar("val_cap_loss", val_loss[1], epoch)
        writer.add_scalar("val_match_loss", val_loss[2], epoch)
    elif task == "caption":
        print(f" train loss: {train_loss[0]:.3f}")
        print(f" val loss: {val_loss[0]:.3f}")
        if kwargs.get('metrics', None) is not None:
            _log_metric()
        writer.add_scalar("train_cap_loss", train_loss[0], epoch)
        writer.add_scalar("val_cap_loss", val_loss[0], epoch)
    elif task == "match":
        print(f" train loss: {train_loss[0]:.3f}")
        print(f" val loss: {val_loss[0]:.3f}")
        writer.add_scalar("train_match_loss", train_loss[0], epoch)
        writer.add_scalar("val_match_loss", val_loss[0], epoch)

    if 'lr' in kwargs:
        writer.add_scalar('lr', kwargs['lr'], epoch)

    if 'sample' in kwargs:
        truth_caption, pred_caption, vid = kwargs['sample']
        print(f"{vid} truth\t: {truth_caption} \n {vid} pred\t: {pred_caption}")


def train_epoch(model: MMT4Caption, optimizer, dataloader, mode, local_args):
    model.train()
    model.module.mode(mode) if local_args.multi_gpu else model.mode(mode)
    running_loss, running_cap_loss, running_match_loss = 0, 0, 0
    loader_len = len(dataloader)
    # feat_ts, feat_mask_ts, batch_captions, batch_vids
    for v_feats, v_masks, captions, vids in tqdm(dataloader):
        v_feats = [i.to(local_args.device) for i in v_feats]
        v_masks = [i.to(local_args.device) for i in v_masks]
        if mode != 'cross':
            loss = model(v_feats, v_masks, captions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if local_args.multi_gpu:
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss /= local_args.world_size
            running_loss += loss.item()
        else:
            loss, cap_loss, match_loss = model(v_feats, v_masks, captions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if local_args.multi_gpu:
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(cap_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(match_loss, op=dist.ReduceOp.SUM)
                loss /= local_args.world_size
                cap_loss /= local_args.world_size
                match_loss /= local_args.world_size
            running_loss += loss.item()
            running_cap_loss += cap_loss.item()
            running_match_loss += match_loss.item()
    return running_loss / loader_len, running_cap_loss / loader_len, running_match_loss / loader_len


@torch.no_grad()
def val_epoch(model: MMT4Caption, dataloader, mode, local_args):
    model.eval()
    model.module.mode(mode) if local_args.multi_gpu else model.mode(mode)
    loader_len = len(dataloader)
    running_loss, running_cap_loss, running_match_loss = 0, 0, 0
    for v_feats, v_masks, captions, vids in dataloader:
        v_feats = [i.to(local_args.device) for i in v_feats]
        v_masks = [i.to(local_args.device) for i in v_masks]
        if mode != 'cross':
            loss = model.module(v_feats, v_masks, captions)
            running_loss += loss.item()
        else:
            loss, cap_loss, match_loss = model.module(v_feats, v_masks, captions)
            running_loss += loss.item()
            running_cap_loss += cap_loss.item()
            running_match_loss += match_loss.item()
    return running_loss / loader_len, running_cap_loss / loader_len, running_match_loss / loader_len


@torch.no_grad()
def eval_epoch(model: MMT4Caption, data_iter, dataloader, max_len, local_args):
    # evaluate
    model_core = model.module if local_args.multi_gpu else model
    model.eval()
    model_core.mode("caption")
    vid2result, video2caption = {}, data_iter.video2caption
    for v_feats, v_masks, _, vids in tqdm(dataloader):
        pred_captions = v2t_batch(model_core, v_feats, v_masks, max_len=max_len, local_args=local_args)
        vid2result.update(list(zip(vids, pred_captions)))
    # Coco eval
    gts, samples, IDs = make_coco_sample(vid2result, video2caption)
    scorer = COCOScorer(verbose=False)
    scorer.score(gts, samples, IDs)
    return scorer.eval['Bleu_4'], scorer.eval['METEOR'], scorer.eval['ROUGE_L'], scorer.eval['CIDEr']
    # # syn the data
    # metrics_ts = torch.Tensor([scorer.eval['Bleu_4'], scorer.eval['METEOR'], scorer.eval['ROUGE_L'], scorer.eval['CIDEr']])
    # if local_args.multi_gpu:
    #     metrics_ts = metrics_ts.to(local_args.device)
    #     tensor_list = [torch.zeros(4, device=local_args.device) for _ in range(local_args.world_size)]
    #     # print(tensor_list[0].device, metrics_ts.device)
    #     dist.all_gather(tensor_list, metrics_ts)  # tensor_list: all rank is same
    #     return tensor_list
    # else:
    #     return metrics_ts


@torch.no_grad()
def v2t_single(model: MMT4Caption, video_feat, max_len, local_args):
    model.eval()
    video_feat = [i.unsqueeze(0).to(local_args.device) for i in video_feat]

    result = model.greedy_decode(video_feat, max_len=max_len)[0]
    result = result.replace("[CLS]", "").replace("[SEP]", "")
    return result


def mmt4caption_train(cfg: dict, local_args):
    # build model
    model = MMT4Caption(cfg['model'], device=local_args.device).to(local_args.device)
    model.mode(cfg['train']['task'])
    if 'univl' in cfg['model']['caption_decoder'] and cfg['model']['caption_decoder']['univl'] is not None:
        model.load_cap_decoder_from_univl(cfg['model']['caption_decoder']['univl'])
    if cfg['model']['pretrained_model'] is not None:
        model.load_state_dict(torch.load(cfg['model']['pretrained_model'], map_location=local_args.device),
                              strict=False)
    if local_args.multi_gpu:
        model = DDP(model, device_ids=[local_args.local_rank], output_device=local_args.local_rank)
        model_core = model.module
    else:
        model_core = model

    # build stuffs
    optimizer, lr_scheduler, early_stopping, writer = build_stuffs(cfg['train'], model, local_args)

    # build dataloaders
    train_iter, train_dataloader, train_sampler = build_dataloader(cfg['data']['train'], local_args.multi_gpu)
    val_iter, val_dataloader, _ = build_dataloader(cfg['data']['validation'], local_args.multi_gpu)
    eval_iter, eval_dataloader, _ = build_dataloader(cfg['data']['eval'], local_args.multi_gpu)

    # START
    for epoch in range(cfg['train']['epoch']):
        # Set epoch for sampler
        if train_sampler is not None:
            # print("train_sampler set epoch!!")
            train_sampler.set_epoch(epoch)
        # Start training
        train_loss = train_epoch(model, optimizer, train_dataloader, mode=cfg['train']['task'], local_args=local_args)
        lr_scheduler.step()

        # Do many validations (only in rank:0)
        val_loss, metrics = None, None
        # calculate val loss
        if local_args.is_main_rank:
            val_loss = val_epoch(model, val_dataloader, mode=cfg['train']['task'], local_args=local_args)
        dist.barrier()  # syn each process
        # calculate metrics
        if local_args.is_main_rank and cfg['train'].get('metric_earlystop', True) is True:
            metrics = eval_epoch(model, eval_iter, eval_dataloader, max_len=cfg['test']['max_length'], local_args=local_args)
        dist.barrier()  # syn each process
        # predict a sample
        pred_caption, truth_caption, vid = None, None, None
        if local_args.is_main_rank:
            video_feat, truth_caption, vid = val_iter[random.randint(0, len(val_iter) - 1)]
            pred_caption = v2t_single(model_core, video_feat, max_len=cfg['test']['max_length'], local_args=local_args)
        dist.barrier()  # syn each process

        # logging (only in rank:0)
        logging(writer, epoch, cfg['train']['task'], train_loss, val_loss,
                lr=optimizer.state_dict()['param_groups'][0]['lr'],
                sample=(truth_caption, pred_caption, vid),
                metrics=metrics)

        # early stopping
        if cfg['train'].get('metric_earlystop', True) is True:
            # get metric score data from rank:0 to update the early_stopping
            met_score = torch.zeros([1], dtype=torch.float) if metrics is None else torch.Tensor([sum(metrics)])
            met_score = met_score.to(local_args.device)
            dist.all_reduce(met_score, op=dist.ReduceOp.SUM)
            early_stopping(-met_score.cpu().item(), model_core, do_save=local_args.is_main_rank)
        else:
            if val_loss is None:
                val_loss = 0.0
            elif type(val_loss) is tuple:
                val_loss = val_loss[0]
            else:
                val_loss = val_loss
            # get metric score data from rank:0 to update the early_stopping
            val_loss = torch.Tensor([val_loss]).to(local_args.device)
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            early_stopping(val_loss.cpu().item(), model_core, do_save=local_args.is_main_rank)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        # save
        if epoch % cfg['train']['save_frequency'] == 0 and epoch != 0 and local_args.is_main_rank:
            print("Saving checkpoint...")
            torch.save(model_core.state_dict(),
                       os.path.join(cfg['train']['save_dir'], f"{cfg['train']['tag']}_epoch{epoch}.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str,
                        help="The path of '.json' config file")
    parser.add_argument("-ws", "--world_size", type=int, default=4,
                        help="The number of GPUs(Only need when --multi_gpu is on)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cpu", action="store_true", help="use cpu")
    group.add_argument("--gpu", action="store_true", help="use gpu")
    group.add_argument("--multi_gpu", action="store_true", help="use multiple gpu")
    args_ = parser.parse_args()

    # configure hardware
    args_ = configure_hardware(args_)

    # set seed
    setup_seed(666)

    # load config
    cfg_ = Config(args_.config)
    if args_.is_main_rank:
        cfg_.display()

    mmt4caption_train(cfg_.data, args_)
