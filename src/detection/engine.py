from collections import namedtuple
import math
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from . import (
    learning_rate,
    loss_reduction,
    metric_logging,
    metrics,
)

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = metric_logging.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr',
                            metric_logging.SmoothedValue(window_size=1,
                                                         fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = learning_rate.warmup_lr_scheduler(optimizer,
                                                         warmup_iters,
                                                         warmup_factor)

    for images, targets in metric_logger.log_every(data_loader,
                                                   print_freq,
                                                   header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = loss_reduction.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def _get_gt_pred_df(model,
                    data_loader,
                    device):
    """turn ground truth annotations and predictions into
    ```pandas.DataFrame``s that are used to compute mAP"""
    gt_records = []  # 'gt' = "ground truth"
    pred_records = []

    pbar = tqdm(data_loader)

    for batch_id, batch in enumerate(pbar):
        img, target = batch[0], batch[1]

        # first use `target` to extend ground truth records
        assert len(target) == 1
        target = target[0]
        bboxes = target['boxes'].numpy().tolist()
        labels = target['labels'].numpy().tolist()
        set_size = len(bboxes)
        for bbox, label in zip(bboxes, labels):
            gt_records.append(
                {
                    'id': batch_id,
                    'class': label,
                    'xmin': bbox[0],
                    'ymin': bbox[1],
                    'xmax': bbox[2],
                    'ymax': bbox[3],
                    'img_set_size': set_size,
                }
            )

        # use `img` + `model` to extend prediction records
        img = [img_.to(device) for img_ in img]  # hack, model expects list of images
        with torch.no_grad():
            out = model(img)
            assert len(out) == 1  # because batch size is 1. so we can assume batch id is image id
            out_dict = out[0]
            boxes_np = out_dict['boxes'].cpu().numpy()
            labels_np = out_dict['labels'].cpu().numpy()
            scores_np = out_dict['scores'].cpu().numpy()
            for bbox, label, score in zip(boxes_np, labels_np, scores_np):
                pred_records.append(
                    {
                        'id': batch_id,
                        'class': label,
                        'score': score,
                        'xmin': bbox[0],
                        'ymin': bbox[1],
                        'xmax': bbox[2],
                        'ymax': bbox[3],
                        'img_set_size': set_size,
                    }
                )

    gt_df = pd.DataFrame.from_records(gt_records)
    gt_df['detected'] = False

    pred_df = pd.DataFrame.from_records(pred_records)

    return gt_df, pred_df


EvalResults = namedtuple('EvalResults',
                         field_names=['rec', 'prec', 'ap'])


@torch.no_grad()
def evaluate(model,
             data_loader,
             device,
             overlap_thresh=0.5):
    """evaluate model, by computing mean Average Precision with Pascal VOC-like approach

    Parameters
    ----------
    model : torch.nn.Module
    data_loader : torch.utils.dataloader
    device : str
        torch 'device', 'cpu' or 'cuda'
    overlap_thresh : float
        Threshold used to determine whether
        predicted bounding boxes overlap
        with ground truth, when computing Average
        Precision. Default is 0.5

    Returns
    -------
    mAP : float
        mean Average Precision across classes,
        where Average Precision is computed by
        ``detection.metrics.class_avg_precision``
    ap_class_map : dict
        where keys are class labels, and values
        are named tuples containing results returned
        by ``detection.metrics.class_avg_precision``
        (with fields 'rec', 'prec', and 'ap')
    gt_df_out : list
        of pandas.DataFrame, one for each class in
        the ground truth labels obtained from
        the ``data_loader``
    """
    model.eval()

    gt_df, pred_df = _get_gt_pred_df(model, data_loader, device)

    classes = gt_df['class'].unique()
    ap_class_map = {}
    gt_df_out = []
    for class_label in classes:
        rec, prec, ap, gt_df_class = metrics.class_avg_precision(gt_df,
                                                                 pred_df,
                                                                 class_int=class_label,
                                                                 overlap_thresh=overlap_thresh)
        ap_class_map[class_label] = EvalResults(rec=rec, prec=prec, ap=ap)
        gt_df_out.append(gt_df_class)

    mAP = np.array(
        [eval_result.ap for eval_result in ap_class_map.values()]
    ).mean()

    return mAP, ap_class_map, gt_df_out
