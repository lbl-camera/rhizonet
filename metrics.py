import os
import torch
import json 
import numpy as np
import torchmetrics
from skimage import io
from utils import createBinaryAnnotation, transform_annot

def evaluate_all_metrics(pred, target, task='binary', num_classes=2, background_index=0):

    cnfmat = torchmetrics.ConfusionMatrix(
                                        num_classes=num_classes,
                                        task=task,
                                        normalize=None
                                        )

    cnfmat = cnfmat(pred, target)
    true = torch.diag(cnfmat)
    tn = true[background_index]
    tp = torch.cat([true[:background_index], true[background_index + 1:]])

    fn = (cnfmat.sum(1) - true)[torch.arange(cnfmat.size(0)) != background_index]
    fp = (cnfmat.sum(0) - true)[torch.arange(cnfmat.size(1)) != background_index]

    acc = torch.sum(true) / torch.sum(cnfmat)
    precision = torch.sum(tp) / torch.sum(tp + fp)
    recall = torch.sum(tp) / torch.sum(tp + fn)
    iou = torch.sum(tp) / (torch.sum(cnfmat) - tn)
    iou_per_class = tp / (tp + fp + fn)

    dice = 2*torch.sum(tp) / (torch.sum(cnfmat) - tn + tp) 

    return acc.item(), precision.item(), recall.item(), iou.item(), dice.item()


def main(pred_path, label_path, log_dir):
    pred_list = sorted([os.path.join(pred_path, e) for e in os.listdir(pred_path) if not e.startswith(".")])
    label_list = sorted([os.path.join(label_path, e) for e in os.listdir(label_path) if not e.startswith(".")])
    dict_metrics = {}
    for pred,lab in zip(pred_list, label_list):
        prediction, label = torch.tensor(transform_annot(io.imread(pred), value_map={"0":"0", "255":"1"})), torch.tensor(transform_annot(io.imread(lab), value_map={"0":"0", "85":"1", "170":"0"}))
        # dict_metrics["preds_path"] = pred
        # dict_metrics["label_path"] = lab
        filename = os.path.basename(pred)
        dict_metrics[filename] = {}
        acc, prec, rec, iou, dice = evaluate_all_metrics(prediction, label)
        dict_metrics[filename]['accuracy'] = acc
        dict_metrics[filename]['precision'] = prec
        dict_metrics[filename]['recall'] = rec
        dict_metrics[filename]['IOU'] = iou
        dict_metrics[filename]['Dice'] = dice
    # print("Metrics: \n {}".format(dict_metrics))
    with open(os.path.join(log_dir, 'metrics.json'), 'w') as f:
        json.dump(dict_metrics, f)
            
if __name__ == '__main__':
    pred_path = "/home/zsordo/rhizonet-fovea/results/training_patches64_ex7ex9_batch32_dropout40/predictions/"
    label_path = "/home/zsordo/rhizonet-fovea/data/test_data/labels/"
    log_dir = "/home/zsordo/rhizonet-fovea/results/training_patches64_ex7ex9_batch32_dropout40"
    main(pred_path, label_path, log_dir)