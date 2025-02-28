"""
Script for evaluation metrics on a given set of predictions and groundtruth. 
The metrics used are Accuracy, Precision, Recall and IOU. 

Usage:
    pip install rhizonet
    evalmetrics_rhizonet ---pred_path "path" --label_path "path" --log_dir "path" --task "binary" --num_classes "2" --frg_class 85
"""

import os
import torch
import json 
import numpy as np
from argparse import ArgumentParser
import torchmetrics
from skimage import io
from typing import Tuple, Sequence, List

try:
    from .utils import createBinaryAnnotation, MapImage
except ImportError:
    from utils import createBinaryAnnotation, MapImage


def calculate_all_metrics(pred: torch.Tensor, 
                         target: torch.Tensor,  
                         task: str = 'binary', 
                         num_classes: int = 2, 
                         background_index: int = 0) -> Tuple[float, float, float, float]:
    """
    Evaluate Accuracy, Precision, Recall and IOU based on prediction and groundtruth and for a given ask (e.g. `binary`).

    Args:
        pred (torch.Tensor): predicted image
        target (torch.Tensor): groundtruth image
        task (str, optional): type of classification for each pixel (binary or multi-class). Defaults to 'binary'.
        num_classes (int, optional): number of classes. Defaults to 3.
        background_index (int, optional): index value associated to the background. Defaults to 0.

    Returns:
        Tuple[float, float, float, float]: averaged accuracy, precision, recall and IOU
    """

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
    dice = 2 * torch.sum(tp) / (2 * torch.sum(tp) + torch.sum(fp) + torch.sum(fn))
    
    return acc.item(), precision.item(), recall.item(), iou.item(), dice.item()


def evaluate(pred_path: str, label_path: str, log_dir: str, task: str, num_classes: int, frg_class: int = 255) -> None:
    """
    Reads the prediction and groundtruth images, evaluates the metrics Accuracy, Precision, Recall and IOU.
    Saves results in `metrics.json` file in the specified `log_dir`. 
    There are 2 options: 
    - evaluate metrics on multi-class prediction.
    - evaluate metrics on binary segmentation mask (e.g. if the prediction is processed into a binary mask with root as foreground and the rest labeled as background.)

    Args:
        pred_path (str): filepath of the predicted image
        label_path (str): filepath of the groundtruth image
        log_dir (str): filepath where results will be saved in a json file
        num_classes (int): number of class labels
        task (int): type of segmentation task if processing binary segmentation masks or multiclass segmentation images (e.g. `binary` or `multiclass`)
        frg_class (int): value of the foreground class when creating binary segmentation masks

    """
    pred_list = sorted([os.path.join(pred_path, e) for e in os.listdir(pred_path) if not e.startswith(".")])
    label_list = sorted([os.path.join(label_path, e) for e in os.listdir(label_path) if not e.startswith(".")])
    dict_metrics = {}


    # if number of classes in the prediction is 2 then binary else multiclass with number of classes 
    for pred_path, lab_path in zip(pred_list, label_list):
        pred = io.imread(pred_path) # Predicted image 
        lab = io.imread(lab_path) # Groundtruth image

        if task == 'binary':
            lab = createBinaryAnnotation(lab, frg_class)
            lab = torch.Tensor(lab/255.0)

        # Check if prediction is scaled by 255 for visualization 
        if np.min(pred) >= 0 and np.max(pred) == 255:
            pred = torch.Tensor(pred/255.0)

        original_labels = np.unique(pred)
        pred = MapImage(torch.tensor(pred), original_labels, reverse=False)
        lab = MapImage(torch.tensor(lab), original_labels, reverse=False)
        filename = os.path.basename(pred_path)

        dict_metrics[filename] = {}
        acc, prec, rec, iou, dice = calculate_all_metrics(pred, lab, task=task, num_classes=num_classes)
        dict_metrics[filename]['accuracy'] = acc
        dict_metrics[filename]['precision'] = prec
        dict_metrics[filename]['recall'] = rec
        dict_metrics[filename]['IOU'] = iou
        dict_metrics[filename]['Dice'] = dice
                                                                                                                                                 
    # print("Metrics: \n {}".format(dict_metrics))
    with open(os.path.join(log_dir, 'metrics.json'), 'w') as f:
        json.dump(dict_metrics, f)
        
def main():
    parser = ArgumentParser(conflict_handler='resolve', description='Run inference using trained model')
    parser.add_argument("--pred_path", type=str,
                        default=".results/training_patches64_ex7ex9_batch32_dropout40/predictions/",
                        help="path to the predictions")
    parser.add_argument("--label_path", 
                        default="path to the groundtruth")
    parser.add_argument("--log_dir",
                        default="/home/zsordo/rhizonet-fovea/results/training_patches64_ex7ex9_batch32_dropout40",
                        help="path where the json file containing results will be saved")
    
    parser.add_argument("--task",
                        default='binary',
                        choices=['binary', 'multiclass'],
                        help="type of segmentation task if processing binary segmentation masks or multiclass segmentation images")
    parser.add_argument("--num_classes",
                        default=2,
                        help="Number of classes including background")
    parser.add_argument("--frg_class",
                        default=255,
                        help="value of the foreground class when using binary segmentation masks")
    
    args = parser.parse_args()
    args = vars(args)
    evaluate(args['pred_path'], args ['label_path'], args['log_dir'], args['task'], args['num_classes'], args['frg_class'])

if __name__ == '__main__':
    main()