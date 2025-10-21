"""
Script for running inference using a pre-trained residual U-Net model for image segmentation.

Usage:
    pip install rhizonet
    predict_rhizonet --config_file ./setup_files/setup-predict.json 
"""

import os
import glob
import argparse
import json
import re
from tqdm import tqdm
from argparse import ArgumentParser
from skimage import io, util, color, filters, exposure, measure
from PIL import Image
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.color import rgb2lab

from monai.data import ArrayDataset, create_test_image_2d, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    ScaleIntensityRange,
    EnsureType
)

from PIL import ImageDraw
import torchvision.transforms.functional as TF
from datetime import datetime

from typing import Tuple, List, Dict, Sequence, Union, Any
from collections.abc import Callable

# Import parent modules 
try:
    from .utils import MapImage, createBinaryAnnotation, extract_largest_component_bbox_image
    from .unet2D import Unet2D, PredDataset2D, ImageDataset, tiff_reader, dynamic_scale
except ImportError:
    from utils import MapImage, createBinaryAnnotation, extract_largest_component_bbox_image
    from unet2D import Unet2D, PredDataset2D, ImageDataset, tiff_reader, dynamic_scale


def _parse_training_variables(argparse_args) -> Dict:
    """ 
    Parse and merge training variables from a JSON configuration file and command-line arguments.

    Args:
        argparse_args (Namespace): Command-line arguments parsed by argparse.

    Returns:
        Dict: Updated arguments 
    """


    args = vars(argparse_args)
    # overwrite argparse defaults with config file
    with open(args["config_file"]) as file_json:
        config_dict = json.load(file_json)
        args.update(config_dict)
    args['pred_patch_size'] = tuple(args['pred_patch_size']) # tuple expected, not list
    if args['gpus'] is None:
        args['gpus'] = -1 if torch.cuda.is_available() else 0

    return args


def transform_image(img_path:str) -> Tuple[np.ndarray, str]:
    """
    Reads the filepath and returns the image in the correct shape for inference (C, H, W)

    Args:
        img_path (str): Filepath of the input image

    Returns:
        Tuple[np.ndarray, str]: Image in the correct shape, Filepath of the image 
    """
    transform = Compose(
        [
            EnsureType()
        ]
    )
    img = io.imread(img_path)# only 3 modalities are accepted in the channel dimension for now
    if img.ndim == 4 and img.shape[-1] < 4:  # If shape is (h, w, d, c) assuming there are maximum 4 channels or modalities 
        img = np.transpose(img[..., :3] , (3, 0, 1, 2))  # Move channel to the first position
        img = dynamic_scale(img)
    elif img.ndim == 3 and img.shape[-1] <= 4:  # If shape is (h, w, c)
        img = np.transpose(img[..., :3] , (2, 0, 1))  # Move channel to the first position
        img = dynamic_scale(img)
    elif img.ndim == 2: # if no batch dimension then create one
        img = np.expand_dims(img, axis=-1)
        img = dynamic_scale(img)
        img = np.transpose(img, (2, 0, 1))
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}, channel dimension should be last and image should be either 2D or 3D")
        
    img = transform(img)
    return img, img_path


def pred_function(
        image : torch.Tensor, 
        model: Callable,
        pred_patch_size: Sequence[int]
        ) -> torch.Tensor:
    """
    Sliding window inference on `image` with `model`

    Args:
        image (torch.Tensor): input image to be processed
        model (Callable): given input tensor ``image`` in shape BCHW[D], the outputs of the function call ``model(input)`` should be a tensor.
        pred_patch_size (Sequence[int]): spatial window size for inference

    Returns:
        torch.Tensor: prediction tensor
    """
    
    return sliding_window_inference(inputs=image, roi_size=pred_patch_size, sw_batch_size=1, predictor=model)


def predict_step(
        image_path: str, 
        model: Callable,
        pred_patch_size: Sequence[int]
        ) -> torch.Tensor:
    """
    Call trained model and run inference on input image given by the filepath using monai's ``sliding_window_inference`` function. 

    Args:
        image_path (str): filepath of the input image to be processed
        model (Callable): given input tensor ``image`` in shape BCHW[D], the outputs of the function call ``model(input)`` should be a tensor.
        pred_patch_size (Sequence[int]): spatial window size for inference

    Returns:
        torch.Tensor: prediction obtained by:
            - using argmax (computes maximum value along the class dimension)   
            - casting the tensor to torch.uint8 (byte) and scaling to 255 for visualization
    """
    image, _ = transform_image(image_path)
    cropped_image = extract_largest_component_bbox_image(image.unsqueeze(0), lab=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_cropped_image = torch.tensor(cropped_image).to(device)
    logits = pred_function(tensor_cropped_image.float(), model, pred_patch_size)
    pred = torch.argmax(logits, dim=1).byte().squeeze(dim=1)
    # pred = (pred * 255).byte()
    return pred


def get_prediction(
        file: str, 
        unet: Callable,
        pred_patch_size: Sequence[int], 
        save_path: str, 
        labels: Sequence[int],
        binary_preds: bool,
        frg_class: int):
    """
    Convert the prediction to a binary segmentation mask and saves the image in the ``save_path`` filepath specified in the configuration file.

    Args:
        file (str): input image filepath
        unet (Callable): trained callable model
        pred_patch_size (Sequence[int]): spatial window size for inference
        save_path (str): path in which the predictions will be saved
        labels (Sequence[int]): the labels used for annotating the groundtruth
        binary_preds (bool): generate binary predictions (e.g. root vs background) or keep all class labels
        frg_class (int): value of the foreground class when using binary segmentation masks
    """

    prediction = predict_step(file, unet, pred_patch_size).squeeze(0)
    prediction = MapImage(prediction, labels, reverse=True)
    pred = prediction.cpu().numpy().squeeze().astype(np.uint8)
    # pred_img, mask = elliptical_crop(pred, 1000, 1500, width=1400, height=2240)
    if binary_preds:
        binary_mask = createBinaryAnnotation(pred, frg_class=frg_class).astype(np.uint8)
        io.imsave(os.path.join(save_path, os.path.basename(file).split('.')[0] + ".png"), binary_mask, check_contrast=False)
    else:
        io.imsave(os.path.join(save_path, os.path.basename(file).split('.')[0] + ".png"), pred, check_contrast=False)


def predict_model(args: Dict):
    """
    Compile all functions above to run inference on a list of images

    Args:
        args (Dict): arguments specified in the configuration file
    """

    pred_data_dir = args['pred_data_dir']
    save_path = args['save_path']
    labels = args['labels']
    binary_preds = args['binary_preds']
    frg_class = args['frg_class']
    
    os.makedirs(save_path, exist_ok=True)
    
    # Looping through all ecofab folders in the pred_data_dir directory
    for ecofab in sorted(os.listdir(pred_data_dir)):
        if not ecofab.startswith("."):

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            unet = Unet2D.load_from_checkpoint(args['model_path']).to(device)
            unet.eval()

            if os.path.isdir(os.path.join(pred_data_dir, ecofab)): #if you have a timeseries of one ecofab 
                os.makedirs(os.path.join(save_path, ecofab), exist_ok=True)
                lst_files = sorted(os.listdir(os.path.join(pred_data_dir, ecofab)))
                for file in tqdm(lst_files):
                    if not file.startswith("."):
                        print("Predicting for {}".format(file))
                        file_path = os.path.join(pred_data_dir, ecofab, file)
                        get_prediction(file_path, unet, args['pred_patch_size'], os.path.join(save_path, ecofab), labels, binary_preds, frg_class)
            else:
                print("Predicting for {}".format(ecofab))
                file_path = os.path.join(pred_data_dir, ecofab)
                get_prediction(file_path, unet, args['pred_patch_size'], save_path, labels, binary_preds, frg_class) 

def main():

    parser = argparse.ArgumentParser(conflict_handler='resolve', description='Run inference using trained model')
    parser.add_argument("--config_file", type=str,
                        default="setup_files/setup-predict.json",
                        help="json file training data parameters")
    parser.add_argument("--gpus", type=int, default=None, help="how many gpus to use")
    args = parser.parse_args()
    args = _parse_training_variables(args)

    predict_model(args)

if __name__ == '__main__':
    main()
