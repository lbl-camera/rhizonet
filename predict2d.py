import os
import glob
import argparse
import json
import re
from tqdm import tqdm
from argparse import ArgumentParser
from skimage import io, util, color
from PIL import Image
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
import matplotlib.pyplot as plt

from unet2D import Unet2D

from unet2D import PredDataset2D, ImageDataset, tiff_reader
from monai.data import ArrayDataset, create_test_image_2d, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AddChannel,
    Compose,
    ScaleIntensityRange,
    EnsureType
)

from utils import transform_pred_to_annot, extract_largest_component_bbox_image, createBinaryAnnotation
from PIL import ImageDraw
import torchvision.transforms.functional as TF
from datetime import datetime


def transform_image(img_path):
    transform = Compose(
        [
            AddChannel(),
            ScaleIntensityRange(a_min=0, a_max=255,
                                b_min=0.0, b_max=1.0, clip=True,
                                ),
            EnsureType()
        ]
    )
    img = np.array(Image.open(img_path))[...,:3]
    img = transform(np.transpose(img, (2, 0, 1)))
    return (img, img_path)


def pred_function(image, model, pred_patch_size):
    return sliding_window_inference(image, pred_patch_size, 1, model)


def predict_step(image_path, model, pred_patch_size):
    image, img_path = transform_image(image_path)
    cropped_image = extract_largest_component_bbox_image(image, predict=True)
    logits = pred_function(image, model, pred_patch_size)
    pred = torch.argmax(logits, dim=1).byte().squeeze(dim=1)
    pred = (pred * 255).byte()
    return pred


def predict_timeseries(unet, file, pred_patch_size):
    prediction = predict_step(file, unet, pred_patch_size)[0, :, :]
    pred = transform_pred_to_annot(prediction.numpy().squeeze().astype(np.uint8))
    raw = np.array(Image.open(file))
    date = file.split("/")[-1].split('_')[2] #hardcoding
    return pred, raw, date


def elliptical_crop(img, center_x, center_y, width, height):
    # Open image using PIL
    image = Image.fromarray(np.uint8(img))
    image_width, image_height = image.size

    # Create an elliptical mask using PIL
    mask = Image.new('1', (image_width, image_height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2), fill=1)

    # Convert the mask to a PyTorch tensor
    mask_tensor = TF.to_tensor(mask)

    # Apply the mask to the input image using element-wise multiplication
    cropped_image = TF.to_pil_image(torch.mul(TF.to_tensor(image), mask_tensor))

    return image, np.array(cropped_image)



def get_biomass(binary_img):
    roi = binary_img > 0
    nerror = 0
    binary_img = binary_img * roi
    biomass = np.unique(binary_img.flatten(), return_counts=True)
    try:
        nbiomass = biomass[1][1]
    except:
        nbiomass = 0
        nerror += 1
        print("Seg error in ")
    return nbiomass


'''
This following function can be used to output comparison plots between the prediction 
and the raw image instead of just the prediction 
'''


def get_prediction(file, unet, pred_patch_size, pred_path, ecofab):
    prediction = predict_step(file, unet, pred_patch_size)[0, :, :]
    pred = transform_pred_to_annot(prediction.numpy().squeeze().astype(np.uint8))
    # pred_img, mask = elliptical_crop(pred, 1000, 1500, width=1400, height=2240)
    binary_mask = createBinaryAnnotation(pred).astype(np.uint8) # Long Tensor for integers
    io.imsave(os.path.join(pred_path, os.path.basename(file).split('.')[0] + ".png"), binary_mask, check_contrast=False)

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument("--config_file", type=str,
                        default="./setup_files/setup-predict2d.json",
                        help="json file contraining data parameters")
    parser.add_argument("--gpus", type=int, default=None, help="how many gpus to use")
    args = parser.parse_args()
    args = _parse_training_variables(args)


    pred_data_dir = args['pred_data_dir']
    pred_path = args['pred_path']
    
    # Looping through all ecofab folders in the pred_data_dir directory
    for ecofab in sorted(os.listdir(pred_data_dir)):
        if not ecofab.startswith("."):

            print("Predicting for {}".format(ecofab))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            unet = Unet2D.load_from_checkpoint(args['model_path']).to(device)
            unet.eval()

            if os.path.isdir(os.path.join(pred_data_dir, ecofab)): #if you have a timeseries of one ecofab 
                os.makedirs(os.path.join(pred_path, ecofab), exist_ok=True)
                lst_files = sorted(os.listdir(os.path.join(pred_data_dir, ecofab)))
                for file in tqdm(lst_files):
                    if not file.startswith("."):
                        file_path = os.path.join(pred_data_dir, ecofab, file)
                        get_prediction(file_path, unet, args['pred_patch_size'], os.path.join(pred_path, ecofab))
            else:
                file_path = os.path.join(pred_data_dir, ecofab)
                get_prediction(file_path, unet, args['pred_patch_size'], pred_path)


def _parse_training_variables(argparse_args):
    """ Merges parameters from json config file and argparse, then parses/modifies parameters a bit"""
    args = vars(argparse_args)
    # overwrite argparse defaults with config file
    with open(args["config_file"]) as file_json:
        config_dict = json.load(file_json)
        args.update(config_dict)
    args['pred_patch_size'] = tuple(args['pred_patch_size']) # tuple expected, not list
    if args['gpus'] is None:
        args['gpus'] = -1 if torch.cuda.is_available() else 0

    return args


if __name__ == '__main__':
    main()
