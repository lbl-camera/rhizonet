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

from dataset import PredDataset2D, ImageDataset, tiff_reader
from monai.data import ArrayDataset, create_test_image_2d, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AddChannel,
    Compose,
    ScaleIntensityRange,
    EnsureType
)

from utils import transform_pred_to_annot
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
    img = np.array(Image.open(img_path))
    img = transform(np.transpose(img, (2, 0, 1)))
    return (img, img_path)


def pred_function(image, model, pred_patch_size):
    return sliding_window_inference(image, pred_patch_size, 1, model)


def predict_step(image_path, model, pred_patch_size):
    image, img_path = transform_image(image_path)
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


def createBinaryAnnotation(img):
    '''Find all the annotations that are root, then NOT root, then combine'''
    u = np.unique(img)
    bkg = np.zeros(img.shape)  # background
    frg = (img == u[2]).astype(int) * 255
    return bkg + frg


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
    pred_img, mask = elliptical_crop(pred, 1000, 1500, width=1400, height=2240)
    binary_mask = createBinaryAnnotation(mask).astype(np.uint8)
    io.imsave(os.path.join(pred_path, file.split(ecofab + '/')[1][:-4] + ".png"), binary_mask, check_contrast=False)

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument("--config_file", type=str,
                        default="/Users/zinebsordo/Desktop/berkeleylab/zineb/monai_unet2D/setup_files/setup-predict2d.json",
                        help="json file contraining data parameters")
    parser.add_argument("--gpus", type=int, default=None, help="how many gpus to use")
    args = parser.parse_args()

    args = _parse_training_variables(args)
    # Looping through all ecofab folders in the pred_data_dir directory
    for ecofolder in sorted(os.listdir(args['pred_data_dir'])):
        if ecofolder.startswith("eco"):

            print("Predicting for {}".format(ecofolder))

            pred_data_dir = os.path.join(args['pred_data_dir'], ecofolder)
            pred_path = os.path.join(args['pred_path'], ecofolder)

            if not os.path.exists(pred_path):
                os.makedirs(pred_path)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            unet = Unet2D.load_from_checkpoint(args['model_path']).to(device)
            unet.eval()

            lst_files = sorted(os.listdir(pred_data_dir))
            for file in tqdm(lst_files):
                if not file.startswith("."):
                    file_path = os.path.join(pred_data_dir, file)
                    get_prediction(file_path, unet, args['pred_patch_size'], pred_path, ecofolder)


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
