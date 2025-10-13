"""
Script for creating small size patches given a specific patch size and sliding window size.
Patches are saved only if at least a certain portion of pixels within the image are annotated to account for class imbalance. 
This percentage is also specified in the configuration file.

Usage:
    pip install rhizonet
    patchify_rhizonet --config_file ./setup_files/setup-prepare.json 
"""

import os
import json
import random
from argparse import ArgumentParser
import argparse
from pathlib import Path

import numpy as np
from skimage import io, util
import os
import sys
import glob
import re
from tqdm import tqdm

from typing import Dict, Tuple, Sequence 

try:
    from .utils import get_image_paths, common_files
except ImportError:
    from utils import get_image_paths, common_files


def parse_prepare_variables(argparse_args):
    """ 
    Parse and merge training variables from a JSON configuration file and command-line arguments.

    Args:
        argparse_args (Namespace): Command-line arguments parsed by argparse.

    Returns:
        Dict: Updated arguments 
    """
    args = vars(argparse_args)
    with open(args["config_file"]) as file_json:
        config_dict = json.load(file_json)
        args.update(config_dict)

    args['train_patch_size'] = tuple(args['train_patch_size'])  
    args['step_size'] = tuple(args['step_size']) 
    return args


def prepare_patches(args: Dict):
    """
    Create small size patches with the patch size and sliding windiw size given in the ``prepare_patches`` configuration file
    Patches are saved only if at least a certain portion of pixels within the image are annotated to account for class imbalance. 
    This percentage is also specified in the configuration file.

    Args:
        args (Dict): dictionary of all processing arguments 

    Returns:
        None: save the image in the specified folder 
    """
    
    image_dir = os.path.join(args['save_path'], "images")
    label_dir = os.path.join(args['save_path'], "labels")
    for folder in [image_dir, label_dir]:
        os.makedirs(folder, exist_ok=True)

    '''For the raw data path, the images and labels paths are given in the setup json file'''
    path_images = get_image_paths(args['images_path'])
    path_labels = get_image_paths(args['labels_path'])
    endpoint_img, endpoint_lbl = Path(path_images[0]).suffix, Path(path_labels[0]).suffix 
    print("Number of images: {} \n Number of labels: {}".format(len(path_images), len(path_labels)))

    # Add condition to keep only matching files in images and labels directory
    label_prefix = args['label_prefix']
    unique_files = common_files(args['images_path'], args['labels_path'], prefix=label_prefix)
    matching_path_images = sorted([os.path.join(args['images_path'], e + endpoint_img) for e in unique_files])
    matching_path_labels = sorted([os.path.join(args['labels_path'], label_prefix + e + endpoint_lbl) for e in unique_files])
    print("Number of maching images: {} \n Number of maching labels: {}".format(len(matching_path_images), len(matching_path_labels)))

    '''
    Note: 
        The following code puts aside m images and labels given that the random split of patches in the training
        builds train/val/test sets that could each contain patches of the same image. 
        That option puts aside `unseen` full-size images.
    '''

    n = len(matching_path_images)
    if args['nb_pred_data'] is not None:
        m = int(args['nb_pred_data'])
        pred_idx = random.sample(list(range(n)), m)
    else:
        pred_idx = []
    train_idx = [e for e in range(n) if e not in pred_idx]

    for i in tqdm(train_idx):
        
        image, label = io.imread(matching_path_images[i]), io.imread(matching_path_labels[i])
        if image.shape[2] >= 3: # RGB or RGBA image
            image_patch_size = args['train_patch_size'] + (image.shape[2], ) 
            image_step_size = args['step_size'] + (3,)
        elif len(image.shape) == 2: # Grayscale image 
            image_patch_size = args['train_patch_size'] 
            image_step_size = args['step_size'] 
        else:
            raise AttributeError("image should be of RGB, RGBA or grayscale")
        
        img_crop = np.vstack(
            util.view_as_windows(image, window_shape=image_patch_size, step=image_step_size))
        img_crop = img_crop.squeeze(1)

        label_crop = np.vstack(
            util.view_as_windows(label, window_shape=args['train_patch_size'], step=args['step_size']))

        for j, (img, lab) in enumerate(zip(img_crop, label_crop)):
            if (np.count_nonzero(lab) / lab.size) > args['min_labeled']:
                f_img = os.path.basename(matching_path_images[i]).split('.')[0]
                f_label = os.path.basename(matching_path_labels[i]).split('.')[0]

                fname_img = f"{f_img}_img_{i:04d}_crop-{j:04d}.tif"
                fname_label = f"{f_label}_img_{i:04d}_crop-{j:04d}.png"

                io.imsave(os.path.join(image_dir, fname_img), util.img_as_ubyte(img), check_contrast=False)
                io.imsave(os.path.join(label_dir, fname_label), util.img_as_ubyte(lab), check_contrast=False)

    if args['nb_pred_data'] is not None:
        pred_image_dir = os.path.join(args['save_pred_path'], 'images')
        pred_label_dir = os.path.join(args['save_pred_path'], 'labels')
        os.makedirs(pred_image_dir, exist_ok=True)
        os.makedirs(pred_label_dir, exist_ok=True)

        for i in tqdm(pred_idx):
            image, label = io.imread(matching_path_images[i]), io.imread(matching_path_labels[i])
            f_img = os.path.basename(matching_path_images[i])
            f_label = os.path.basename(matching_path_labels[i])

            io.imsave(os.path.join(pred_image_dir, f_img), util.img_as_ubyte(image))
            io.imsave(os.path.join(pred_label_dir, f_label), util.img_as_ubyte(label))

    return None


def main():

    parser = ArgumentParser(conflict_handler='resolve', description="Patch cropping parameter setting")
    parser.add_argument("--config_file", type=str,
                        default="../docs/setup_files/setup-prepare.json",
                        help="json file training data parameters")
    args = parser.parse_args()
    args = parse_prepare_variables(args)

    prepare_patches(args)

if __name__ == '__main__':
    main()