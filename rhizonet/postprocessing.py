"""
Script for applying postprocessing methods to inferred predictions using a the convex hull method as well as morphological operations. 

Usage:
    postprocess_rhizonet --config_file ./setup_files/setup-processing.json 
"""

import os
import numpy as np
from skimage import io, util, color
import argparse
from tqdm import tqdm
import json
import torch
from typing import List, Union, Sequence, Tuple

from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)
from skimage.measure import label
from skimage.morphology import convex_hull_image, disk

def _parse_training_variables(argparse_args):
    """ 
    Parse processing variables from the JSON postprocessing configuration file and command-line arguments.

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
    if args['gpus'] is None:
        args['gpus'] = -1 if torch.cuda.is_available() else 0

    return args

def getLargestCC(segments: np.ndarray) -> np.ndarray:
    """
    Find the largest connected component within a 2D image 

    Args:
        segments (np.ndarray): A 2D binary or multi-class array (NumPy array) where labeled regions are to be identified.
            - Binary image: Where non-zero values represent objects (foreground), and zero values represent the background.
            - Multi-class image: Where different integer values represent distinct classes.

    Returns:
        np.ndarray: A 2D  mask corresponding to the largest object.
    """
    labels = label(segments)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=segments.flat))
    return largestCC


def maxProjection(limg: List[np.ndarray], ndown: int =1) -> np.ndarray:
    """
    Apply maximum projection to a list of 2D slices.
    The images in the list limg are downsampled by ndown before the maximum intensity projection is computed

    Args:
        limg (List[np.ndarray]): list of 2D images. 
        ndown (int, optional): integer parameter that controls the downsampling factor for the 2D images along dimensions x and y when performing the maximum projection. Defaults to 1 so no downsampling.

    Returns:
        np.ndarray: 2D projected image. 
    """
    #max projection aka. z-max
    IM_MAX = limg[0][::ndown,::ndown]
    for n in np.arange(1, len(limg),ndown): #ndown here for low variation on Z
        IM_MAX = np.maximum(IM_MAX, (limg[n][::ndown,::ndown]))
    return IM_MAX


def processing(data_path: str, 
               output_path: str, 
               area_opening_param: int = 500, 
               area_closing_param: int = 200, 
               disk_radius: int = 4):
    """
    Perform post-processing on predictions for noise removal, using the convex hull method and morphological operations. 

    Args:
        data_path (str): Data directory
        output_path (str): Directory where processed predictions will be saved
        area_opening_param (int, optional): minimum area of objects to retain when (eroding, dilating). Defaults to 500.
        area_closing_param (int, optional): minimum area of objects to fill when (dilating, eroding). Defaults to 200.
        disk_radius (int, optional): radius of the disk object to use when applying morphological operations. Defaults to 4.

    Returns: None
    """
    element = disk(disk_radius)

    for e in tqdm(sorted([e for e in os.listdir(data_path) if not e.startswith(".")])):
        pred_data = os.path.join(data_path, e)
        pred_chull_dir = os.path.join(output_path, e)

        if os.path.isdir(pred_data):
            os.makedirs(pred_chull_dir, exist_ok=True)

            # Get hull convex shape of overlapped images for one ecofolder
            lst_img = []
            for file in sorted([e for e in os.listdir(pred_data) if not e.startswith(".")]):
                print("Processing {}".format(file))

                path = os.path.join(pred_data, file)
                img = io.imread(path)
                lst_img.append(img)
            proj_img = maxProjection(lst_img)
        else:
            print("Processing {}".format(e))

            os.makedirs(output_path, exist_ok=True)
            img = io.imread(pred_data)
            proj_img = maxProjection([img])
            
        # apply morphological dilation
        dilated_img = dilation(proj_img, element)

        lcomponent = getLargestCC(dilated_img)
        chull = convex_hull_image(lcomponent)

        if os.path.isdir(pred_data): #if you have a timeseries of one ecofab 

            for j, file in enumerate(sorted([e for e in os.listdir(pred_data) if not e.startswith(".")])):
                path = os.path.join(pred_data, file)
                img = io.imread(path)
    
                result = np.zeros_like(img)
                result[chull == 1] = img[chull == 1]
    
                # apply morphological operations (area opening on area closing)
                pred = area_opening(area_closing(result, area_closing_param), area_opening_param)
    
                # apply additional erosion to obtain thinner roots
                pred_eroded = erosion(pred, disk(2))
                io.imsave(os.path.join(pred_chull_dir, file), pred_eroded, check_contrast=False)
                
        else: # if you have one image for a given ecofab instead of a timeseries
            result = np.zeros_like(img)
            result[chull == 1] = img[chull == 1]

            # apply morphological operations (area opening on area closing)
            pred = area_opening(area_closing(result, area_closing_param), area_opening_param)
            # apply additional erosion to obtain thinner roots
            pred_eroded = erosion(pred, disk(2))
            io.imsave(pred_chull_dir, pred_eroded, check_contrast=False)


def main():
    parser = argparse.ArgumentParser(conflict_handler='resolve', description="Arguments for applying Convx Hull post processing to a set of binary masks")
    parser.add_argument("--config_file", type=str,
                        default="./setup_files/setup-processing.json",
                        help="json file training data parameters")
    parser.add_argument("--gpus", type=int, default=None, help="how many gpus to use")

    args = parser.parse_args()
    args = _parse_training_variables(args)

    processing(args['data_path'], args['output_path'], args['area_opening'], args['area_closing'],args['disk_radius'])


if __name__ == '__main__':
    main()