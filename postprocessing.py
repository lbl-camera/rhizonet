import os
import numpy as np
from skimage import io, util, color
import argparse
from tqdm import tqdm
import json
import torch

from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)
from skimage.measure import label
from skimage.morphology import convex_hull_image


def getLargestCC(segments):
    '''Return a mask corresponding to the largest object'''
    labels = label(segments)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=segments.flat))
    return largestCC


def maxProjection(limg, ndown=1):
    #max projection aka. z-max
    IM_MAX = limg[0][::ndown,::ndown]
    for n in np.arange(1,len(limg),ndown): #ndown here for low variation on Z
        IM_MAX = np.maximum(IM_MAX, (limg[n][::ndown,::ndown]))
    return IM_MAX


def get_hull_mask(data_path, output_path, area_opening_param=500, area_closing_param=200,):
    element = np.array([[1, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 1, 0, 0],
                        [0, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 0],
                        [0, 0, 1, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1]])

    for e in tqdm(sorted([e for e in os.listdir(data_path) if not e.startswith(".DS_Store")])):
        print("Processing ecofab {}".format(e))

        pred_data = os.path.join(data_path, e)
        pred_chull_dir = os.path.join(output_path, e)

        if not os.path.exists(pred_chull_dir):
            os.makedirs(pred_chull_dir)

        # Get hull convex shape of overlapped images for one ecofolder
        lst_img = []
        for file in sorted([e for e in os.listdir(pred_data) if not e.startswith(".")]):
            path = os.path.join(pred_data, file)
            img = io.imread(path)
            lst_img.append(img)
        proj_img = maxProjection(lst_img)

        # apply morphological dilation
        dilated_img = dilation(proj_img, element)

        lcomponent = getLargestCC(dilated_img)
        chull = convex_hull_image(lcomponent)

        for j, file in enumerate(sorted([e for e in os.listdir(pred_data) if not e.startswith(".")])):
            path = os.path.join(pred_data, file)
            img = io.imread(path)

            result = np.zeros_like(img)
            result[chull == 1] = img[chull == 1]

            # apply morphological operations (area opening on area closing)
            pred = area_opening(area_closing(result, area_closing_param), area_opening_param)
            io.imsave(os.path.join(pred_chull_dir, file), pred, check_contrast=False)


def _parse_training_variables(argparse_args):
    """ Merges parameters from json config file and argparse, then parses/modifies parameters a bit"""
    args = vars(argparse_args)
    # overwrite argparse defaults with config file
    with open(args["config_file"]) as file_json:
        config_dict = json.load(file_json)
        args.update(config_dict)
    if args['gpus'] is None:
        args['gpus'] = -1 if torch.cuda.is_available() else 0

    return args


def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument("--config_file", type=str,
                        default="/Users/zinebsordo/Desktop/berkeleylab/zineb/monai_unet2D/setup_files/setup-processing.json",
                        help="json file contraining data parameters")
    parser.add_argument("--gpus", type=int, default=None, help="how many gpus to use")

    args = parser.parse_args()
    args = _parse_training_variables(args)

    get_hull_mask(args['data_path'], args['output_path'], args['area_opening'], args['area_closing'],)


if __name__ == '__main__':
    main()
