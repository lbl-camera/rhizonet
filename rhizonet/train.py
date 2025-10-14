"""
Script for training and evaluation a residual U-Net model for image segmentation.
The training process of RhizoNet includes logging, checkpointing and metrics evaluation.

Dependencies:
- PyTorch Lightning
- MONAI
- Scikit-image
- wandb

Usage:
    pip install rhizonet
    train_rhizonet --config_file ./setup_files/setup-train.json --gpus 2 --strategy "ddp" --accelerator "gpu"
"""

import os
import json
import csv
import numpy as np
from argparse import ArgumentParser
import torch
import glob
from tqdm import tqdm
import pytorch_lightning as pl
from skimage import io, color
from argparse import Namespace
from pathlib  import Path

from monai.data import list_data_collate
from lightning.pytorch.loggers import WandbLogger
import wandb


# Import parent modules
try:
    from .utils import MapImage, createBinaryAnnotation, get_image_paths
    from .metrics import evaluate
    from .unet2D import Unet2D, ImageDataset, PredDataset2D
    from .simpleLogger import mySimpleLogger
except ImportError:
    from utils import MapImage, createBinaryAnnotation, get_image_paths
    from metrics import evaluate
    from unet2D import Unet2D, ImageDataset, PredDataset2D
    from simpleLogger import mySimpleLogger



def _parse_training_variables(argparse_args):
    """ 
    Parse and merge training variables from a JSON configuration file and command-line arguments.

    Args:
        argparse_args (Namespace): Command-line arguments parsed by argparse.

    Returns:
        tuple: Updated arguments, dataset parameters, and model parameters. 
    """
    args = vars(argparse_args)

    # overwrite argparse defaults with config file
    with open(args["config_file"]) as file_json:
        config_dict = json.load(file_json)
        args.update(config_dict)

    dataset_args, model_args = args['dataset_params'], args['model_params']
    dataset_args['patch_size'] = tuple(dataset_args['patch_size'])  # tuple expected, not list
    model_args['pred_patch_size'] = tuple(model_args['pred_patch_size'])  # tuple expected, not list

    return args, dataset_args, model_args


def train_model(args):
    """
    Train and evaluate RhizoNet on a specified dataset.

    Args:
        args (Namespace): Command-line containing: 
            - config_file (str): Path to the JSON Configuration of the model.
            - gpus (int): Number of gpu nodes to use training.
            - strategy (str): Strategy to use for training (e.g., 'ddp', 'dp')
            - accelerator (str): cpu or gpu training
    
    Returns: None

    Notes:
        - The evaluation results are saved to the file specified by 'save_path' in the configuration file 
        - Training and validation metrics are also available in the WandDB project 'rhizonet'
        - Predictions associated to the full size images specified in 'pred_data_dir' are generated after training and saved in the 'save_path' directory.
        - Metrics (accuracy, precision, recall and IOU) are evaluated on full size images specified in 'pred_data_dir' and results are saved in a metrics.json files 

    Example::
        Run this script using the following command-line if 2 GPU nodes available: 
            python train.py --gpus 2 --strategy "ddp" --config_file "./setup_files/setup-train.json"
        
        Run this script using the following command-line if 1 GPU node available: 
            python train.py --gpus 1 --strategy "dp" --config_file "./setup_files/setup-train.json"
    """


    # Load image and label filepaths 
    args, dataset_params, model_params = _parse_training_variables(args)
    image_dir, label_dir, log_dir = model_params['image_dir'], model_params['label_dir'], model_params['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    
    # Save json parameters to log directory
    with open(os.path.join(log_dir, 'training_parameters.json'), 'w') as f:
        json.dump(args, f)

    images, labels = [], []

    images = get_image_paths(image_dir)
    labels = get_image_paths(label_dir)

#    Get matching images and labels 
    lbl_extension = Path(labels[0]).suffix
    i = 0 # count number of images removed that did not appear in list of labels
    print("Checking that all filepaths in images have matching label")
    for img_path in tqdm(images):
        img_name = os.path.splitext(img_path)[0].split("/")[-1]
        lbl_name = dataset_params["label_prefix"] + img_name + lbl_extension
        if not os.path.isfile(os.path.join(label_dir, lbl_name)):
            images.remove(img_path)
            i += 1
    print("Removed {} images from the list of images after match checking".format(i))

    # Split data into training, validation and test sets
    train_len, val_len, test_len = np.cumsum(np.round(len(images) * np.array(dataset_params['data_split'])).astype(int))
    idx = np.random.permutation(np.arange(len(images)))

    train_images = [images[i] for i in idx[:train_len]]
    train_labels = [labels[i] for i in idx[:train_len]]
    val_images = [images[i] for i in idx[train_len:val_len]]
    val_labels = [labels[i] for i in idx[train_len:val_len]]
    test_images = [images[i] for i in idx[val_len:]]
    test_labels = [labels[i] for i in idx[val_len:]]
    
    # Create datasets
    train_dataset = ImageDataset(train_images, train_labels, dataset_params, training=True)
    val_dataset = ImageDataset(val_images, val_labels, dataset_params, )
    test_dataset = ImageDataset(test_images, test_labels, dataset_params, )

    # Initialize Lightning module
    unet = Unet2D(train_dataset, val_dataset, **model_params)

    # Set up logging and callbacks
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb_logger = WandbLogger(log_model="all",
                               project="rhizonet")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=log_dir,
        filename="checkpoint-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        every_n_epochs=1,
        save_weights_only=True,
        verbose=True,
        monitor="val_acc",
        mode='max')
    stopping_callback = pl.callbacks.EarlyStopping(monitor='val_loss',
                                                   min_delta=1e-3,
                                                   patience=10,
                                                   verbose=True,
                                                   mode='min')
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch', log_momentum=False)

    # Initialize PyTorch Lightning trainer
    trainer = pl.Trainer(
        # precision="16-mixed", 
        # accumulate_grad_batches=2, # for batch size 2, add these 2 lines if memory issues
        default_root_dir=log_dir,
        callbacks=[checkpoint_callback, lr_monitor, stopping_callback],
        log_every_n_steps=1,
        enable_checkpointing=True,
        logger=wandb_logger,
        accelerator=args['accelerator'],
        devices=args['gpus'],
        strategy=args['strategy'],
        num_sanity_val_steps=0,
        max_epochs=model_params['nb_epochs']
    )

    # Train the model
    trainer.fit(unet)

    # Test the model
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=model_params['batch_size'], shuffle=False,
        collate_fn=list_data_collate, num_workers=model_params["num_workers"],
        persistent_workers=True, pin_memory=torch.cuda.is_available())
    trainer.test(unet, test_loader, ckpt_path='best', verbose=True)

    # Generate predictions
    pred_img_path = os.path.join(model_params['pred_data_dir'], "images")
    pred_lab_path = os.path.join(model_params['pred_data_dir'], "labels")
    predict_dataset = PredDataset2D(pred_img_path, dataset_params)
    predict_loader = torch.utils.data.DataLoader(
        predict_dataset, batch_size=1, shuffle=False,
        collate_fn=list_data_collate, num_workers=model_params["num_workers"],
        persistent_workers=True, pin_memory=torch.cuda.is_available())
    
    predictions = trainer.predict(unet, predict_loader)
    
    # Save predictions
    pred_path = os.path.join(log_dir, 'predictions')
    os.makedirs(pred_path, exist_ok=True)

    labels = dataset_params['class_values']
    for (pred, _, fname) in predictions:
        pred = MapImage(pred, labels, reverse=True)
        pred = pred.numpy().squeeze().astype(np.uint8)
        fname = os.path.basename(fname[0]).split('.')[0] + ".png"
        # pred_img, mask = elliptical_crop(pred, 1000, 1500, width=1400, height=2240)
        if dataset_params['binary_preds']:
            binary_mask = createBinaryAnnotation(pred, dataset_params['frg_class']).astype(np.uint8)
            io.imsave(os.path.join(pred_path, fname), binary_mask, check_contrast=False)
        else:
            io.imsave(os.path.join(pred_path, fname), pred, check_contrast=False)


    # Evaluate metrics on full size test images 
    if dataset_params['binary_preds']:
        evaluate(pred_path, pred_lab_path, log_dir, task='binary', num_classes=2, frg_class = dataset_params['frg_class'])
    else:
        evaluate(pred_path, pred_lab_path, log_dir, task='multiclass', num_classes=len(labels), frg_class = dataset_params['frg_class'])
        

def main():
    parser = ArgumentParser(conflict_handler='resolve')
    parser.add_argument("--config_file", type=str,
                        default="../data/setup_files/setup-train.json",
                        help="json file training data parameters")
    parser.add_argument("--gpus", type=int, default=1, help="how many gpus to use")
    parser.add_argument("--strategy", type=str, default='ddp', help="pytorch strategy")
    parser.add_argument("--accelerator", type=str, default='gpu', help="cpu or gpu accelerator")

    args = parser.parse_args()

    train_model(args)

if __name__ == "__main__":
    main()

