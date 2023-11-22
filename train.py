import os
import json
import csv
import numpy as np
from argparse import ArgumentParser
import torch
import glob
import pytorch_lightning as pl
from skimage import io, color

from unet2D import Unet2D, ImageDataset, PredDataset2D
from simpleLogger import mySimpleLogger
from monai.data import list_data_collate
from pytorch_lightning.loggers import NeptuneLogger


def main():
    parser = ArgumentParser(conflict_handler='resolve')
    parser.add_argument("--config_file", type=str,
                        default="/Users/zinebsordo/Desktop/berkeleylab/zineb/monai_unet2D/setup_files/setup-unet2d.json",
                        help="json file contraining data parameters")
    parser.add_argument("--device", type=str, default='cpu', choices=['cpu', 'gpu'], help="choose cpu or gpu")
    parser.add_argument("--nodes", type=int, default=1, help="number of gpu or cpu nodes")
    parser.add_argument("--strategy", type=str, default='ddp', help="pytorch strategy")

    args = parser.parse_args()

    # get vars from JSON files
    args, dataset_params, model_params = _parse_training_variables(args)
    data_dir, log_dir = model_params['data_dir'], model_params['log_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    '''The training data should be in folders names images and labels --> to specify in the readme file'''
    images_dir, label_dir = data_dir + "/images", data_dir + "/labels"
    images, labels = [], []
    # for f in os.listdir(images_dir): # if images are in subfolders e.g. in date subfolders
    images += sorted(glob.glob(os.path.join(images_dir, "*.tif")))
    labels += sorted(glob.glob(os.path.join(label_dir,  "*.png")))

    # randomly split data into train/val/test_masks
    train_len, val_len, test_len = np.cumsum(np.round(len(images) * np.array(dataset_params['data_split'])).astype(int))
    idx = np.random.permutation(np.arange(len(images)))

    train_images = [images[i] for i in idx[:train_len]]
    train_labels = [labels[i] for i in idx[:train_len]]
    val_images = [images[i] for i in idx[train_len:val_len]]
    val_labels = [labels[i] for i in idx[train_len:val_len]]
    test_images = [images[i] for i in idx[val_len:]]
    test_labels = [labels[i] for i in idx[val_len:]]
    # create datasets
    train_dataset = ImageDataset(train_images, train_labels, dataset_params, training=True)
    val_dataset = ImageDataset(val_images, val_labels, dataset_params, )
    test_dataset = ImageDataset(test_images, test_labels, dataset_params, )

    # initialise the LightningModule
    unet = Unet2D(train_dataset, val_dataset, **model_params)

    # set up loggers and checkpoints
    # my_logger = mySimpleLogger(log_dir=log_dir,
    #                            keys=['val_acc', 'val_prec', 'val_recall', 'val_iou'])

    neptune_logger = NeptuneLogger(
        project="zsordo/Rhizonet",
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5OTVlOGY4ZC05MmNjLTRiNTItOTU0Yy0wMzUxN2UyNDk4NmMifQ==",
        log_model_checkpoints=False,
    )

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

    # initialise Lightning's trainer. (put link to pytorch lightning)
    trainer = pl.Trainer(
        default_root_dir=log_dir,
        callbacks=[checkpoint_callback, lr_monitor, stopping_callback],
        log_every_n_steps=1,
        enable_checkpointing=True,
        logger=neptune_logger,
        accelerator=args['device'],
        devices=args['nodes'],
        strategy=args['strategy'],
        num_sanity_val_steps=0,
        max_epochs=model_params['nb_epochs']
    )

    # train
    trainer.fit(unet)

    # test_masks
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=model_params['batch_size'], shuffle=False,
        collate_fn=list_data_collate, num_workers=model_params["num_workers"],
        persistent_workers=True, pin_memory=torch.cuda.is_available())
    trainer.test(unet, test_loader, ckpt_path='best', verbose=True)


def _parse_training_variables(argparse_args):
    """ Merges parameters from json config file and argparse, then parses/modifies parameters a bit"""
    args = vars(argparse_args)
    # overwrite argparse defaults with config file
    with open(args["config_file"]) as file_json:
        config_dict = json.load(file_json)
        args.update(config_dict)
    dataset_args, model_args = args['dataset_params'], args['model_params']
    dataset_args['patch_size'] = tuple(dataset_args['patch_size'])  # tuple expected, not list
    model_args['pred_patch_size'] = tuple(model_args['pred_patch_size'])  # tuple expected, not list
    return args, dataset_args, model_args


if __name__ == "__main__":
    main()
