import os
import json
import csv
import numpy as np
from argparse import ArgumentParser
import torch
import glob
import pytorch_lightning as pl
from utils import get_weights, transform_pred_to_annot, transform_annot, elliptical_crop, createBinaryAnnotation
from skimage import io, color
import metrics

from pytorch_lightning.strategies import DDPStrategy
from unet2D import ImageDataset, PredDataset2D, Unet2D
from simpleLogger import mySimpleLogger
from monai.data import list_data_collate
from pytorch_lightning.loggers import NeptuneLogger, WandbLogger
# from lightning.pytorch.loggers import WandbLogger 


def main():
    parser = ArgumentParser(conflict_handler='resolve')
    parser.add_argument("--config_file", type=str,
                        default="./setup_files/setup-unet2d.json",
                        help="json file contraining data parameters")
    parser.add_argument("--gpus", type=int, default=1, help="how many gpus to use")
    parser.add_argument("--strategy", type=str, default='ddp', help="pytorch strategy")

    args = parser.parse_args()

    # get vars from JSON files
    args, dataset_params, model_params = _parse_training_variables(args)
    data_dir, log_dir = model_params['data_dir'], model_params['log_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    images_dir, label_dir = data_dir + "/images", data_dir + "/labels"
    split_json = "/home/zsordo/rhizonet-fovea/data/split_patches64_wex10ex11.json"
    with open(split_json, 'r') as file:
        data = json.load(file)
    train_images = [os.path.join(images_dir, e) for e in data['train']['images']]
    train_labels = [os.path.join(label_dir, e) for e in data['train']['labels']]
    val_images = [os.path.join(images_dir, e) for e in data['val']['images']]
    val_labels = [os.path.join(label_dir, e) for e in data['val']['labels']]
    test_images = [os.path.join(images_dir, e) for e in data['test']['images']]
    test_labels = [os.path.join(label_dir, e) for e in data['test']['labels']]


    # train_images = [os.path.join(images_dir, e) for e in data['train']]
    # train_labels = [os.path.join(label_dir, "Annotation" + e.replace("tif", "png")) for e in data['train']]
    # val_images = [os.path.join(images_dir, e) for e in data['val']]
    # val_labels = [os.path.join(label_dir, "Annotation" + e.replace("tif", "png")) for e in data['val']]
    # test_images = [os.path.join(images_dir, e) for e in data['test']]
    # test_labels = [os.path.join(label_dir, "Annotation" + e.replace("tif", "png")) for e in data['test']]


    # create datasets
    train_dataset = ImageDataset(train_images, train_labels, dataset_params, training=True)
    val_dataset = ImageDataset(val_images, val_labels, dataset_params,)
    test_dataset = ImageDataset(test_images, test_labels, dataset_params,)

    
    # initialise the LightningModule
    unet = Unet2D(train_dataset, val_dataset, **model_params)

    # set up loggers and checkpoints
    # my_logger = mySimpleLogger(log_dir=log_dir,
    #                            keys=['val_acc', 'val_prec', 'val_recall', 'val_iou'])

    # neptune_logger = NeptuneLogger(
    #     project="zsordo/Rhizonet",
    #     api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5OTVlOGY4ZC05MmNjLTRiNTItOTU0Yy0wMzUxN2UyNDk4NmMifQ==",
    #     log_model_checkpoints=False,
    # )
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

    # initialise Lightning's trainer. (put link to pytorch lightning)
    trainer = pl.Trainer(
        default_root_dir=log_dir,
        callbacks=[checkpoint_callback, lr_monitor, stopping_callback],
        log_every_n_steps=1,
        enable_checkpointing=True,
        logger=wandb_logger,
        accelerator='gpu',
        devices=args['gpus'],
        strategy=args['strategy'],
        num_sanity_val_steps=0,
        max_epochs=model_params['nb_epochs']
    )

    # train
    trainer.fit(unet)
    
    # test_masks
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        collate_fn=list_data_collate, num_workers=model_params["num_workers"],
        persistent_workers=True, pin_memory=torch.cuda.is_available())
    trainer.test(unet, test_loader, ckpt_path='best', verbose=True)

 # predict and save
    # predict_dataset = PredDataset2D(model_params['pred_data_dir'], dataset_params)
    split_json = "/home/zsordo/rhizonet-fovea/data/split_wlab_wex10ex11.json"
    with open(split_json, 'r') as file:
        pred_data = json.load(file)
    pred_images = [os.path.join(model_params['pred_data_dir'], e) for e in pred_data['test']['images']]
    predict_dataset = PredDataset2D(pred_images, dataset_params)

    predict_loader = torch.utils.data.DataLoader(
        predict_dataset, batch_size=model_params['batch_size'], shuffle=False,
        collate_fn=list_data_collate, num_workers=model_params["num_workers"],
        persistent_workers=True, pin_memory=torch.cuda.is_available())
    predictions = trainer.predict(unet, predict_loader)
    # save predictions
    pred_path = os.path.join(log_dir, 'predictions')
    pred_lab_path = os.path.join(model_params['pred_data_dir'], "labels")

    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
    for (pred, _, fname) in predictions:
        pred = transform_pred_to_annot(pred.numpy().squeeze().astype(np.uint8))
        fname = os.path.basename(fname[0]).replace('tif', 'png')
        pred_img, mask = elliptical_crop(pred, 1000, 1500, width=1400, height=2240)
        # binary_mask = mask.numpy().squeeze().astype(np.uint8)
        binary_mask = createBinaryAnnotation(mask).numpy().squeeze().astype(np.uint8)
        io.imsave(os.path.join(pred_path, fname), binary_mask, check_contrast=False)

# add metrics evaluation on full size images using metrics.py 
    metrics.main(pred_path, pred_lab_path, log_dir)


        
def _parse_training_variables(argparse_args):
    """ Merges parameters from json config file and argparse, then parses/modifies parameters a bit"""
    args = vars(argparse_args)
    # overwrite argparse defaults with config file
    with open(args["config_file"]) as file_json:
        config_dict = json.load(file_json)
        args.update(config_dict)
    dataset_args, model_args = args['dataset_params'], args['model_params']
    dataset_args['patch_size'] = tuple(dataset_args['patch_size']) # tuple expected, not list
    model_args['pred_patch_size'] = tuple(model_args['pred_patch_size']) # tuple expected, not list
    if args['gpus'] is None:
        args['gpus'] = -1 if torch.cuda.is_available() else 0

    return args, dataset_args, model_args


if __name__ == "__main__":
    main()
