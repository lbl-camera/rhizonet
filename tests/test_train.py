import pytest
import os 
import numpy as np 
import torch
import tempfile

from argparse import Namespace
from argparse import ArgumentParser
from unittest.mock import Mock, MagicMock
from unittest.mock import patch
import wandb
from pytorch_lightning.loggers import WandbLogger
import sys
if os.path.exists("/Users/zsordo/Desktop/local.txt"):
    sys.path.append("/Users/zsordo/Desktop/rhizonet")
from rhizonet.train import train_model, _parse_training_variables
from rhizonet.unet2D import Unet2D
import pytorch_lightning as pl


def test_unet2d_initialization():
    train_ds = MagicMock()  # Mock training dataset
    val_ds = MagicMock()    # Mock validation dataset
    
    model_params = {
    "input_channels": 3,
    "class_values": [0, 85, 170],
    "spatial_dims": 2, 
    "pred_patch_size": (
      64,
      64
    ),
    "task": "multiclass",
    "background_index": 0,
    "num_classes": 3,
    "model": "resnet",
    "batch_size": 1,
    "lr": 3e-4,
    "num_workers": 32,
    "nb_epochs": 1
    }

    # Valid initialization
    model = Unet2D(train_ds, val_ds, **model_params)
    assert isinstance(model, Unet2D), "Model initialization failed"
    
    # Invalid model type
    with pytest.raises(AttributeError):
        Unet2D(train_ds, val_ds, model='invalid_model_type', 
               num_classes=3, 
               pred_patch_size=(64, 64), 
               spatial_dims=2,
               class_values=[0, 85, 170],
               task='multiclass')


def test_forward_pass():
    train_ds = MagicMock()
    val_ds = MagicMock()

    # Test for resnet 
    model = Unet2D(train_ds, val_ds, model='resnet', 
                input_channels=3,
               num_classes=3, 
               pred_patch_size=(64, 64), 
               spatial_dims=2,
               class_values=[0, 85, 170],
               task='multiclass')
    
    input_tensor = torch.randn(1, 3, 64, 64)  # Batch size 1, 3 channels, 64x64 image
    output = model.forward(input_tensor)
    
    assert output.shape == (1, 3, 64, 64), f"Unexpected output shape: {output.shape}"

    # Test for swin
    model = Unet2D(train_ds, val_ds, model='swin', 
                input_channels=3,
               num_classes=3, 
               pred_patch_size=(64, 64), 
               spatial_dims=2,
               class_values=[0, 85, 170],
               task='multiclass')
    
    input_tensor = torch.randn(1, 3, 64, 64)  # Batch size 1, 3 channels, 64x64 image
    output = model.forward(input_tensor)
    
    assert output.shape == (1, 3, 64, 64), f"Unexpected output shape: {output.shape}"


def test_training_validation_step():
    # Mock the wandb logger
    with tempfile.TemporaryDirectory() as temp_dir:
        wandb.init(dir=temp_dir, project="test_project", mode="offline")
        wandb_logger = WandbLogger(project="test_project", save_dir=temp_dir)


        train_ds = MagicMock()
        val_ds = MagicMock()
        model = Unet2D(train_ds, val_ds, model='resnet', 
                    input_channels=3,
                    num_classes=3, 
                    pred_patch_size=(64, 64), 
                    spatial_dims=2,
                    class_values=[0, 85, 170],
                    lr=3e-4,
                    batch_size=4,
                    task='multiclass',
                    num_workers=1,
                    logger=wandb_logger)
        model.has_executed = True
        model.log = MagicMock()

        batch = {"image": torch.randn(4, 3, 64, 64), "label": torch.randint(0, 3, (4, 64, 64))}
        optimizer = model.configure_optimizers()[0][0]

        optimizer.zero_grad()
        loss = model.training_step(batch, batch_idx=0)
        loss.backward()
        optimizer.step()
        assert loss.item() > 0, "Loss should be greater than 0"

        val_logs = model.validation_step(batch, batch_idx=0)
        assert "loss" in val_logs, "Validation step should log loss"

        if os.path.exists("train_tensor.pt"):
            os.remove("train_tensor.pt")
        if os.path.exists("network_graph.png"):
            os.remove("network_graph.png")


def test_prediction_function():
    train_ds = MagicMock()
    val_ds = MagicMock()
    model = Unet2D(train_ds, val_ds, model='resnet', 
                input_channels=3,
                num_classes=3, 
                pred_patch_size=(64, 64), 
                spatial_dims=2,
                class_values=[0, 85, 170],
                task='multiclass')
    
    input_image = torch.randn(1, 3, 256, 256)  
    preds = model.pred_function(input_image)
    
    assert preds.shape == (1, 3, 256, 256), f"Unexpected prediction shape: {preds.shape}"


def test_logging():
    with tempfile.TemporaryDirectory() as temp_dir:
        # wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(dir=temp_dir, project="test_project", mode="offline")
        
        wandb_logger = WandbLogger(project="test_project", save_dir=temp_dir)

        model = Unet2D(train_ds=None, val_ds=None, model='resnet', 
                input_channels=3,
                num_classes=3, 
                pred_patch_size=(64, 64), 
                spatial_dims=2,
                class_values=[0, 85, 170],
                task='multiclass',
                logger=wandb_logger)
        model.log = MagicMock()

        model._compute_cnf_stats = MagicMock(return_value=(0.9, 0.85, 0.8, 0.75, 0.824))

        model.on_validation_epoch_end()

        # Check that logger.log is called with correct values
        model.log.assert_any_call('val_acc', 0.9, prog_bar=True, sync_dist=True)
        model.log.assert_any_call('val_precision', 0.85, prog_bar=False, sync_dist=True)
        model.log.assert_any_call('val_recall', 0.8, prog_bar=False, sync_dist=True)
        model.log.assert_any_call('val_iou', 0.75, prog_bar=True, sync_dist=True)
        model.log.assert_any_call('val_dice', 0.824, prog_bar=True, sync_dist=True)


if __name__ == "__main__":
    pytest.main(["-v", __file__])


