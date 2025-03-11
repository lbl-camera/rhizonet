import json
import pytest
import os
import numpy as np
from skimage import io
from skimage.morphology import dilation
from unittest.mock import patch, MagicMock
from PIL import Image
from torchvision.transforms import ToPILImage
import sys
if os.path.exists("/Users/zsordo/Desktop/local.txt"):
    sys.path.append("/Users/zsordo/Desktop/rhizonet")
    
from rhizonet.unet2D import ImageDataset, tiff_reader
from rhizonet.utils import extract_largest_component_bbox_image


@pytest.fixture
def mock_images_files(tmp_path):
    image_files = [tmp_path / f"image_{i}.tif" for i in range(5)]
    label_files = [tmp_path / f"label_{i}.png" for i in range(5)]
    for img, lab in zip(image_files, label_files):
        img.touch()
        lab.touch()
    return image_files, label_files


@pytest.fixture
def synthetic_sample(tmp_path):
    """
    Create synthetic image and label sample as temporary files for testing.
    """
    image_path = tmp_path / "test_image.tif"
    label_path = tmp_path / "test_label.tif"
    image_data = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    label_data = (np.random.randint(0, 2, size=(256, 256)) * 255).astype(np.uint8)
    Image.fromarray(image_data).save(image_path)
    io.imsave(label_path, label_data)

    return {
        "image_path": str(image_path),
        "label_path": str(label_path),
        "image_data": image_data,
        "label_data": label_data,
    }


@pytest.fixture
def mock_args():
    return {
    "input_channels": 3,
    "class_values": (0, 85, 170),
    "data_split": [
      0.8,
      0.1,
      0.1
    ],
    "translate_range": 0.2,
    "rotate_range": 0.05,
    "scale_range": 0.1,
    "shear_range": 0.1,
    "patch_size": (
      64,
      64
    ),
    "image_col": "None",
    "boundingbox": True,
    "dilation": True,
    "disk_dilation": 2
    }


def test_mismatched_data_label(mock_images_files, mock_args):
    data_fnames, label_fnames = mock_images_files
    label_fnames = label_fnames[:-1]  # Remove one label to cause a mismatch
    with pytest.raises(SystemExit):
        ImageDataset(data_fnames, label_fnames, mock_args)


def test_tiff_reader(synthetic_sample):
    reader = tiff_reader(
        image_col=None,
        boundingbox=True,
        dilation=True,
        disk_dilation=3,
        keys=["image", "label"]
    )
    data_dict = {
        "image": synthetic_sample["image_path"],
        "label": synthetic_sample["label_path"],
    }

    transformed_data = reader(data_dict)
    assert "image" in transformed_data
    assert "label" in transformed_data

    # Check shapes and types
    assert transformed_data["image"].shape[0] == 3  # Channel dimension should be first
    assert isinstance(transformed_data["image"], np.ndarray)
    assert transformed_data["label"].shape[0] == 1  # Channel dimension should be first for labels
    assert isinstance(transformed_data["label"], np.ndarray)


def test_invalid_input_shape(tmp_path):

    topil = ToPILImage()
    invalid_image_path = tmp_path / "invalid_image.tif"

    invalid_image_data = (np.random.rand(256, 256) * 255).astype(np.float32)
    invalid_image_data = topil(invalid_image_data)
    invalid_image_data.save(invalid_image_path)

    invalid_label_data = np.random.randint(0, 2, (256,256)).astype(np.uint8)
    invalid_label_path =  tmp_path / "invalid_label.tif"

    io.imsave(invalid_label_path, invalid_label_data)


    reader = tiff_reader(
        image_col=None,
        boundingbox=True,
        dilation=True,
        disk_dilation=3,
        keys=["image", "label"]
    )

    data_dict = {
        "image": str(invalid_image_path),
        "label": str(invalid_label_path),  
    }

    # Expecting a ValueError for invalid input shape
    with pytest.raises(ValueError, match=r"Unexpected image shape"):
        reader(data_dict)


def test_dataset_initialization(mock_images_files, mock_args):

    data_fnames, label_fnames = mock_images_files
    dataset = ImageDataset(data_fnames, label_fnames, mock_args)

    assert dataset.Nsamples == len(data_fnames)
    assert dataset.patch_size == mock_args['patch_size']
    assert dataset.spatial_dims == len(mock_args['patch_size'])
    assert isinstance(dataset.boundingbox, bool)
    assert isinstance(dataset.dilation, bool) 


def test_transforms_training(mock_args):
    dataset = ImageDataset([], [], mock_args, training=True)
    transforms = dataset.get_data_transforms(training=True, boundingbox=None, dilation=False, disk_dilation=2)
    assert "RandFlipd" in str(transforms.transforms)
    assert "RandAffined" in str(transforms.transforms)
    assert "MapLabelValued" in str(transforms.transforms)
    assert "CastToTyped" in str(transforms.transforms)
    assert "EnsureTyped" in str(transforms.transforms)


def test_transforms_validation(mock_args):
    dataset = ImageDataset([], [], mock_args, training=False)
    transforms = dataset.get_data_transforms(training=False, boundingbox=None, dilation=1, disk_dilation=2)
    assert "RandFlipd" not in str(transforms.transforms)
    assert "RandAffined" not in str(transforms.transforms)
    assert "Resized" in str(transforms.transforms)
    assert "MapLabelValued" in str(transforms.transforms)
    assert "CastToTyped" in str(transforms.transforms)
    assert "EnsureTyped" in str(transforms.transforms)


if __name__ == "__main__":
    pytest.main(["-v", __file__])

