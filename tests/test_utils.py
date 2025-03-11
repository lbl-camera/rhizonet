import os
import torch
import numpy as np 
import pytest
import unittest
import numpy as np
import torch
from PIL import Image

import sys
if os.path.exists("/Users/zsordo/Desktop/local.txt"):
    sys.path.append("/Users/zsordo/Desktop/rhizonet")
from rhizonet.utils import (
    extract_largest_component_bbox_image,
    get_weights,
    MapImage,
    elliptical_crop,
    get_image_paths,
    contrast_img,
    createBinaryAnnotation,
    get_biomass
)

# def test_extract_largest_cc_bbox():
#     img = torch.rand((3, 128, 128), dtype=torch.float32)
#     output = extract_largest_component_bbox_image(img, predict=True)
#     assert isinstance(output,torch.tensor)
#     assert output.ndim == 3


@pytest.mark.parametrize("include_background, expected_length", [
    (True, 3),
    # (False, 2)
])
def test_get_weights(include_background, expected_length):
    labels = torch.tensor([[0, 1, 2], [2, 0, 1]])
    classes = [0, 1, 2]
    weights = get_weights(labels, classes=classes, device='cpu', include_background=include_background)
    assert len(weights) == expected_length
    assert all(w >= 0 for w in weights)


def test_MapImage():
    allowed_values = [0, 85, 170]
    img = np.random.choice(allowed_values, size=(10, 10))
    mapped_values = [0, 1, 2]
    mapped_img = MapImage(img, original_values=allowed_values, reverse=False)
    assert mapped_img.shape == img.shape
    assert np.all(np.isin(mapped_img, mapped_values))


def test_elliptical_crop():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    cropped_image, _ = elliptical_crop(img, 50, 50, 40, 60)
    assert cropped_image.size == (100, 100)

def test_get_image_paths(tmp_path):
    temp_dir = tmp_path / "test_images"
    temp_dir.mkdir()
    for i in range(3):
        (temp_dir / f"image_{i}.jpg").write_text("")
    paths = get_image_paths(str(temp_dir))
    assert len(paths) == 3


def test_contrast_img():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    contrasted = contrast_img(img)
    assert contrasted.shape == img.shape[:-1]
    assert np.max(contrasted) <= 1.0


@pytest.mark.parametrize("img_values, expected_fg_value", [
    ([0, 1, 255], 255),
    ([0, 1], 255)
])
def test_createBinaryAnnotation(img_values, expected_fg_value):
    img = np.random.choice(img_values, (100, 100), replace=True).astype(np.uint8)
    binary_mask = createBinaryAnnotation(img, expected_fg_value)
    assert binary_mask.shape == img.shape
    assert np.all(np.isin(binary_mask, [0, expected_fg_value]))


def test_get_biomass():
    binary_img = np.zeros((100, 100), dtype=np.uint8)
    binary_img[20:40, 30:50] = 1
    biomass = get_biomass(binary_img)
    assert biomass == 400  # Area of 20x20 pixels = 400


if __name__ == "__main__":
    pytest.main(["-v", __file__])