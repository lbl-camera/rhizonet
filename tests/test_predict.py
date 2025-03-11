import os
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from skimage import io
from PIL import Image
import pytest
from torchvision.transforms import ToPILImage
import sys


if os.path.exists("/Users/zsordo/Desktop/local.txt"):
    sys.path.append("/Users/zsordo/Desktop/rhizonet")


from rhizonet.predict import transform_image, predict_step, get_prediction, predict_model, pred_function
from rhizonet.unet2D import Unet2D


def test_transform_image():
    """Test the image transformation logic."""
    test_img_path = "test_image.png"
    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    Image.fromarray(test_img).save(test_img_path)

    img, img_path = transform_image(test_img_path)

    assert img.shape == (3, 256, 256), "Image transformation failed."
    assert img_path == test_img_path, "Image path mismatch."

    os.remove(test_img_path)


def test_get_prediction():
    """Test the binary mask generation and saving."""
    test_file = "test_image.png"
    save_path = "test_output"
    os.makedirs(save_path, exist_ok=True)

    topil = ToPILImage()
    test_img = torch.rand(3, 256, 256, dtype=torch.float32)
    test_img = topil(test_img)
    test_img.save(test_file)

    mock_model = MagicMock()
    mock_model.return_value = torch.randn(1, 3, 128, 128)  # Mock logits output

    get_prediction(test_file, mock_model, (128, 128), save_path, labels=[1, 0], binary_preds=True, frg_class=255)

    saved_file = os.path.join(save_path, os.path.basename(test_file).split('.')[0] + ".png")
    assert os.path.exists(saved_file), "Prediction output not saved."

    os.remove(test_file)
    os.remove(saved_file)
    os.rmdir(save_path)

def test_get_prediction_grayscale():
    """Test the binary mask generation and saving."""
    test_file = "test_image.png"
    save_path = "test_output"
    os.makedirs(save_path, exist_ok=True)

    topil = ToPILImage()
    test_img = torch.rand(256, 256, dtype=torch.float32)
    test_img = topil(test_img)
    test_img.save(test_file)

    mock_model = MagicMock()
    mock_model.return_value = torch.randn(1, 1, 128, 128)  # Mock logits output

    get_prediction(test_file, mock_model, (128, 128), save_path, labels=[1, 0], binary_preds=True, frg_class=255)

    saved_file = os.path.join(save_path, os.path.basename(test_file).split('.')[0] + ".png")
    assert os.path.exists(saved_file), "Prediction output not saved."

    os.remove(test_file)
    os.remove(saved_file)
    os.rmdir(save_path)



if __name__ == "__main__":
    pytest.main(["-v", __file__])
