import pytest
import numpy as np
import os
from argparse import Namespace
import json
from skimage import io, util
from tempfile import TemporaryDirectory
import sys
if os.path.exists("/Users/zsordo/Desktop/local.txt"):
    sys.path.append("/Users/zsordo/Desktop/rhizonet")
from rhizonet.prepare_patches import parse_prepare_variables, prepare_patches

@pytest.fixture
def sample_config():
    return {
        "images_path": "./images",
        "labels_path": "./labels",
        "save_path": "./output",
        "train_patch_size": [64, 64],
        "step_size": [32, 32],
        "min_labeled": 0.1,
        "nb_pred_data": None,
        "label_prefix": ""
    }

def test_parse_prepare_variables(sample_config):
    sample_config = {
        "images_path": "./images",
        "labels_path": "./labels",
        "save_path": "./output",
        "train_patch_size": [64, 64],
        "step_size": [32, 32],
        "min_labeled": 0.1,
        "nb_pred_data": None,
        "label_prefix": ""
    }

    with TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(sample_config, f)

        args = Namespace(config_file=config_path)
        parsed_args = parse_prepare_variables(args)

        assert parsed_args['train_patch_size'] == (64, 64)
        assert parsed_args['step_size'] == (32, 32)
        assert parsed_args['min_labeled'] == 0.1

def test_prepare_patches():
    sample_config = {
        "images_path": "./images",
        "labels_path": "./labels",
        "save_path": "./output",
        "train_patch_size": (64, 64),
        "step_size": (32, 32),
        "min_labeled": 0.1,
        "nb_pred_data": None,
        "label_prefix": "Annotation"
    }
    with TemporaryDirectory() as temp_dir:
        images_dir = os.path.join(temp_dir, "images")
        labels_dir = os.path.join(temp_dir, "labels")
        os.makedirs(images_dir)
        os.makedirs(labels_dir)

        for i in range(2):
            img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
            label = np.zeros((128, 128), dtype=np.uint8)
            label[32:96, 32:96] = 1  # Annotated region

            io.imsave(os.path.join(images_dir, f"image_{i}.png"), img)
            io.imsave(os.path.join(labels_dir, f"Annotationimage_{i}.png"), label)

        sample_config["images_path"] = images_dir
        sample_config["labels_path"] = labels_dir
        sample_config["save_path"] = os.path.join(temp_dir, "output")

        prepare_patches(sample_config)

        output_images_dir = os.path.join(sample_config['save_path'], "images")
        output_labels_dir = os.path.join(sample_config['save_path'], "labels")

        assert os.path.exists(output_images_dir)
        assert os.path.exists(output_labels_dir)

        output_images = sorted(os.listdir(output_images_dir))
        output_labels = sorted(os.listdir(output_labels_dir))

        assert len(output_images) > 0
        assert len(output_images) == len(output_labels)

        for label_file in output_labels:
            label = io.imread(os.path.join(output_labels_dir, label_file))
            assert (np.count_nonzero(label) / label.size) > sample_config['min_labeled']


if __name__ == "__main__":
    pytest.main(["-v", __file__])