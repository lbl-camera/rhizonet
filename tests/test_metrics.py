import os
import json
import pytest
import torch
import numpy as np
from tempfile import TemporaryDirectory
from skimage import io
import sys
if os.path.exists("/Users/zsordo/Desktop/local.txt"):
    sys.path.append("/Users/zsordo/Desktop/rhizonet")
    
from rhizonet.metrics import calculate_all_metrics, evaluate

# Sample prediction and groundtruth data
@pytest.fixture
def sample_data():
    pred = torch.tensor([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 1]
    ])
    target = torch.tensor([
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 1]
    ])
    return pred, target

# Test the metric calculation function
def test_calculate_all_metrics(sample_data):
    pred, target = sample_data
    acc, prec, rec, iou, dice = calculate_all_metrics(pred, target, task="binary", num_classes=2)

    # Expected results
    expected_acc = 15/16 
    expected_prec = 6/7
    expected_rec = 1
    expected_iou = 6/7
    expected_dice = 12/13

    assert acc == pytest.approx(expected_acc, rel=1e-2)
    assert prec == pytest.approx(expected_prec, rel=1e-2)
    assert rec == pytest.approx(expected_rec, rel=1e-2)
    assert iou == pytest.approx(expected_iou, rel=1e-2)
    assert dice == pytest.approx(expected_dice, rel=1e-2)


# Test the evaluation function
def test_evaluate():
    # Create temporary directories for predictions, labels, and logs
    with TemporaryDirectory() as pred_dir, TemporaryDirectory() as label_dir, TemporaryDirectory() as log_dir:
        # Create dummy predictions and labels
        pred1 = torch.tensor([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1]
        ], dtype=torch.uint8)*255
        label1 = torch.tensor([
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1]
        ], dtype=torch.uint8)*255
        pred_path1 = os.path.join(pred_dir, "pred1.png")
        label_path1 = os.path.join(label_dir, "label1.png")
        io.imsave(pred_path1, pred1)
        io.imsave(label_path1, label1)

        evaluate(pred_dir, label_dir, log_dir, task="binary", num_classes=2)

        # Check that the metrics file is created
        metrics_path = os.path.join(log_dir, "metrics.json")
        assert os.path.exists(metrics_path)

        # Load and verify the metrics
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        assert "pred1.png" in metrics

        # Expected results
        expected_acc = 15/16 
        expected_prec = 6/7
        expected_rec = 1
        expected_iou = 6/7
        expected_dice = 12/13

        assert metrics["pred1.png"]["accuracy"] == pytest.approx(expected_acc, rel=1e-2)
        assert metrics["pred1.png"]["precision"] == pytest.approx(expected_prec, rel=1e-2)
        assert metrics["pred1.png"]["recall"] == pytest.approx(expected_rec, rel=1e-2)
        assert metrics["pred1.png"]["IOU"] == pytest.approx(expected_iou, rel=1e-2)
        assert metrics["pred1.png"]["Dice"] == pytest.approx(expected_dice, rel=1e-2)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
