import os
import numpy as np
import pytest
from skimage.morphology import disk, dilation, erosion, area_closing, area_opening, convex_hull_image
from skimage.io import imsave, imread
from tempfile import TemporaryDirectory
import sys
if os.path.exists("/Users/zsordo/Desktop/local.txt"):
    sys.path.append("/Users/zsordo/Desktop/rhizonet")
from rhizonet.postprocessing import getLargestCC, maxProjection, processing


def test_getLargestCC():
    binary_image = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [1, 1, 0, 0]
    ])
    
    largest_cc = getLargestCC(binary_image)

    expected_output = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=bool)
    print(largest_cc)

    assert np.array_equal(largest_cc, expected_output), "Largest connected component is incorrect"


def test_maxProjection():
    images = [
        np.array([[0, 1], [2, 3]]),
        np.array([[4, 0], [1, 5]])
    ]

    max_proj = maxProjection(images)
    
    expected_output = np.array([[4, 1], [2, 5]])

    assert np.array_equal(max_proj, expected_output), "Max projection is incorrect"


def test_processing():
    with TemporaryDirectory() as data_dir, TemporaryDirectory() as output_dir:
        # Create dummy input images
        img1 = np.zeros((10, 10), dtype=np.uint8)
        img2 = np.zeros((10, 10), dtype=np.uint8)
        img1[3:7, 3:7] = 1  # Square in the center
        img2[2:8, 2:8] = 1  # Larger square

        os.makedirs(data_dir, exist_ok=True)
        imsave(os.path.join(data_dir, "img1.png"), img1)
        imsave(os.path.join(data_dir, "img2.png"), img2)

        # Run processing
        processing(data_dir, output_dir, area_opening_param=5, area_closing_param=5, disk_radius=2)

        # Validate output
        output_img1 = imread(os.path.join(output_dir, "img1.png"))
        output_img2 = imread(os.path.join(output_dir, "img2.png"))

        # Ensure the convex hull is applied and postprocessing is performed
        assert output_img1.sum() < img1.sum(), "Postprocessing failed for img1"
        assert output_img2.sum() < img2.sum(), "Postprocessing failed for img2"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
