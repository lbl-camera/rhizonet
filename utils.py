import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
import os
from skimage.color import rgb2hsv
from skimage import exposure


def class_count(data):
    tot = sum(np.unique(data.flatten(), return_counts=True)[1])
    l_count = []
    for i in range(len(np.unique(data.flatten()))):
        l_count.append((np.unique(data.flatten(), return_counts=True)[1][i] / tot,
                        np.unique(data.flatten(), return_counts=True)[0][i]))
    return l_count


# def get_weights(labels):
#     print(np.unique(labels.flatten()/len(labels.flatten())))

#     freq = np.bincount(labels.flatten())/len(labels.flatten())
#     freq = 1/freq
#     freq = torch.tensor(freq, dtype=torch.float)
#     print(freq)
#     return freq

def get_weights(labels):
    flat_labels = labels.view(-1)
    class_counts = torch.bincount(flat_labels)
    class_weights = torch.zeros_like(class_counts, dtype=torch.float)
    class_weights[class_counts.nonzero()] = 1 / class_counts[class_counts.nonzero()]
    class_weights /= class_weights.sum()
    print("class weights {}".format(class_weights))
    return class_weights


def transform_pred_to_annot(image):
    if isinstance(image, np.ndarray):
        data = image.copy()
    else:
        data = image.detach()
    data[data == 0] == 0
    data[data == 254] = 85
    data[data == 255] = 170
    return data


# the alternative is to use MapLabelValued(["label"], [0, 85, 170],[0, 1, 2])
def transform_annot(image):
    if isinstance(image, np.ndarray):
        data = image.copy()
    else:
        data = image.detach()
    data[data == 0] == 0
    data[data == 85] = 1
    data[data == 170] = 2
    return data


def elliptical_crop(img, center_x, center_y, width, height, col=bool, ):
    # Open image using PIL
    if col:
        image = Image.fromarray(img, 'RGB')
    else:
        image = Image.fromarray(img)
    image_width, image_height = image.size

    # Create an elliptical mask using PIL
    mask = Image.new('1', (image_width, image_height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2), fill=1)

    # Convert the mask to a PyTorch tensor
    mask_tensor = TF.to_tensor(mask)

    # Apply the mask to the input image using element-wise multiplication
    cropped_image = TF.to_pil_image(torch.mul(TF.to_tensor(image), mask_tensor))

    return image, cropped_image


def get_image_paths(dir):
    image_files = []
    for root, directories, files in os.walk(dir):
        for filename in files:
            if not filename.startswith(".DS_Store"):
                image_files.append(os.path.join(root, filename))  # hardcoding
    return image_files


def contrast_img(img):
    # HSV image
    hsv_img = rgb2hsv(img)  # 3 channels
    # select 1channel
    img = hsv_img[:, :, 0]
    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))
    # Equalization
    img = exposure.equalize_hist(img)
    # Adaptive Equalization
    img = exposure.equalize_adapthist(img, clip_limit=0.03)
    return img