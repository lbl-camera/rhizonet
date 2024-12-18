import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
import os
from skimage.color import rgb2hsv
from skimage import exposure
from skimage import io, filters, measure
from scipy import ndimage as ndi

def extract_largest_component_bbox_image(img, lab=None, predict=False):
    # Load and preprocess the image
    if predict:
        img = img.cpu().numpy()
        image = img[0, 2, ...]
    else:
        image = img[2, :, :]
    
    image = ndi.gaussian_filter(image, sigma=2)

    # Threshold the image
    threshold = filters.threshold_isodata(image)
    binary_image = image < threshold

    # Label connected components
    label_image = measure.label(binary_image)

    # Measure properties of the connected components
    props = measure.regionprops(label_image)

    # Find the largest connected component by area
    if props:
        largest_component = max(props, key=lambda x: x.area)
        largest_component_mask = label_image == largest_component.label
    else:
        largest_component_mask = np.zeros_like(binary_image, dtype=bool)

    # Fill all holes in the largest connected component
    filled_largest_component_mask = ndi.binary_fill_holes(largest_component_mask)

    # Get the bounding box of the largest connected component
    min_row, min_col, max_row, max_col = largest_component.bbox

    # Crop the ORIGINAL image to the bounding box dimensions
    cropped_image = img[..., min_row:max_row, min_col:max_col]
    # Create a new image with the cropped content
    new_image = np.zeros_like(cropped_image)
    new_image[..., filled_largest_component_mask[min_row:max_row, min_col:max_col]] = cropped_image[...,
        filled_largest_component_mask[min_row:max_row, min_col:max_col]]
    
    if lab is not None:
        cropped_label = lab[..., min_row:max_row, min_col:max_col]
        new_label = np.zeros_like(cropped_label)
        new_label[..., filled_largest_component_mask[min_row:max_row, min_col:max_col]] = cropped_label[...,
        filled_largest_component_mask[min_row:max_row, min_col:max_col]]
        return new_image, new_label
    else:
        if predict:
            return torch.Tensor(new_image).to('cuda')
        else:
            return new_image
        

def class_count(data):
    tot = sum(np.unique(data.flatten(), return_counts=True)[1])
    l_count = []
    for i in range(len(np.unique(data.flatten()))):
        l_count.append((np.unique(data.flatten(), return_counts=True)[1][i] / tot,
                        np.unique(data.flatten(), return_counts=True)[0][i]))
    return l_count



def compute_class_weights(labels, classes, device, include_background=False, ):
    """Compute class weights based on the presence of classes in the labels."""
    # Ensure labels are on the correct device
    labels = labels.to(device)

    batch_size = labels.size(0)
    if not include_background:
        classes.remove(0)

    n = len(classes)
    class_counts = torch.zeros(n, device=device)
    for c in range(n):
        if not include_background:
            class_counts[c] = (labels == c + 1).sum().float()
        else:
            class_counts[c] = (labels == c).sum().float()
            
    total_pixels = labels.numel() // batch_size
    class_weights = total_pixels / (class_counts + 1e-6)  # Add small value to avoid division by zero

    # Normalize weights to sum to the number of classes
    class_weights = class_weights / class_weights.sum() * len(classes)
    
    print("class weights {}".format(class_weights))
    return class_weights


def get_weights(labels, classes, device, include_background=False,):
    labels = labels.to(device)
    if not include_background:
        classes.remove(0)

    flat_labels = labels.view(-1)
    n = len(classes)
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


# def transform_annot(image):
#     if isinstance(image, np.ndarray):
#         data = image.copy()
#     else:
#         data = image.detach()
#     data[data == 0] == 0
#     data[data == 85] = 1
#     data[data == 170] = 2
#     return data

# the alternative is to use MapLabelValued(["label"], [0, 85, 170],[0, 1, 2])
def transform_annot(image, value_map):
    if isinstance(image, np.ndarray) :      
        data = image.copy()
    else:
        data = image.detach()
    for key in value_map.keys():
        data[data == int(key)] = int(value_map[key])
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


def createBinaryAnnotation(img):
    '''Find all the annotations that are root, then NOT root, then combine'''
    if isinstance(img, torch.Tensor):
        u = torch.unique(img)
        bkg = torch.zeros(img.shape)  # background
        try: 
            frg = (img == u[2]).int() * 255
        except: 
            frg = (img == u[1]).int() * 255    
    elif isinstance(img, np.ndarray):
        u = np.unique(img)
        bkg = np.zeros(img.shape)  # background
        try: 
            frg = (img == u[2]).astype(int) * 255
        except: 
            frg = (img == u[1]).astype(int) * 255    
    else:
        raise TypeError("Input should be a PyTorch tensor or a NumPy array.")
    return bkg + frg