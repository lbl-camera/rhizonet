import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
import os
from skimage.color import rgb2hsv
from skimage import exposure
from skimage import io, filters, measure
from scipy import ndimage as ndi
from typing import Union, List, Tuple, Sequence, Dict
from monai.transforms import MapLabelValued


def extract_largest_component_bbox_image(img: Union[np.ndarray, torch.Tensor], 
                                         lab: Union[np.ndarray, torch.Tensor] = None,) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Extract the largest connected component (LCC) of the given image and get the bounding box associated with this LCC. Depending on the input parameters, it can either:

    - Crop the image to the bounding box.
    - Apply a binary mask to the image, maintaining the same shape as the input image, with everything but the LCC set as a black background.

    This function can be applied to both the image and the annotation (label) if provided.

    Args:
        img (Union[numpy.ndarray, torch.Tensor]): 
            The input image. 
            - Can be a NumPy array or PyTorch tensor with shape (C, H, W) or (B, C, H, W) if a batch dimension is included.
            - Must not be None; if None, the function raises a ValueError.
        lab (Union[numpy.ndarray, torch.Tensor], optional): 
            The label or annotated image. Defaults to None.


    Returns:
        Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]: 
            - If `lab` is None: Returns the processed image as a NumPy array or PyTorch tensor.
            - If `lab` is provided: Returns a tuple containing:
                - The processed image (NumPy array).
                - The processed label image (NumPy array).
    """

    if img is None:
        raise ValueError("Image cannot be None.")
    elif isinstance(img, np.ndarray):
        print("Processing a NumPy array.")
    elif isinstance(img, torch.Tensor):
        print("Processing a PyTorch tensor")
        img = np.array(img)
    else:
        raise TypeError("Input must be a numpy.ndarray, torch.Tensor, or None.")
    
    # Remove dimension if there is a batch dim and format is (B, C, H, W) or (B, H, W) for grayscale images
    if len(img.squeeze().shape) >= 3 and img.squeeze().shape[0]>=3: #RGB or RGBA images after removing the batch dimension
        image = img.squeeze()
        image = image[2, ...] # choose a channel 
    else:
        image = img.squeeze()
        
    # Get the largest connected component
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

    # Create a new image with the cropped content but keeping input image shape
    new_image = np.zeros_like(img)

    # Applying the mask without cropping out the ROI 
    new_image[..., min_row:max_row, min_col:max_col] = cropped_image * filled_largest_component_mask[min_row:max_row, min_col:max_col]

    # Processing the label image
    if lab is not None:
        cropped_label = lab[..., min_row:max_row, min_col:max_col]
        new_label = np.zeros_like(lab)
        new_label[..., min_row:max_row, min_col:max_col] = cropped_label * filled_largest_component_mask[min_row:max_row, min_col:max_col]
        return new_image, new_label
    
    return new_image
        

def get_weights(
        labels: torch.Tensor, 
        classes: List[int], 
        device: str, 
        include_background=False,
        ) -> List[float]:
    """
    Computes the weights of each class in the batch of labeled images 

    Args:
        labels (torch.Tensor): Batch of labeled imagse with each pixel of an image equal to a class value. 
        classes (List[int]): List of the classes
        device (str): training device should be 'cuda' if training on GPU
        include_background (bool, optional): Boolean to include or not the background valued as 0 when calculating the weight of the classes in the images. Defaults to False.

    Returns:
        List[float]: List of the class weights (floats).
    """

    labels = labels.to(device)
    if not include_background:
        classes.remove(0)
    flat_labels = labels.view(-1)

    # FIX IF NOT ALL CLASSES ARE IN THE LABEL INPUT 
    n = len(classes)
    class_counts = torch.bincount(flat_labels, minlength=n)
    class_weights = torch.zeros(n, dtype=torch.float, device=device)
    nonzero_mask = class_counts > 0
    class_weights[nonzero_mask] = 1 / class_counts[nonzero_mask]
    class_weights /= class_weights.sum()
    # print("class weights {}".format(class_weights))

    return class_weights


def MapImage(
        image: Union[np.ndarray, torch.Tensor], 
        original_values: List[int],
        reverse: bool
        ) -> Union[np.ndarray, torch.Tensor]:
    """
    Maps the current values of a given input image to the values given by the tuple (current values, new values).

    Args:
        image (Union[np.ndarray, torch.Tensor]): The input image to transform
        original_values (List[int]): List of original values to be mapped
        reverse (bool): True if mapping the other way around back to original values

    Raises:
        TypeError: If the input image is neither a numpy array or a torch tensor

    Returns:
        Union[np.ndarray, torch.Tensor]: the transformed input after mapping.
    
    Example::
        transformed_image = MapImage(image, [0, 85, 170])
        The values will be mapped to [0, 1, 2] with the first value of the original values specified being the background index. 
    """
    if isinstance(image, np.ndarray) :      
        data = image.copy()
    elif isinstance(image, torch.Tensor):
        data = image.detach()
    else:
        raise TypeError("Input must be a numpy.ndarray, torch.Tensor")
    
    target_values = list(range(len(original_values)))
    if not reverse:
        # Create the transform
        map_label_transform = MapLabelValued(["label"], original_values, target_values)
    else:
        # Create the transform
        map_label_transform = MapLabelValued(["label"], target_values, original_values)
        
    # Apply the transform
    mapped_label_image = map_label_transform({"label": data})["label"]

    return mapped_label_image


def elliptical_crop(img: np.ndarray, 
                    center_x: int, center_y: int, 
                    width: int, 
                    height: int
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crops out an elliptical shape out of the input image and sets the rest as background 

    Args:
        img (np.ndarray): Input image
        center_x (int): Center x coordinate of the ellipse 
        center_y (int): Center y coordinate of the ellipse 
        width (int): Width of the wanted ellipse
        height (int): Height of the wanted ellipse

    Returns:
        Tuple[np.ndarray, np.ndarray]: cropped output image
    """

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

    return image, np.array(cropped_image)



def get_image_paths(dir: str) -> List[str]:
    """
    Goes through a folder directory and lists all filepaths

    Args:
        dir (str): folder directory to extract from all the filepaths 

    Returns:
        List[str]: list of all filepaths
    """

    image_files = []
    for root, directories, files in os.walk(dir):
        for filename in files:
            if not filename.startswith("."):
                image_files.append(os.path.join(root, filename))  
    return sorted(image_files)


def contrast_img(img: np.ndarray) -> np.ndarray:
    """
    Applies the Adaptive Equalization or histogram equalization contrast method. This method
    enhances the contrast of an image by adjusting the intensity values of pixels based on the 
    distribution of pixel intensities in the image's histogram. 

    Args:
        img (np.ndarray): input image

    Returns:
        np.ndarray: contrasted image
    """
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


def createBinaryAnnotation(img: Union[np.ndarray, torch.Tensor],
                           frg_class: int) -> Union[np.ndarray, torch.Tensor]:
    """
    Creates a binary mask out of the prediction result with root as foreground and the rest as background

    Args:
        img (Union[np.ndarray, torch.Tensor]): Input image
        frg_class (int): value of the foreground class when creating binary segmentation masks

    Raises:
        TypeError: if the input is neither a numpy array or a torch tensor or if the foreground value is wrong. 

    Returns:
        Union[np.ndarray, torch.Tensor]: binary mask 
    """

    if isinstance(img, torch.Tensor):
        u = torch.unique(img)
        bkg = torch.zeros(img.shape)  # background
        if len(u) == 1:
            print("This prediction only contains background")
            return img
        else:
            try: 
                frg = (img == frg_class).int() * 255
            except: 
                print("Error in the foreground value")
    elif isinstance(img, np.ndarray):
        u = np.unique(img)
        bkg = np.zeros(img.shape)  # background
        if len(u) == 1:
            print("This prediction only contains background")
            return img
        else:
            try: 
                frg = (img == frg_class).astype(int) * 255
            except: 
                print("Error in the foreground value")
    else:
        raise TypeError("Input should be a PyTorch tensor or a NumPy array.")
    return bkg + frg

def common_files(folder1, folder2, prefix="Annotation"):
    """
    Compare two folders and return a list of base filenames (ignoring extensions)
    that exist in both folders.
    Files in folder2 may have a prefix (default: 'Annotation') that is ignored.
    """
    def get_basenames(folder, remove_prefix=False):
        basenames = set()
        for f in os.listdir(folder):
            if not os.path.isfile(os.path.join(folder, f)):
                continue
            name, _ = os.path.splitext(f)
            if remove_prefix and name.startswith(prefix):
                name = name[len(prefix):]
            basenames.add(name)
        return basenames

    base1 = get_basenames(folder1)
    base2 = get_basenames(folder2, remove_prefix=True)
    unique = base1.intersection(base2) #intersection()
    return sorted(unique)

    
def get_biomass(binary_img: np.ndarray) -> int:
    """
    Calculate the biomass by counting the number of pixels equal to 1

    Args:
        binary_img (np.ndarray): input image as binary mask

    Returns:
        int: integer value corresponding to the pixel count or root biomass. 
    """
    roi = binary_img > 0
    nerror = 0
    binary_img = binary_img * roi
    biomass = np.unique(binary_img.flatten(), return_counts=True)
    try:
        nbiomass = biomass[1][1]
    except:
        nbiomass = 0
        nerror += 1
        print("Seg error in ")
    return nbiomass

