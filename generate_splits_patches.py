import os 
import json 
import glob 

from collections import defaultdict


def create_split_data(image_to_patches, image_to_labels, image_list):
    return {
        "images": [image_to_patches[img] for img in image_list],
        "labels": [image_to_labels[img] for img in image_list]
    }


def main(data_dir, save_path_patches, json_filename):

    images_dir, label_dir = data_dir + "/images", data_dir + "/labels"
    image_filenames, label_filenames = [], []

    image_filenames += sorted(os.listdir(images_dir))
    label_filenames += sorted(os.listdir(label_dir))
    
    # Group images and labels by f_img
    image_to_patches = defaultdict(list)
    image_to_labels = defaultdict(list)

    for img_file, lbl_file in zip(image_filenames, label_filenames):
        f_img = img_file.split('_img_')[0]+'.tif'
        f_label_file = lbl_file.split('_img_')[0] +'.png'
        # Add image and label to the respective groups
        image_to_patches[f_img].append(img_file)
        image_to_labels[f_label_file].append(lbl_file)

   
    '''If we want to save the crop patches instead of the corresponding full size images, apply the following'''
    
    split = {}
    # If we already have a split json file 
    with open(json_filename, 'r') as split_file:
        data = json.load(split_file)

    train_images = data['train']['images']
    train_labels = data['train']['labels']
    
    val_images = data['val']['images']
    val_labels = data['val']['labels']

    test_images = data['test']['images']
    test_labels = data['test']['labels']


    split['train'] = {
        "images": [img for img in train_images for img in image_to_patches[img]],
        "labels": [lbl for lbl in train_labels for lbl in image_to_labels[lbl]]
    }

    split['val'] = {
        "images": [img for img in val_images for img in image_to_patches[img]],
        "labels": [lbl for lbl in val_labels for lbl in image_to_labels[lbl]]
    }

    split['test'] = {
        "images": [img for img in test_images for img in image_to_patches[img]],
        "labels": [lbl for lbl in test_labels for lbl in image_to_labels[lbl]]
    }
 
    # Step 5: Save the splits as JSON files
    with open(json_filename_patches, 'w') as split_file:
        json.dump(split, split_file, indent=4)

        
if __name__ == '__main__':
    data_path = "/home/zsordo/rhizonet-fovea/data_v3/patches64"
    json_filename_patches = "/home/zsordo/rhizonet-fovea/data_v3/split_patches64_w3ex10ex11_wotest.json"
    json_filename_full ="/home/zsordo/rhizonet-fovea/data_v3/split_wlab_w3ex10ex11.json"
    os.makedirs(json_filename_patches, exist_ok=True)
    main(data_path, json_filename_patches, json_filename=json_filename_full)
