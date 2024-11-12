import os 
import json 
import glob 

from collections import defaultdict
from sklearn.model_selection import train_test_split


def create_split_data(image_to_patches, image_to_labels, image_list):
    return {
        "images": [image_to_patches[img] for img in image_list],
        "labels": [image_to_labels[img] for img in image_list]
    }


def main(data_dir, save_path):

    images_dir, label_dir = data_dir + "/images", data_dir + "/labels"
    image_filenames, label_filenames = [], []

    image_filenames += sorted(os.listdir(images_dir))
    label_filenames += sorted(os.listdir(label_dir))
    
    
#     full_image_filenames += sorted([os.path.join(images_dir, e) for e in os.listdir(images_dir) if not e.startswith(".")])
#     full_label_filenames += sorted([os.path.join(label_dir, e) for e in os.listdir(label_dir) if not e.startswith(".")])

    # Split images into train, validation, and test sets
    train_images, test_images = train_test_split(image_filenames, train_size=0.85, random_state=42)
    val_images, test_images = train_test_split(test_images, test_size=0.25, random_state=42)  
    
    train_labels, test_labels = train_test_split(label_filenames, train_size=0.85, random_state=42)
    val_labels, test_labels = train_test_split(test_labels, test_size=0.25, random_state=42)  

    print("Train set {}".format(len(train_images)/len(image_filenames)*100))
    print("Val set {}".format(len(val_images)/len(image_filenames)*100))
    print("Test set {}".format(len(test_images)/len(image_filenames)*100))

    # split = {}
    # split['train'] = train_images
    # split['val'] = val_images
    # split['test'] = test_images 
    # with open(os.path.join(save_path, 'split_allexp.json'), 'w') as split_file:
    #     json.dump(split, split_file, indent=4)    

    train_labels = ["Annotation" + e.replace("tif", "png") for e in train_images]
    val_labels = ["Annotation" + e.replace("tif", "png") for e in val_images]
    test_labels = ["Annotation" + e.replace("tif", "png") for e in test_images]

    split_wlab = {}
    split_wlab['train'] = {
        "images": train_images,
        "labels": train_labels
    }

    split_wlab['val'] = {
        "images": val_images,
        "labels": val_labels
    }

    split_wlab['test'] = {
        "images": test_images,
        "labels": test_labels
    }
 
    with open(os.path.join(save_path, 'split_wlab_w3ex10ex11.json'), 'w') as split_file:
        json.dump(split_wlab, split_file, indent=4)
        
        
if __name__ == '__main__':
    data_path = "/home/zsordo/rhizonet-fovea/data_v3"
    save_path = "/home/zsordo/rhizonet-fovea/data_v3/"
    os.makedirs(save_path, exist_ok=True)
    main(data_path, save_path)
