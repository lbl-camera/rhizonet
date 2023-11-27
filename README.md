# Rhizonet
Pipeline for deep-learning based 2D image segmentation of plant root grown in EcoFABs using a Residual U-net.

## Description

This code gives the tools to pre-process 2D RGB images and train a deep learning segmentation model using pytorch-lightning for code organization, logging and metrics for training and prediction. It uses as well the library monai for data augmentation and creating a Residual U-net model. 
The training patches can be created using the data preparation code for cropping and patching. 

The training was done on a dataset of multiple ecofabs (plants with different nutrition types) at the two last timestamps. The use of at least one gpu is necessary for training on small patch-size images.
The predictions can be done on any other timestamp by loading the appropriate model path. The Google Colab tutorial below details the steps to do so with a given subset of images and 3 possible model weights (varying with the size of the used patches).
It is also possible to apply the post-processing using the Google Colab tutorial on the predicted images which uses cropping and morphological operations, and plot the extracted biomass from the processed predictions. 
# Getting started
# Steps to train the model

## Dependencies installation

Run the following to install libraries after creating your environment: 

- Download repo
    ```commandline
    git clone https://github.com/lbl-camera/rhizonet.git
    ```
- Create environment 

    ```commandline
    conda env create -f environment.yml 
    ```

- The file setup-unet2d.json in the folder setup-files is the file to modify that contains the directories with the data.

## Executing program

- On NERSC Perlmutter interactive node
  - create the unet model using settings specified in setup_files/setup-unet2d.py and train: 

    ```commandline
    module load python
    conda activate rootNET
    python train.py setup_files/setup-unet2d.json
    ```

  - prediction: 
    ```commandline
    python predict2d.py setup_files/setup-unet2d.json
    ```
The predictions are saved in the directory "pred_path" and the data is in "pred_data_dir"
  
  - post-processing: 
    ```commandline
    python postprocessing.py setup_files/setup-unet2d.json
    ```
The processed images are saved in the directory "output_path" and the data is in "data_path"

- Submitting NERSC batch jobs:
  - prepare patches:
    ```commandline
    sbatch batch_scripts/preapre_patches.sh
    ```
  
  - training: 
    ```commandline
    sbatch batch_scripts/train_unet2d.sh
    ```
  
  - prediction: 
    ```commandline
    sbatch batch_scripts/predict_unet2d.sh
    ```
  
  - post-processing: 
    ```commandline
    sbatch batch_scripts/processing_unet2d.sh
    ```
    
## Authors

Zineb Sordo (zsordo@lbl.gov), Dani Ushizima (dushizima@lbl.gov) 
