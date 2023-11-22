# Rhizonet
segmentation for plant roots grown in EcoFABs
Pipeline for deep-learning based image segmentation of plant root 2D slices at different timestamps using a U-net.

## Description

This code gives the tools to pre-process 2D RGB images and train a deep learning segmentation model using pytorch-lightning for code organization, logging and metrics, and multi-GPU parallelization for training and prediction. It uses as well the library monai for data augmentation and creating a U-net model. 
The training is done preprocessed 2D patches. The training patches can be created using the data preparation code for cropping and patching. 

The training was done on a dataset of multiple ecofabs (plants with different nutrition types) at the two last timestamps. The use of at least one gpu is necessary for training on small patch-size images.
The predictions can be done on any other timestamp by loading the appropriate model path. 
Finally,it is possible to apply the post-processing code on the predicted images which uses cropping and morphological operations.
# Getting started

## Dependencies installation

Run the following to install libraries after creating your environment: 

- Download repo
    ```commandline
    git clone https://github.com/dani-lbnl/zineb.git
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
  -

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
