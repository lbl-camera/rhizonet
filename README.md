# Rhizonet
Pipeline for deep-learning based 2D image segmentation of plant root grown in EcoFABs using a Residual U-net.

## Description

This code gives the tools to pre-process 2D RGB images and train a deep learning segmentation model using pytorch-lightning for code organization, logging and metrics for training and prediction. It uses as well the library monai for data augmentation and creating a Residual U-net model. 
The training patches can be created using the data preparation code for cropping and patching. 

The training was done on a dataset of multiple ecofabs (plants with different nutrition types) at the two last timestamps. The use of at least one gpu is necessary for training on small patch-size images.
The predictions can be done on any other timestamp by loading the appropriate model path. The Google Colab tutorial below details the steps to do so with a given subset of images and 3 possible model weights (varying with the size of the used patches).
It is also possible to apply the post-processing using the Google Colab tutorial on the predicted images which uses cropping and morphological operations, and plot the extracted biomass from the processed predictions. 

# Google Colab Tutorial for predicting and processing images
This [Google Colab Tutorial](https://colab.research.google.com/drive/1uJa1bHYfm076xCEhWcG20DVSdMIRh-lr?usp=drive_link) is a short notebook that can load 3 possible model weights depending the model type preferred (3 model weights for each patch size trained model), generate predictions and process these predictions given 2 random unseen EcoFAB images of the same experiment. It also generates plots of the extracted biomass for each nutrition type at each date and compares it to the groundtruth (which is the manually scaled biomass by biologists). 


# Getting started for running the code from scratch 

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
## License Agreement

RhizoNet Copyright (c) 2023, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of any
required approvals from the U.S. Dept. of Energy). All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

(1) Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

(2) Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

(3) Neither the name of the University of California, Lawrence Berkeley
National Laboratory, U.S. Dept. of Energy nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes, patches,
or upgrades to the features, functionality or performance of the source
code ("Enhancements") to anyone; however, if you choose to make your
Enhancements available either publicly, or directly to Lawrence Berkeley
National Laboratory, without imposing a separate written license agreement
for such Enhancements, then you hereby grant the following license: a
non-exclusive, royalty-free perpetual license to install, use, modify,
prepare derivative works, incorporate into other computer software,
distribute, and sublicense such enhancements or derivative works thereof,
in binary and source code form.


## References
* Zordo, Andeer, Sethian, Northen, [Ushizima. RhizoNet segments plant roots to assess biomass and growth for enabling self-driving labs, Nature Scientific Reports 2024](https://www.nature.com/articles/s41598-024-63497-8.pdf)
* [Ushizima, Zordo, Andeer, Sethian, Northen. RhizoNet: Image Segmentation for Plant Root in Hydroponic Ecosystem, bioRXiv 2023](https://www.biorxiv.org/content/10.1101/2023.11.20.565580v1)
* Huang, Perlmutter, Su, Quenum, Shevchenko, Zenyuk, [Ushizima. Detecting lithium plating dynamics in a solid-state battery with operando X-ray computed tomography using machine learning, Nature Computational Materials 2023](https://www.nature.com/articles/s41524-023-01039-y)
    
    
## Authors

Zineb Sordo (zsordo@lbl.gov), Dani Ushizima (dushizima@lbl.gov) 
