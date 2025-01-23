# RhizoNET



[![PyPI](https://badgen.net/pypi/v/rhizonet?cache-bust=%3Ctimestamp%3E)](https://pypi.org/project/rhizonet/)
[![License](https://badgen.net/pypi/license/rhizonet)](https://github.com/lbl-camera/rhizonet)
<!-- [![Build Status](https://github.com/lbl-camera/rhizonet/actions/workflows/rhizonet-CI.yml)](https://github.com/lbl-camera/rhizonet/actions/workflows/rhizonet-CI.yml) -->
[![Documentation Status](https://readthedocs.org/projects/rhizonet/badge/?version=latest)](https://rhizonet.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/github/lbl-camera/rhizonet/graph/badge.svg?token=CuKaQXQLkt)](https://codecov.io/github/lbl-camera/rhizonet)

Pipeline for deep-learning based 2D image segmentation of plant root grown in EcoFABs using a Residual U-net.

* License: MIT license
* Documentation: https://rhizonet.readthedocs.io
* Paper: https://www.nature.com/articles/s41598-024-63497-8


## Installation
```commandline
pip install rhizonet
```


## Features

* Create patches
```commandline
patchify_rhizonet --config_file ./setup_files/setup-pprepare.json 
```

* Train
```commandline
train_rhizonet --config_file ./setup_files/setup-unet2d.json --gpus 2 --strategy ddp --accelerator gpu
```

* Inference
```commandline
predict_rhizonet --config_file ./setup_files/setup-predict.json 
```

* Post-processing
```commandline
postprocess_rhizonet --config_file ./setup_files/setup-processing.json 
```

* Evaluate metrics
```commandline
evalmetrics_rhizonet ---pred_path "path" --label_path "path" --log_dir "path" --task "binary" --num_classes "2"
```
## Description

This code gives the tools to pre-process 2D RGB images and train a deep learning segmentation model using pytorch-lightning for code organization, logging and metrics for training and prediction. It uses as well the library monai for data augmentation and creating a Residual U-net model. 
The training patches can be created using the data preparation code for cropping and patching. 

The training was done on a dataset of multiple ecofabs (plants with different nutrition types) at the two last timestamps. The use of at least one gpu is necessary for training on small patch-size images.
The predictions can be done on any other timestamp by loading the appropriate model path. The Google Colab tutorial below details the steps to do so with a given subset of images and 3 possible model weights (varying with the size of the used patches).
It is also possible to apply the post-processing using the Google Colab tutorial on the predicted images which uses cropping and morphological operations, and plot the extracted biomass from the processed predictions. 


# Google Colab Tutorial for predicting and processing images
This [Google Colab Tutorial](https://colab.research.google.com/drive/1uJa1bHYfm076xCEhWcG20DVSdMIRh-lr?usp=drive_link) is a short notebook that can load 3 possible model weights depending the model type preferred (3 model weights for each patch size trained model), generate predictions and process these predictions given 2 random unseen EcoFAB images of the same experiment. It also generates plots of the extracted biomass for each nutrition type at each date and compares it to the groundtruth (which is the manually scaled biomass by biologists). 


## License Agreement and Copyright

MIT License

Copyright (c) 2025, Zineb Sordo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Credits

Please reference this work:
 <div class="row">
      <pre class="col-md-offset-2 col-md-8">
      @article{Sordo2024-ul,
                title    = "{RhizoNet} segments plant roots to assess biomass and growth for
                            enabling self-driving labs",
                author   = "Sordo, Zineb and Andeer, Peter and Sethian, James and Northen,
                            Trent and Ushizima, Daniela",
                abstract = "Flatbed scanners are commonly used for root analysis, but typical
                            manual segmentation methods are time-consuming and prone to
                            errors, especially in large-scale, multi-plant studies.
                            Furthermore, the complex nature of root structures combined with
                            noisy backgrounds in images complicates automated analysis.
                            Addressing these challenges, this article introduces RhizoNet, a
                            deep learning-based workflow to semantically segment plant root
                            scans. Utilizing a sophisticated Residual U-Net architecture,
                            RhizoNet enhances prediction accuracy and employs a convex hull
                            operation for delineation of the primary root component. Its main
                            objective is to accurately segment root biomass and monitor its
                            growth over time. RhizoNet processes color scans of plants grown
                            in a hydroponic system known as EcoFAB, subjected to specific
                            nutritional treatments. The root detection model using RhizoNet
                            demonstrates strong generalization in the validation tests of all
                            experiments despite variable treatments. The main contributions
                            are the standardization of root segmentation and phenotyping,
                            systematic and accelerated analysis of thousands of images,
                            significantly aiding in the precise assessment of root growth
                            dynamics under varying plant conditions, and offering a path
                            toward self-driving labs.",
                journal  = "Scientific Reports",
                volume   =  14,
                number   =  1,
                pages    = "12907",
                month    =  jun,
                year     =  2024
                }
      </pre>
    </div>

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter)
and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.



