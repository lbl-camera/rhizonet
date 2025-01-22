# RhizoNET



[![PyPI](https://badgen.net/pypi/v/rhizonet)](https://pypi.org/project/rhizonet/)
[![License](https://badgen.net/pypi/license/rhizonet)](https://github.com/lbl-camera/rhizonet)
[![Build Status](https://github.com/lbl-camera/rhizonet/actions/workflows/rhizonet-CI.yml)](https://github.com/lbl-camera/rhizonet/actions/workflows/rhizonet-CI.yml)
[![Documentation Status](https://rhizonet.readthedocs.io/en/latest/)](https://rhizonet.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/github/lbl-camera/rhizonet/graph/badge.svg?token=CuKaQXQLkt)](https://codecov.io/github/lbl-camera/rhizonet)

Segmentation pipeline for EcoFAB images

* License: MIT license
* Documentation: https://rhizonet.readthedocs.io

## Installation
```commandline
pip install rhizonet
```


## Features

* Create patches
* Train
* Inference
* Post-processing
* Evaluate metrics


## Copyright Notice 

RhizoNet Copyright (c) 2023, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of
any required approvals from the U.S. Dept. of Energy) and University
of California, Berkeley. All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.


## License Agreement 

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



