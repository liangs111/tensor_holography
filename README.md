# Towards Real-time Photorealistic 3D Holography with Deep Neural Networks
### [Project Page](http://cgh.csail.mit.edu)  | [Paper](https://dx.doi.org/10.1038/s41586-020-03152-0) | [Data](https://drive.google.com/drive/folders/1TYDNfrfkehAJiUpDLadjxJzjDdgvC-GT?usp=sharing)

[Liang Shi](https://people.csail.mit.edu/liangs), [Beichen Li](https://www.linkedin.com/in/beichen-li-ba9b34106/), [Changil Kim](https://changilkim.com), [Petr Kellnhofer](https://kellnhofer.xyz), [Wojciech Matusik](https://cdfg.mit.edu/wojciech)

This repository contains the code to reproduce the results presented in "Towards Real-time Photorealistic 3D Holography with Deep Neural Networks" Nature 2021. **Please read the license before using the software.**

## Getting Started

This code was developed in python 3.7 and Tensorflow 1.15. You can set up a conda environment with the required dependencies using:

```
conda env create -f environment.yml
conda activate tensorholo
```

After downloading the hologram dataset, place all subfolders (`/*_384`, `/*_192`) into `/data` directory. The dataset contains raw images and a tfrecord generated for each subfolder. The code by default loads the tfrecord for training, testing and validation.

**The current codebase doesn't contain the training code. We will soon make it available in the second phase of release.** The current codebase does contain a pretrained CNN for 8um pitch SLMs and code snippet to evaluate the CNN performance on the validation set.


## High-level structure

The code is organized as follows:

* ```main.py``` defines/trains/validates/evaluates the CNN.
* ```optics.py``` contains optics-related helper functions and various implementations of double phase encoding. 
* ```util.py``` contains several utility functions for network training.
* ```tfrecord.py``` contains code to generate and parse tfrecord.

## Reproducing the experiments

#### Validate the pretrained model on the validation set
```
python main.py --validate-mode
```

#### Evaluate the pretrained model on arbitrary RGB-D inputs
```
python main.py --eval-mode
```
with following options
```
parser.add_argument('--eval-res-h', default=1080, type=int, help='Input image height in evaluation mode')
parser.add_argument('--eval-res-w', default=1920, type=int, help='Input image width in evaluation mode')
parser.add_argument('--eval-rgb-path', default=os.path.join(cur_dir, "data", "example_input", "couch_rgb.png"), help='Input rgb image path in evaluation mode')
parser.add_argument('--eval-depth-path', default=os.path.join(cur_dir, "data", "example_input", "couch_depth.png"), help='Input depth image path in evaluation mode')
parser.add_argument('--eval-output-path', default=os.path.join(cur_dir, "data", "example_input"), help='Output directory for results')
parser.add_argument('--eval-depth-shift', default=0, help='Depth shift (in mm) from the predicted midpoint hologram to the target hologram plane')
parser.add_argument('--gaussian-sigma', default=0.7, help='Sigma of Gaussian kernel used by AA-DPM')
parser.add_argument('--gaussian-width', default=3, type=int, help='Width of Gaussian kernel used by AA-DPM')
parser.add_argument('--phs-max', default=3.0, help='Maximum phase modulation of SLM in unit of pi')
parser.add_argument('--use-maimone-dpm', action='store_true', help='Use DPM of Maimone et al. 2017')
```


## Citation
If you find our work useful in your research, please cite:

```
@article{Shi2021:TensorHolography,
    title   = "Towards real-time photorealistic {3D} holography with deep neural
                networks",
    author  = "Shi, Liang and Li, Beichen and Kim, Changil and Kellnhofer, Petr
                and Matusik, Wojciech",
    journal = "Nature",
    volume  =  591,
    number  =  7849,
    pages   = "234--239",
    year    =  2021
}
```

## License
Our dataset and code, with exception of the files in "data/example_image", are licensed under a custom license provided by the MIT Technology Licensing Office. By downloading the software, you agree to the terms of this License.
