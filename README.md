# Tensor Holography

**Towards Real-time Photorealistic 3D Holography with Deep Neural Networks (TensorHolo V1)**\
Nature 2021\
[Project Page](http://cgh.csail.mit.edu)  | [Paper](https://dx.doi.org/10.1038/s41586-020-03152-0) | [Data](https://drive.google.com/drive/folders/1TYDNfrfkehAJiUpDLadjxJzjDdgvC-GT?usp=sharing)\
[Liang Shi](https://people.csail.mit.edu/liangs), [Beichen Li](https://www.linkedin.com/in/beichen-li-ba9b34106/), [Changil Kim](https://changilkim.com), [Petr Kellnhofer](https://kellnhofer.xyz), [Wojciech Matusik](https://cdfg.mit.edu/wojciech)

**End-to-end Learning of 3D Phase-only Holograms for Holographic Display (TensorHolo V2)**\
Light: Science and Applications 2022 [Impact factor: 20.26 (in 2022)]\
[Project Page](http://cgh-v2.csail.mit.edu)  | [Paper](https://doi.org/10.1038/s41377-022-00894-6) | [Data](https://drive.google.com/drive/folders/1hlnk_yMjm2aebFillJFxFM1UaiI-INPg?usp=sharing)\
[Liang Shi](https://people.csail.mit.edu/liangs), [Beichen Li](https://www.linkedin.com/in/beichen-li-ba9b34106/), [Wojciech Matusik](https://cdfg.mit.edu/wojciech)

This repository contains the code to reproduce the results presented in the above papers. **Please read the License before using the software.**

## (New) Related Works 
**Ergonomic-Centric Holography: Optimizing Realism, Immersion, and Comfort for Holographic Display**\
Arxiv 2023\
Project Page (in preparation)  | [Paper](http://arxiv.org/abs/2306.08138) | [Supplement](http://people.csail.mit.edu/liangs/papers/EC-H23_S.pdf) | Code (in preparation)\
[Liang Shi*](https://people.csail.mit.edu/liangs), [DongHun Ryu*](https://sites.google.com/view/dhryu/), [Wojciech Matusik](https://cdfg.mit.edu/wojciech) (* indicates equal contribution)

## Getting Started
**8/9/2022 Update: TensorHolo V2 code/dataset released.**

This code runs with Python 3.8 and Tensorflow 1.15 (NVIDIA-maintained version to support training on the latest NVIDIA GPUS). You can set up a conda environment with the required dependencies using:

```bash
conda env create -f environment.yml
conda activate tensorholo
pip install nvidia-pyindex
pip install nvidia-tensorflow[horovod]
```
Alternatively, set up the following enviroment if you plan to export the model for TensorRT accelerated inference. The following instructions are tested on ```Ubuntu 20.04``` with ```Python=3.8``` ```CUDA=11.6``` and ```TensorRT=8.4```.

``` bash
# Install CUDA 11.6 (Change to the correct link based on your own system)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.6.1/local_installers/cuda-repo-ubuntu2004-11-6-local_11.6.1-510.47.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-6-local_11.6.1-510.47.03-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-6-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda

# Install TensorRT
# Download deb package from NVIDIA
# Replace xx in the os and tag with your package name
os="ubuntuxx04"
tag="cudax.x-trt8.x.x.x-yyyymmdd"
sudo dpkg -i nv-tensorrt-repo-${os}-${tag}_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-${os}-${tag}/7fa2af80.pub
sudo apt-get update
sudo apt-get install tensorrt

# Install other relevant packages
sudo apt-get install python3-libnvinfer-dev
sudo apt-get install onnx-graphsurgeon

# Add tensorrt bin path to use trtexec
export PATH=/usr/src/tensorrt/bin:$PATH

# Create a tensorrt environment
conda env create -f environment_trt.yml
conda activate trt
pip install nvidia-pyindex
pip install nvidia-tensorflow[horovod]
pip install nvidia-tensorrt
```

## High-level structure

The code is organized as follows:

* ```main.py``` defines/trains/validates/evaluates/exports the CNN for TensorHolo V1.
* ```main_v2.py``` defines/trains/validates/evaluates/exports the CNN for TensorHolo V2.
* ```optics.py``` contains optics-related helper functions and various implementations of double phase encoding.
* ```util.py``` contains several utility functions for network training.
* ```tfrecord.py``` contains code to generate and parse tfrecord.
* ```tensorrt_onnx.py``` contains code to generate a TensorRT engine for accelerated inference.

## [Reproducing the experiments of TensorHolo V1](TensorHolo_v1.md)
## [Reproducing the experiments of TensorHolo V2](TensorHolo_v2.md)


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
```
@article{Shi2022:TensorHolography-v2,
    title    = "End-to-end learning of {3D} phase-only holograms for holographic
                display",
    author   = "Shi, Liang and Li, Beichen and Matusik, Wojciech",
    journal  = "Light Sci Appl",
    volume   =  11,
    number   =  1,
    pages    = "247",
    month    =  aug,
    year     =  2022,
    language = "en"
} 
```
```
@misc{Shi2023:EC-H,
    title={Ergonomic-Centric Holography: Optimizing Realism,Immersion, and Comfort for Holographic Display}, 
    author={Liang Shi and Donghun Ryu and Wojciech Matusik},
    year={2023},
    eprint={2306.08138},
    archivePrefix={arXiv},
    primaryClass={cs.GR}
}
```

## License

Our dataset and code, with the exception of the files in "data/example_image", are licensed under a custom license provided by the MIT Technology Licensing Office. By downloading the software, you agree to the terms of this License.
