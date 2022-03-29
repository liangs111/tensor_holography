# Towards Real-time Photorealistic 3D Holography with Deep Neural Networks

### [Project Page](http://cgh.csail.mit.edu)  | [Paper](https://dx.doi.org/10.1038/s41586-020-03152-0) | [Data](https://drive.google.com/drive/folders/1TYDNfrfkehAJiUpDLadjxJzjDdgvC-GT?usp=sharing)

[Liang Shi](https://people.csail.mit.edu/liangs), [Beichen Li](https://www.linkedin.com/in/beichen-li-ba9b34106/), [Changil Kim](https://changilkim.com), [Petr Kellnhofer](https://kellnhofer.xyz), [Wojciech Matusik](https://cdfg.mit.edu/wojciech)

This repository contains the code to reproduce the results presented in "Towards Real-time Photorealistic 3D Holography with Deep Neural Networks" Nature 2021. **Please read the License before using the software.**

## Getting Started
**3/29/2022 Update: the phase 2 release is out. It comes with the training code and the TensorRT inference code. Phase 1 code users should recreate the conda environment. First, remove the existing conda enviroment with**
```bash
conda env remove --name tensorholo
```
**then follow the instructions below to recreate. New users should directly proceed with following instructions.**

This code runs with Python 3.8 and Tensorflow 1.15 (NVIDIA-maintained version to support training on the latest NVIDIA GPUS). You can set up a conda environment with the required dependencies using:

```bash
conda env create -f environment.yml
conda activate tensorholo
pip install nvidia-pyindex
pip install nvidia-tensorflow[horovod]
```

After downloading the hologram dataset, place all subfolders (`/*_384`, `/*_192`) into `/data` directory. The dataset contains raw images and a tfrecord generated for each subfolder. The code by default loads the tfrecord for training, testing, and validation.

To ease experimental validation of the predicted hologram, the provided dataset is computed for a collimated frustum with a 3D volume with a 6 mm optical path length. We recommend using a setup similar to [Maimone et al. 2017] Figure 10 (Right) to display the hologram. Users should feel free to choose the appropriate focal length of the collimating lens and imaging lens based on their lasers and applications. The dataset is computed for wavelengths at 450nm, 520nm, and 638nm. Mismatch of wavelengths may result in degraded experimental results.

The codebase provides a pretrained CNN for 8um pitch SLMs, code to evaluate/train the CNN, and code to perform accelerated inference with TensorRT.

## High-level structure

The code is organized as follows:

* ```main.py``` defines/trains/validates/evaluates/exports the CNN.
* ```optics.py``` contains optics-related helper functions and various implementations of double phase encoding.
* ```util.py``` contains several utility functions for network training.
* ```tfrecord.py``` contains code to generate and parse tfrecord.
* ```tensorrt_onnx.py``` contains code to generate a TensorRT engine for accelerated inference.

## Reproducing the experiments

#### Validate the pretrained model on the validation set

``` python
python main.py --validate-mode
```

#### Evaluate the pretrained model on arbitrary RGB-D inputs

``` python
python main.py --eval-mode
```

with following options

``` python
parser.add_argument('--eval-res-h', default=1080, type=int, help='Input image height in evaluation mode')
parser.add_argument('--eval-res-w', default=1920, type=int, help='Input image width in evaluation mode')
parser.add_argument('--eval-rgb-path', default=os.path.join(cur_dir, "data", "example_input", "couch_rgb.png"), help='Input rgb image path in evaluation mode')
parser.add_argument('--eval-depth-path', default=os.path.join(cur_dir, "data", "example_input", "couch_depth.png"), help='Input depth image path in evaluation mode')
parser.add_argument('--eval-output-path', default=os.path.join(cur_dir, "data", "example_input"), help='Output directory for results')
parser.add_argument('--eval-depth-shift', default=0, type=float, help='Depth shift (in mm) from the predicted midpoint hologram to the target hologram plane')
parser.add_argument('--gaussian-sigma', default=0.7, type=float, help='Sigma of Gaussian kernel used by AA-DPM')
parser.add_argument('--gaussian-width', default=3, type=int, help='Width of Gaussian kernel used by AA-DPM')
parser.add_argument('--phs-max', default=3.0, type=float, help='Maximum phase modulation of SLM in unit of pi')
parser.add_argument('--use-maimone-dpm', action='store_true', help='Use DPM of Maimone et al. 2017')
```

#### Train models
To retrain from scratch, use

``` python
python main.py --train-mode --model-name full_loss_retrain
```
with following options (users should feel free to add more by replacing variables in the dicts with an input argument)
``` python
parser.add_argument('--train-iters', default=1000, type=int, help='Training iterations')
```
To continue training the provided model, use
``` python
python main.py --train-mode --model-name full_loss --train-iters REPLACE_WITH_DESIRED_TOTAL_ITERATIONS
```
#### Export models to ONNX for TensorRT Inference

``` python
python main.py --export-mode --model-name full_loss
```

with a predefined resolution that users can set by

``` python
parser.add_argument('--trt-res-h', default=1080, type=int, help='Input image height in export (tensorrt) mode')
parser.add_argument('--trt-res-w', default=1920, type=int, help='Input image width in export (tensorrt) mode')
```

#### Run TensorRT inference
Running TensorRT inference requires setting up a separate environment. The following setup instructions are tested on ```Ubuntu 20.04``` with ```Python=3.8``` ```CUDA=11.6``` and ```TensorRT=8.4```.

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
After setting up the environment, if you only want to benchmark the performance of the exported ONNX model with TensorRT. Run the following command for fp32 inference

```
trtexec --onnx=ONNX_MODEL_PATH --saveEngine=TRT_ENGINE_PATH  --explicitBatch
```

and following command for FP16 precision inference

```
trtexec --onnx=ONNX_MODEL_PATH --saveEngine=TRT_ENGINE_PATH  --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16
```
, where ```ONNX_MODEL_PATH``` and ```TRT_ENGINE_PATH``` should be replaced by the actual ONNX model path and the desired path to save the TensorRT engine. To benchmark the model in Python, run the following command.

```python
python tensorrt_onnx.py --benchmark-mode
```
To run at FP16 precision, use
```python
python tensorrt_onnx.py --benchmark-mode --fp-16
```


To evaluate an input with the TensorRT model
```python
python tensorrt_onnx.py --eval-mode
```
and at FP16 precision with
```python
python tensorrt_onnx.py --eval-mode --fp-16
```

with following options

``` python
parser.add_argument('--eval-rgb-path', default=os.path.join(cur_dir, "data", "example_input", "couch_rgb.png"), help='Input rgb image path in evaluation mode')
parser.add_argument('--eval-depth-path', default=os.path.join(cur_dir, "data", "example_input", "couch_depth.png"), help='Input depth image path in evaluation mode')
parser.add_argument('--eval-output-path', default=os.path.join(cur_dir, "data", "example_input"), help='Output directory for results')
```
The benchmarked full model (30 layers/24 filters) performances
| GPU               | Runtime (fp32) | Runtime (fp16) | 
| -----------       | -----------    | -----------    |
| NVIDIA A6000      | 15.8 FPS       | **50.3 FPS**   |
| NVIDIA 2080 Max-Q | 5.3 FPS        | 16.2 FPS       |  


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

Our dataset and code, with the exception of the files in "data/example_image", are licensed under a custom license provided by the MIT Technology Licensing Office. By downloading the software, you agree to the terms of this License.
