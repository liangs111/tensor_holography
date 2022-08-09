# End-to-end Learning of 3D Phase-only Holograms for Holographic Display

### [Project Page](http://cgh-v2.csail.mit.edu)  | [Paper](https://www.nature.com/articles/s41377-022-00894-6) | [Data](https://drive.google.com/drive/folders/1hlnk_yMjm2aebFillJFxFM1UaiI-INPg?usp=sharing)

[Liang Shi](https://people.csail.mit.edu/liangs), [Beichen Li](https://www.linkedin.com/in/beichen-li-ba9b34106/), [Wojciech Matusik](https://cdfg.mit.edu/wojciech)

## Data
Download the hologram dataset from the above link, place all subfolders (`/*_384_v2`) into `/data` directory. The dataset contains raw LDIs and two tfrecords, one for the first layer of the LDIs and the target holograms, the other for the full LDIs and the target holograms. The code by default loads the tfrecords for training, testing, and validation.

The specs of the holograms are identical to MIT-CGH-4K-V1. Please refer to [v1-reproduction-instruction](TensorHolo_v1.md) for more discussion. The codebase provides two pretrained CNNs after stage 1 training, one takes only the first layer of the LDI as input, the other takes the full LDI (available soon in the next commit). Both models are trained to reproduce the target hologram generated from the full LDI. Three pretrained CNNs after stage 2 training are also provided for a display offset of 0mm/6mm/12mm, all takes the first layer of LDI (an RGB-D image) as input.

## Reproducing the experiments

#### Validate the pretrained model after stage 1
``` python
python main_v2.py --validate-mode-s1 --active-max-ldi-layer LDI_LAYER
```
Replace LDI_LAYER with 0 or 4 to use the first layer of the LDI or the full LDI.

#### Validate the pretrain model after stage 2 (pretrained models available for RGB-D input)
``` python
python main_v2.py --activate-ddpm --validate-mode-s2 --active-max-ldi-layer 0 --train-depth-shift DEPTH --eval-depth-shift DEPTH
```
Replace DEPTH with the desired depth shift. 

#### Evaluate the pretrained model after stage 2 (pretrained models available for RGB-D input)

``` python
python main_v2.py --eval-mode --activate-ddpm --active-max-ldi-layer 0 --train-depth-shift DEPTH --eval-depth-shift DEPTH
```

with following options
``` python
parser.add_argument('--eval-mode', action='store_true', help='Run in evaluation mode')
parser.add_argument('--eval-res-h', default=1080, type=int, help='Input image height in evaluation mode')
parser.add_argument('--eval-res-w', default=1920, type=int, help='Input image width in evaluation mode')
parser.add_argument('--eval-rgb-path', default=os.path.join(cur_dir, "data", "example_input", "couch_rgb.png"), help='Input rgb image path in evaluation mode')
parser.add_argument('--eval-depth-path', default=os.path.join(cur_dir, "data", "example_input", "couch_depth.png"), help='Input depth image path in evaluation mode')
parser.add_argument('--eval-output-path', default=os.path.join(cur_dir, "data", "example_input"), help='Output directory for results')
parser.add_argument('--eval-depth-shift', default=0, type=float, help='Depth shift (in mm) from the predicted midpoint hologram to the target hologram plane')
parser.add_argument('--gaussian-sigma', default=0.0, type=float, help='Sigma of Gaussian kernel used by AA-DPM')
parser.add_argument('--gaussian-width', default=3, type=int, help='Width of Gaussian kernel used by AA-DPM')
parser.add_argument('--phs-max', default=2.0, type=float, help='Maximum phase modulation of SLM in unit of pi')
parser.add_argument('--use-maimone-dpm', action='store_true', help='Use DPM of Maimone et al. 2017')
parser.add_argument('--k', default=1.0, type=float, help='k for generating Fourier-space mask used by BL-DPM')
parser.add_argument('--use-bldpm', action='store_true', help='Use BL-DPM of Sui et al. 2021')
```
When users trained their own full LDI input models, namely when --active-max-ldi-layer will be set above 0, a suffix of the LDI layer index will be appended to the given --eval-rgb-path and --eval-depth-path. For example, when --active-max-ldi-layer is 2, the default --eval-rgb-path is will be interpreted as couch_rgb_0.png, couch_rgb_1.png, couch_rgb_2.png, similarly for --eval-depth-path. Please name your LDI layers using this convention to faciliate proper data loading.

Two other filtering options are provided, the anti-aliasing double phase method (AA-DPM) [Shi et al. 2021], and the band-limited double phase method (BL-DPM) [Sui et al. 2021]. To compare with DDPM, evaluate AA-DPM with command
``` python
python main_v2.py --eval-mode --active-max-ldi-layer 0 --eval-depth-shift DEPTH --gaussian-sigma DESIRED_SIGMA --gaussian-width DESIRED_WIDTH
```
replace DEPTH, DESIRED_SIGMA, DESIRED_WIDTH with the desired values.

Evaluate BL-DPM with command
``` python
python main_v2.py --eval-mode --active-max-ldi-layer 0 --eval-depth-shift DEPTH --use-bldpm --k DESIRED_K
```
replace DESIRED_K with the desired values.

#### Train models
To retrain the model from scratch with only the first stage training

``` python
python main_v2.py --train-mode --model-name full_loss
```
To retrain the model from scratch with both the first and second stage training
``` python
python main_v2.py --train-mode --model-name full_loss --activate-ddpm --train-depth-shift DEPTH
```
Replace DEPTH with the desired depth shift

To retrain the model from scratch with both the first and second stage loss, but without the filtering network
``` python
python main_v2.py --train-mode --model-name full_loss --activate-ddpm --train-depth-shift DEPTH --bypass-ddpm-network
```
Replace DEPTH with the desired depth shift. This mode is only recommended for 0 depth shift.

All related options
``` python
parser.add_argument('--activate-ddpm', action='store_true', help='Load ddpm network together with hologram rendering network; depth shift specified by --train-depth-shift')
parser.add_argument('--bypass-ddpm-network', action='store_true', help='Train/evaluate ddpm without using ddpm network (typical for 0 mm offset)')
parser.add_argument('--train-depth-shift', default=12.0, type=int, help='The epoch to start stage-2 training')
parser.add_argument('--num-epochs', default=4050, type=int, help='Number of training epochs')
parser.add_argument('--epoch_to_start_ddpm_training', default=3000, type=int, help='The epoch to start stage-2 training')
parser.add_argument('--padding', default=0, type=int, help='Padding to the hologram to accommodate out-of-frame diffraction')
```

#### Export models to ONNX for TensorRT Inference
Because onnx doesn't support complex tensor and FFT2/IFFT2, two-stage-trained CNN can only be exported with 0 depth shift

``` python
python main_v2.py --export-mode --model-name full_loss --activate-ddpm --train-depth-shift 0
```
To export the first-stage trained CNN, or ones that bypasse the filtering network,
``` python
python main_v2.py --export-mode --model-name full_loss
python main_v2.py --export-mode --model-name full_loss --activate-ddpm --bypass-ddpm-network --train-depth-shift DEPTH
```
Replace DEPTH with the desired depth shift\
All commands require a predefined resolution that users can set by

``` python
parser.add_argument('--trt-res-h', default=1080, type=int, help='Input image height in export (tensorrt) mode')
parser.add_argument('--trt-res-w', default=1920, type=int, help='Input image width in export (tensorrt) mode')
```

#### Run TensorRT inference (activate trt enviroment)
Benchmark the performance of the exported ONNX model with TensorRT. Run the following command for fp32 inference

```
trtexec --onnx=ONNX_MODEL_PATH --saveEngine=TRT_ENGINE_PATH  --explicitBatch
```

and following command for FP16 precision inference

```
trtexec --onnx=ONNX_MODEL_PATH --saveEngine=TRT_ENGINE_PATH  --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16
```
, where ```ONNX_MODEL_PATH``` and ```TRT_ENGINE_PATH``` should be replaced by the actual ONNX model path and the desired path to save the TensorRT engine. To benchmark the model in Python, run the following command.

```python
python tensorrt_onnx.py --benchmark-mode --v2 --activate-ddpm
```
To run at FP16 precision, use
```python
python tensorrt_onnx.py --benchmark-mode --v2 --activate-ddpm --fp-16
```


To evaluate an input with the TensorRT model
```python
python tensorrt_onnx.py --eval-mode --v2 --activate-ddpm
```
and at FP16 precision with
```python
python tensorrt_onnx.py --eval-mode --v2 --activate-ddpm --fp-16
```

with following options

``` python
parser.add_argument('--eval-rgb-path', default=os.path.join(cur_dir, "data", "example_input", "couch_rgb.png"), help='Input rgb image path in evaluation mode')
parser.add_argument('--eval-depth-path', default=os.path.join(cur_dir, "data", "example_input", "couch_depth.png"), help='Input depth image path in evaluation mode')
parser.add_argument('--eval-output-path', default=os.path.join(cur_dir, "data", "example_input"), help='Output directory for results')
```
The benchmarked full model (30 layers/24 filters + 8 layers/8 filters) performances
| GPU               | Runtime (fp32) | Runtime (fp16) | 
| -----------       | -----------    | -----------    |
| NVIDIA A6000      | 14.7 FPS       | **42.5 FPS**   |