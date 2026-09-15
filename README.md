# ASCPR-Net: Adaptive Structure–Content Prior Learning and Style-Consistent Reconstruction for Dunhuang Mural Inpainting

This repository contains the PyTorch implementation associated with **ASCPR-Net**, a Dunhuang mural inpainting framework for reconstruction under complex damage conditions.

ASCPR-Net is designed to coordinate **structural priors, contextual content information, and style guidance** during mural reconstruction. The method learns adaptive structural information from damaged murals, complements it with multiscale content information from intact regions, and introduces explicit style representations during decoding to improve structural continuity, content recovery, and stylistic consistency.

Repository: https://github.com/autumnst/ASCPR-Net1

> **Note**
>
> The current public repository is a research/development snapshot. Some default paths, iteration settings, and internal module names in the code retain settings from earlier experiments. The **paper configuration described below should be treated as the reference configuration for the results reported in the manuscript**. Before running the code on another machine, replace the local dataset/checkpoint paths in the option files with your own paths.

---

## Method Overview

The ASCPR-Net framework contains three main functional components.

### 1. Adaptive structure–content prior learning

The framework learns structural and content information collaboratively rather than relying only on fixed external structural priors.

- **Adaptive Structural Prior Learning (AdaSPL)** extracts structural cues related to boundaries, line orientation, and spatial organization from damaged mural images.
- **Multiscale content prior aggregation** collects semantic, textural, and contextual information from intact regions.
- **Content-Guided Prior Compensation (CGPCM)** coordinates structural constraints with valid contextual content during reconstruction.

### 2. Explicit style guidance

To improve stylistic consistency between reconstructed regions and their surrounding intact content, the method introduces explicit style information during decoding.

- Multiscale style features are extracted from intact mural regions.
- A learnable style representation is used to model recurring color, texture, and artistic characteristics.
- Style information is introduced into multiple decoding stages to guide color and texture consistency.

### 3. Collaborative reconstruction

Structural, content, and style information are progressively integrated into the main inpainting pathway to generate the final restored mural image.

---

## Repository Structure

The current repository contains the following main files and directories:

```text
ASCPR-Net1/
├── criteria/               # loss functions
├── datasets/               # dataset loading and image/mask transforms
├── models/                 # generator, discriminators, reconstructor and prior modules
├── options/                # training and testing options
├── utils/                  # image processing, distributed training and utility functions
├── train.py                # training entry
├── trainer.py              # training loop and staged optimization
├── test.py                 # testing/inference entry
├── eval_mix.py             # PSNR/SSIM/L2/LPIPS/FID evaluation
├── image_masks.py          # image/mask processing utility
├── run_orignal_test.sh     # example testing script
├── rename.py               # file-renaming utility
└── LICENSE                 # Apache License 2.0
```

The training code constructs the generator, image discriminator, structural reconstructor, and structural discriminator in `train.py`. The main training loop, including the structure-prior pretraining/freeze logic, is implemented in `trainer.py`.

---

## Prerequisites

The experimental environment reported in the paper is:

- Python 3.x
- PyTorch 2.1
- CUDA 11.8
- NVIDIA GPU
- Adam optimizer

The experiments reported in the manuscript were conducted on a **single NVIDIA A800 GPU**.

The current repository does not contain a finalized `requirements.txt`. Based on the released code, the main Python dependencies include:

```bash
pip install torch torchvision numpy pillow tqdm tensorboardX lpips \
            opencv-python matplotlib pandas scipy scikit-image
```

Please install the PyTorch build that matches your CUDA environment.

---

## Datasets

### Dunhuang Mural Dataset

The main experiments use the **Dunhuang Mural Dataset**, which contains 18,370 mural images and artist-created reproductions spanning multiple categories, including religious scenes, historical narratives, figure images, Buddha images, and decorative patterns.

Following the experimental protocol described in the paper, cropping and random rotation are used to increase sample diversity, producing 72,399 images in total. Images are resized to **256 × 256** pixels.

### Random masks

Random irregular masks are obtained from the [NVIDIA Irregular Mask Dataset](https://nv-adlr.github.io/publication/partialconv-inpainting).

### Mold-stain masks

Mold-stain masks are used to simulate mural-specific degradation patterns, including surface contamination, erosion, and local flaking.

The synthetic damage ratio used in the paper ranges from **0% to 60%**.

### Real damaged murals

Real damaged mural samples from the **MuralDH** dataset are used to evaluate the applicability of ASCPR-Net to practical mural degradation scenarios.

### Cross-dataset evaluation

The trained model is further evaluated without additional fine-tuning on three external cultural heritage image datasets:

- Thangka images
- Yongle Palace murals
- Fahai Temple murals

These datasets are used to examine the behavior of the model on cultural heritage images with different color distributions, pattern organizations, material appearances, and stylistic characteristics.

---

## Dataset Organization

The released data loader expects separate image and mask directories. A recommended organization is:

```text
data/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Set the corresponding paths in:

```text
options/train_options.py
options/test_options.py
```

or pass the testing paths through command-line arguments.

### Mask convention

The current data loader:

1. loads masks as single-channel grayscale images;
2. resizes them with nearest-neighbor interpolation;
3. binarizes them using a threshold of `0.5`;
4. internally applies `mask = 1 - mask`.

Therefore, please verify that the mask convention of your own dataset is consistent with this implementation before training or testing.

During training, masks are randomly sampled from the mask directory. During testing, masks are loaded sequentially according to the dataset index.

---

## Image Preprocessing

The released dataset code applies the following preprocessing:

- RGB conversion;
- resize to the configured `load_size`;
- conversion to tensor;
- normalization using mean `(0.5, 0.5, 0.5)` and standard deviation `(0.5, 0.5, 0.5)`.

Thus, input RGB images are normalized approximately to the range `[-1, 1]`.

The default image size in both training and testing options is:

```text
256 × 256
```

---

## Paper Training Configuration

The configuration reported in the manuscript is:

```text
Framework:              PyTorch 2.1
CUDA:                   11.8
GPU:                    NVIDIA A800
Image resolution:       256 × 256
Batch size:             8
Training epochs:        30
Generator learning rate:      5e-5
Discriminator learning rate:  1e-4
Optimizer:              Adam
```

Loss weights reported in the paper are:

```text
lambda_sp    = 1
lambda_valid = 2
lambda_hole  = 60
lambda_perc  = 0.05
lambda_es    = 0.5
lambda_adv   = 0.1
```

The paper adopts a staged optimization strategy:

- **Epochs 1–10:** the structure–content prior network remains learnable and is jointly optimized with the main inpainting network.
- **Epochs 11–30:** the structure–content prior network is frozen, while the style-guidance and reconstruction-related modules continue to be optimized.

---

## Current Repository Defaults

The current `options/train_options.py` retains development defaults that differ from the final paper configuration. Important defaults include:

```text
batch_size        = 2
load_size         = (256, 256)
sigma             = 2.0
gen_lr            = 1e-4
D2G_lr            = 0.1
lr_finetune       = 5e-5
train_iter        = 700000
rec_pretrain_iter = 100000
save_interval     = 40000
```

The training code uses `rec_pretrain_iter` to separate two stages:

- before the threshold, the structural reconstructor and its discriminator are trainable;
- after the threshold, the structural branch is frozen and used to provide prior information.

For reproduction of the manuscript results, align these settings with the paper configuration before training.

---

## Training

### Important path configuration

The current repository contains machine-specific default paths in `options/train_options.py` and `options/test_options.py`. In addition, `train.py` contains a local checkpoint path used when loading the pretrained structural branch during a resumed second-stage run.

Before training, update these paths to match your environment.

The safest workflow for the current research snapshot is:

1. edit the dataset/checkpoint defaults in `options/train_options.py`;
2. edit the validation paths in `options/test_options.py`;
3. update the structural checkpoint path in `train.py` if a resumed second-stage run is required;
4. run:

```bash
python train.py
```

### Training from scratch

To start without a main pretrained checkpoint, set:

```text
pre_trained = ""
start_iter  = 1
```

in `options/train_options.py`.

The training script creates the training dataset through `datasets.dataset.create_image_dataset`, constructs the model components, and invokes the training loop in `trainer.py`.

### Distributed training

The code contains PyTorch distributed-training support using the `WORLD_SIZE` environment variable and the NCCL backend. Single-GPU training is used when `WORLD_SIZE=1`.

---

## Testing

The testing interface is implemented in `test.py`.

General usage:

```bash
python test.py \
  --pre_trained [path to checkpoint] \
  --image_root [path to test images] \
  --mask_root [path to test masks] \
  --result_root [path to output directory] \
  --number_eval [number of images to evaluate]
```

The repository also provides `run_orignal_test.sh` as an example. A portable version is:

```bash
python test.py \
  --pre_trained checkpoints/model.pt \
  --image_root data/test/images \
  --mask_root data/test/masks \
  --result_root results/ASCPR-Net \
  --number_eval 300
```

The default testing options are:

```text
batch_size  = 1
load_size   = (256, 256)
sigma       = 2.0
number_eval = 500
```

During testing, the script creates:

```text
result_root/
├── generate/
├── corrupt/
└── ground_truth/
```

for generated results, corrupted inputs, and ground-truth images, respectively.

---

## Evaluation

The paper evaluates ASCPR-Net using five metrics:

- **L2** ↓
- **PSNR** ↑
- **SSIM** ↑
- **LPIPS** ↓
- **FID** ↓

The repository includes `eval_mix.py` for quantitative evaluation. It computes image-level metrics and aggregates the results. The evaluation code uses:

- OpenCV for image loading;
- LPIPS for perceptual similarity;
- `skimage.metrics.structural_similarity` for SSIM;
- Inception features and SciPy matrix operations for FID;
- pandas for CSV output.

Before running `eval_mix.py`, update the paths near the configuration section of the script:

```text
test_root
referpath
maskpath
```

The script writes per-image and averaged metric results to CSV files.

---

## Experiments Reported in the Paper

The manuscript includes:

- quantitative and qualitative comparison under random masks;
- quantitative and qualitative comparison under mold-stain masks;
- evaluation across different damage ratios;
- component ablation studies;
- expert evaluation;
- cross-dataset evaluation on Thangka, Yongle Palace, and Fahai Temple images;
- evaluation on real damaged mural samples;
- computational complexity analysis.

---

## Data Availability

The datasets used in this study include the Dunhuang Mural Dataset, the NVIDIA Irregular Mask Dataset, the MuralDH dataset, and external cultural heritage image datasets containing Thangka, Yongle Palace mural, and Fahai Temple mural images.

Publicly available datasets should be obtained from their original sources. Due to copyright and data-use restrictions associated with cultural heritage images, processed mural images, derived training/test data, and mold-stain masks are not redistributed publicly in this repository. Research data may be made available from the corresponding author upon reasonable request for non-commercial academic research purposes.

---

## Code Availability

The source code associated with this work is available in this repository:

https://github.com/autumnst/ASCPR-Net1

The current repository is licensed under the **Apache License 2.0**.

---

## Citation

If this repository or the ASCPR-Net framework is useful for your research, please cite the corresponding paper.

The bibliographic information can be updated after formal publication.

```bibtex
@misc{Tan2026ASCPRNet,
  author = {Tan, Qiulin and Liu, Hongcai and Li, Qiaoqiao and Wang, Weilan and Zhao, Dongdong},
  title = {ASCPR-Net: Adaptive Structure Content Prior Learning and Style-Consistent Reconstruction for Dunhuang Mural Inpainting},
  year = {2026},
  note = {Manuscript}
}
```

---

## Acknowledgements

This repository builds on research in image inpainting and adaptive-prior learning. We thank the authors of the public datasets and open-source implementations used for comparison and evaluation.

The repository structure follows the implementation lineage of the adaptive-prior inpainting codebase used in this study.

---

## License

This repository is released under the **Apache License 2.0**. See `LICENSE` for details.

---

## Contact

For questions regarding the paper, implementation, or research data, please contact:

**Qiaoqiao Li**  
Corresponding author  
Northwest Minzu University  
Email: `liqq@xbmu.edu.cn`
