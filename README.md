# ASCPR-Net: Adaptive Structure–Content Prior Learning and Style-Consistent Reconstruction for Dunhuang Mural Inpainting

This repository provides the PyTorch implementation of **ASCPR-Net**, a framework for Dunhuang mural inpainting that integrates adaptive structural priors, contextual content compensation, and explicit style guidance to improve structural continuity, content recovery, and stylistic consistency.

**Repository:** https://github.com/autumnst/ASCPR-Net

## Method Overview

ASCPR-Net consists of three complementary components:

- **Adaptive Structural Prior Learning (AdaSPL):** learns structural cues, including boundaries, line orientations, and spatial relationships, from damaged murals.
- **Content-Guided Prior Compensation (CGPCM):** integrates multiscale contextual information from intact regions to support coherent reconstruction.
- **Prototype-based style guidance:** introduces learnable style representations during decoding to improve color and texture consistency.

## Repository Structure

```text
ASCPR-Net/
├── criteria/          # Loss functions
├── datasets/          # Dataset loading and transforms
├── models/            # Network architectures
├── options/           # Training and testing options
├── utils/             # Utilities
├── train.py           # Training
├── trainer.py         # Training loop
├── test.py            # Inference
├── eval_mix.py        # Quantitative evaluation
└── LICENSE            # Apache License 2.0
```

The mold-stain mask generation code is also provided in the repository.

## Requirements

- Python 3.x
- PyTorch 2.1
- CUDA 11.8
- NVIDIA GPU

Install the main dependencies:

```bash
pip install torch torchvision numpy pillow tqdm tensorboardX lpips opencv-python matplotlib pandas scipy scikit-image
```

## Datasets

### Dunhuang Mural Dataset

The Dunhuang Mural Dataset contains **18,370** original images, including mural images and reproductions. The original dataset is first split into training, validation, and test sets at a ratio of **8:1:1**. Data augmentation is then applied exclusively to the training set, expanding it to **72,399** images. The validation and test sets are not augmented. The input resolution is **256 × 256**.

### Random Masks

Random irregular masks are available from the [NVIDIA Irregular Mask Dataset](https://nv-adlr.github.io/publication/partialconv-inpainting).

### Mold-Stain Masks

We provide **8,920 synthetic mold-stain masks** at **256 × 256** resolution, covering damage ratios from **0% to 60%**. The masks were generated using a random-walk-based procedure; the corresponding generation code is available in this repository.

- **Download:** [Baidu Netdisk — provid_masks.zip](https://pan.baidu.com/s/1cJZ4TdkRYGnUZIXZdynOWw)
- **Extraction code:** `5u8t`

### Other Evaluation Datasets

Real damaged murals are evaluated using the [MuralDH](https://github.com/tearsheaven/MuralDH) dataset. Cross-dataset evaluations use Thangka, Yongle Palace mural, and Fahai Temple mural images.

## Dataset Organization

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

Set the corresponding dataset and checkpoint paths in `options/train_options.py` and `options/test_options.py` before running.

## Training

```bash
python train.py
```

## Testing

```bash
python test.py \
  --pre_trained checkpoints/model.pt \
  --image_root data/test/images \
  --mask_root data/test/masks \
  --result_root results/ASCPR-Net \
  --number_eval 300
```

## Evaluation

The evaluation script `eval_mix.py` computes **L2**, **PSNR**, **SSIM**, **LPIPS**, and **FID**. Update `test_root`, `referpath`, and `maskpath` in the script to match your local files.

## Data Availability

The synthetic mold-stain masks are available through the Baidu Netdisk link above, and the mask generation code is included in this repository. Third-party cultural heritage images and datasets should be obtained from their respective sources; the processed mural image datasets are not redistributed here.

## Citation

If you find this repository useful, please cite our work:

```bibtex
@misc{Tan2026ASCPRNet,
  author = {Tan, Qiulin and Liu, Hongcai and Li, Qiaoqiao and Wang, Weilan and Zhao, Dongdong},
  title = {ASCPR-Net: Adaptive Structure Content Prior Learning and Style-Consistent Reconstruction for Dunhuang Mural Inpainting},
  year = {2026},
  note = {Manuscript}
}
```

## Acknowledgements

We thank the authors and contributors of the following open-source repositories for their valuable work:

- [MuralDH](https://github.com/tearsheaven/MuralDH)
- [mural-image-inpainting](https://github.com/qinnzou/mural-image-inpainting)

## License

This project is released under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.
