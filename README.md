# Brain Tumor Segmentation — BraTS 2023

3D brain tumor segmentation on multimodal MRI data using MONAI and PyTorch, with a full MLOps pipeline including experiment tracking, CI/CD, containerized inference, and an interactive demo.

---

## Overview

This project trains a 3D U-Net on the BraTS 2023 dataset to segment brain tumor subregions from multimodal MRI volumes (T1, T1ce, T2, FLAIR). The model predicts three clinically meaningful subregions:

- **Whole Tumor (WT)** — full tumor extent
- **Tumor Core (TC)** — necrotic core and enhancing tissue
- **Enhancing Tumor (ET)** — actively growing tumor region

Beyond model training, the project implements a production-style MLOps pipeline: experiment tracking, automated evaluation, containerized inference, and a live interactive demo where users can upload an MRI volume and visualize predicted segmentation masks across axial, coronal, and sagittal planes.

---

## Demo

> 🔗 Live demo link — coming soon (Hugging Face Spaces)

---

## Results

| Subregion | Dice Score | Hausdorff95 |
|---|---|---|
| Whole Tumor (WT) | TBD | TBD |
| Tumor Core (TC) | TBD | TBD |
| Enhancing Tumor (ET) | TBD | TBD |

*Evaluated on BraTS 2023 held-out validation split. Comparison to published BraTS leaderboard baselines documented in `/notebooks/evaluation.ipynb`.*

---

## Architecture

- **Model:** 3D U-Net with residual connections (MONAI)
- **Input:** 4-channel multimodal MRI volume (T1, T1ce, T2, FLAIR), normalized and skull-stripped
- **Output:** 3-class segmentation mask (WT / TC / ET)
- **Loss:** Dice loss + cross-entropy (combined)
- **Inference:** Sliding window inference over full volume

---

## Project Structure

```
brats-tumor-segmentation/
├── data/                   # gitignored — BraTS NIfTI volumes
├── src/
│   ├── preprocessing/      # skull stripping, bias correction, normalization
│   ├── training/           # model definition, training loop, loss functions
│   ├── inference/          # sliding window inference, postprocessing
│   └── utils/              # shared utilities, visualization helpers
├── notebooks/              # exploration, training walkthrough, evaluation
├── configs/                # YAML training configs
├── tests/                  # pytest unit tests
├── docker/                 # Dockerfile + compose for inference service
├── .github/workflows/      # GitHub Actions CI/CD
├── requirements.txt
└── README.md
```

---

## MLOps Pipeline

- **Experiment tracking:** MLflow — logs hyperparameters, per-epoch metrics, Dice scores, and sample segmentation visualizations as artifacts
- **Model registry:** MLflow model registry with version tags and metric annotations
- **CI/CD:** GitHub Actions — runs linting, unit tests, and model evaluation on every PR
- **Containerization:** Docker — reproducible inference environment
- **Deployment:** Hugging Face Spaces — publicly accessible live demo

---

## Setup & Reproducibility

### Requirements

- Python 3.11
- See `requirements.txt` for full dependency list

### Installation

```bash
git clone https://github.com/AndrewVFranco/brats-tumor-segmentation.git
cd brats-tumor-segmentation
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data

Download BraTS 2023 data from [Synapse](https://www.synapse.org/) (free registration required). Place volumes in `/data/raw/`. See `/notebooks/01_data_exploration.ipynb` for expected directory structure.

### Training

```bash
python src/training/train.py --config configs/train_config.yaml
```

### Inference

```bash
python src/inference/predict.py --input path/to/volume.nii.gz --output path/to/output/
```

---

## Dataset

**BraTS 2023** (Brain Tumor Segmentation Challenge)
- Hosted via Synapse (RSNA-ASNR-MICCAI)
- ~1,200 multimodal MRI cases with expert annotations
- Four MRI modalities per case: T1, T1ce, T2, FLAIR
- Labels: background, necrotic core, peritumoral edema, enhancing tumor

---

## Background

This project was developed as part of a portfolio demonstrating full-stack ML engineering competency in clinical medical imaging. Bringing 8+ years of clinical experience in cardiac telemetry monitoring, informing the design of the system with real-world awareness of clinical workflow constraints and patient safety considerations.

---

## License

MIT License — see `LICENSE` for details.