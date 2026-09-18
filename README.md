# ASLInterpreter

A sign-language recognition project for American Sign Language (ASL) using pose and hand landmark sequences, a spatial-temporal transformer model, and live webcam inference.

This repository builds a gesture recognition pipeline around MediaPipe landmark extraction, skeleton-aware feature preprocessing, and a transformer-based classifier that predicts gloss labels from temporal motion sequences.

## Overview

The project is organized into three main stages:

1. Data preprocessing and keypoint extraction
2. Model training and evaluation
3. Real-time inference from webcam video

The model operates on temporal skeleton sequences represented as joint positions and masks, with additional bone and velocity features. A graph-based adjacency structure is used to model skeletal relationships between body joints and hand landmarks.

## Key features

- MediaPipe pose + hand landmark extraction
- Temporal windowing and motion-aware clip selection
- Body-relative normalization and hand-to-body stitching
- Spatial-temporal transformer model with adjacency-aware attention
- Gloss vocabulary generation from processed dataset splits
- Live webcam recognition pipeline
- Training scripts for evaluation and checkpointing

## Project structure

```text
ASLInterpreter/
├── data/
│   ├── ASL_Citizen/
│   ├── videos/
│   ├── processed/
│   ├── combined_dict.json
│   ├── failed_videos.txt
│   └── ...
├── live/
│   ├── build_gloss.py
│   ├── gloss_vocab.json
│   └── live_infer.py
├── mediapipe_models/
│   ├── hand_landmarker.task
│   └── pose_landmarker_full.task
├── model/
│   ├── spatial_attention.py
│   └── st_pose_model.py
├── preprocess/
│   ├── annotate.py
│   ├── extract.py
├── scripts/
│   ├── blank.py
│   ├── converter.py
│   ├── create_blank_class.py
│   ├── data_checkv2.py
│   ├── ...
├── training/
│   ├── dataset.py
│   └── train.py
├── utils/
│   ├── skeleton.py
│   └── velocity.py
├── model_files/
│   ├── best_model.pt
│   ├── best_model_combined.pt
│   └── best_model_w_blanks.pt
├── requirements.txt
├── test.py
├── README.md
└── .venv/
```

## Requirements

The repository expects a Python environment with PyTorch, OpenCV, MediaPipe, and NumPy installed.

A local virtual environment is recommended:

```bash
python -m venv .venv
. .venv/bin/activate   # Linux/macOS
# or
.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
```

Notes:

- The environment in this repo is configured for a local Windows setup and includes CPU-targeted PyTorch packages.
- MediaPipe model files are expected in the `mediapipe_models/` directory.
- If you are training on GPU, make sure the PyTorch build matches the correct CUDA/XPU driver configuration.

## Data layout

The project expects processed and raw data in the following structure:

- `data/ASL_Citizen/videos/` — ASL Citizen source videos
- `data/videos/` — additional video sources
- `data/processed/` — extracted landmark `.npy` arrays and mask arrays
- `data/combined_dict.json` — dataset metadata / gloss mapping
- `live/gloss_vocab.json` — generated gloss-to-index vocabulary used at inference time

WLASL videos and ASL Citizen videos must be downloaded separately to be used for training.

## Preprocessing pipeline

The extraction step reads videos, samples windows of motion, runs MediaPipe pose and hand detection, and saves landmark tensors for model training.

Run:

```bash
python preprocess/extract.py
```

This script scans the raw video directories, extracts body/hand keypoints, and saves processed arrays such as:

- `data/processed/<video_id>.npy`
- `data/processed/<video_id>_mask.npy`

## Gloss vocabulary

Before running live inference, generate or refresh the label vocabulary:

```bash
python live/build_gloss.py
```

This creates a mapping between gloss names and model output indices in `live/gloss_vocab.json`.

## Training

Training is handled by `training/train.py`.

Basic usage:

```bash
python training/train.py
```

Useful options:

```bash
python training/train.py --eval-only --checkpoint model_files/best_model_w_blanks.pt
```

This script:

- loads the processed dataset
- builds the skeleton adjacency matrix
- creates the `STPoseModel`
- trains with cross-entropy loss
- validates and saves the best checkpoint
- reports final top-1 and top-5 accuracy

Checkpoint files are written to the `model_files/` directory.

## Live inference

The real-time webcam app is implemented in `live/live_infer.py`.

Run:

```bash
python live/live_infer.py
```

What happens:

- opens the webcam
- extracts pose and hand landmarks frame-by-frame
- detects start/end of a sign using motion scores
- fits the segment into the model window length
- loads the trained checkpoint and gloss vocabulary
- predicts the gloss label and displays it on-screen

The script is designed around a fixed window size and a motion-triggered segmentation flow, so it is suitable for on-the-fly recognition rather than processing a full uploaded video.

## Model notes

The model is a skeleton-aware spatiotemporal transformer built around:

- spatial attention with adjacency bias
- causal temporal attention
- joint validity masking
- motion-aware normalization
- bone and velocity features

The core architecture is defined in:

- `model/st_pose_model.py`
- `model/spatial_attention.py`
- `utils/skeleton.py`
- `utils/velocity.py`

The current best model uploaded is `best_model_w_blanks.pt` which achieves:
- Top-1 Accuracy: 51.39%
- Top-5 Accuracy: 79.47%

These results are across 724 glosses/classes using 11,683 test test videos (32,256 videos total across train, validation, and test splits).

## Useful scripts

Additional utilities in the repo include:

- `scripts/preprocess_check.py`: verify preprocessing outputs
- `scripts/reprocess_failed.py`: retry failed video extraction
- `scripts/try_failed_downloads.py`: handle failed dataset fetches
- `scripts/converter.py`: format conversion utilities
- `scripts/blank.py`: blank-frame handling / dataset balancing utilities

See the `scripts/` folder for project-specific maintenance tasks.

## Troubleshooting

### Missing processed data

If the model or vocabulary cannot load correctly, make sure the following exist:

```bash
ls data/processed
ls live
ls model_files
```

If `data/processed` is empty, run the extraction script first.

### Missing MediaPipe models

Ensure the following files exist:

```text
mediapipe_models/hand_landmarker.task
mediapipe_models/pose_landmarker_full.task
```

### Wrong vocabulary / class mismatch

If the checkpoint output size does not match the gloss vocabulary, regenerate the vocabulary with:

```bash
python live/build_gloss.py
```

### Webcam not opening

Verify that:

- your webcam is connected and not already in use
- OpenCV can access the camera device
- the environment is using the expected Python interpreter
- the correct camera is selected in code (i.e. camera 0, camera 1, etc.)

## Typical workflow

```bash
# 1) create environment
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# 2) preprocess data
python preprocess/extract.py

# 3) build gloss vocabulary
python live/build_gloss.py

# 4) train model
python training/train.py

# 5) run live inference
python live/live_infer.py
```

## Summary

This repo is a full-stack ASL recognition prototype that takes raw sign videos, extracts pose and hand keypoints, trains a spatiotemporal transformer on the resulting skeleton sequences, and then performs live gloss inference from a webcam feed.
