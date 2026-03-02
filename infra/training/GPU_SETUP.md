# Akshara OCR GPU Setup Guide

This guide explains how to spin up a GPU instance on RunPod to train the Akshara OCR models natively using the provided `Dockerfile.training`.

## 1. Select a GPU Tier

- **NVIDIA A100 (80GB)**: Best if you want the absolute fastest training time and are training on the full corpus of data with large image sizes. Ideal for large batch sizes.
- **NVIDIA RTX 3090 (24GB) / RTX 4090 (24GB)**: Extremely cost-effective (often $0.40 - $0.70/hr) and perfect for rapid experimentation and debugging. Recommended default.

## 2. Setting Up the RunPod Environment

1. When deploying, select the **RunPod PyTorch** base image (e.g. `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`).
2. Give your pod at least 50-100GB of Volume Drive space to hold the datalake.
3. Expose Port `8888` (for Jupyter) if you need debugging.
4. Set the `WANDB_API_KEY` Environment Variable during Pod configuration securely.

## 3. Install the Training Docker Image

Although RunPod has a PyTorch image, you can securely build and operate within our `Dockerfile.training`. Once connected to the Pod terminal:

```bash
git clone https://github.com/JayHire06/akshara-ocr.git
cd akshara-ocr
docker build -t akshara-training -f infra/training/Dockerfile.training .
```

*Note: Alternatively, you can run `train_job.sh` directly on the RunPod PyTorch base image as it already ships with CUDA and python installed.*

## 4. Mount the DVC Dataset

The remote dataset configuration must be downloaded before training happens. If you correctly configured `dvc-s3` credentials, you can pull the dataset directly in the startup:

```bash
# Add your AWS credentials for DVC
export AWS_ACCESS_KEY_ID="your_key"
export AWS_SECRET_ACCESS_KEY="your_secret"

dvc pull
```

Our script `infra/training/train_job.sh` will inherently call `dvc pull` if the `.dvc` configurations are located in the repository root.

## 5. Start and Monitor the Training Job

We provide a convenient shell script to commence the run. 

```bash
bash infra/training/train_job.sh
```

### Monitoring via Weights & Biases
The system relies on Weights & Biases (W&B) for all log aggregation:
1. Validate that the training script `model/train.py` reports initialization.
2. Open your [W&B dashboard](https://wandb.ai/).
3. A new project automatically synchronizes loss curves, gradients, hardware GPU temperatures, memory utilization, and learning rates on RunPod in real time.
