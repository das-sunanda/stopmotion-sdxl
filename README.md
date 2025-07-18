# Fine-Tuning Stable Diffusion for Stop-Motion Style Synthesis from Text Prompts

## Overview
This project presents a complete pipeline for fine-tuning the Stable Diffusion XL (SDXL) model to generate images in the unique stop-motion animation style, inspired by the film Memoir of a Snail. The pipeline covers custom dataset creation, environment setup using Singularity for scalable and reproducible multi-GPU training, LoRA-based efficient fine-tuning, and user-friendly deployment with a Gradio web app. Evaluation uses both semantic (CLIP Score) and perceptual (LPIPS) metrics, and the project achieves state-of-the-art prompt alignment and visual fidelity for stylized stop-motion image synthesis.

## Features
- **Custom Dataset:** 1000+ hand-labeled image-prompt pairs in stop-motion style, covering diverse scenes and emotions.
- **Scalable Training:** Singularity container supports distributed, multi-GPU fine-tuning with PyTorch and Hugging Face diffusers.
- **Efficient Fine-Tuning:** Uses Low-Rank Adaptation (LoRA) to minimize trainable parameters while maximizing style transfer and prompt alignment.
- **Interactive Demo:** Gradio web app with prompt engineering via LLM, user-adjustable parameters (steps, guidance scale, noise fraction), and high-res output.
- **Reproducible Workflows:** All training and inference steps are fully containerized and compatible with SLURM HPC clusters.

## Dataset
- Curated from the film Memoir of a Snail (2025), frames sampled every ~5 seconds and labeled with neutral, descriptive prompts.
- Format: Hugging Face parquet format
- 
