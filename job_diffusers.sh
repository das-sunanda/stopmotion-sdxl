#!/bin/bash
#SBATCH --job-name=diffusers_train
#SBATCH --partition qgpu72
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --output=diffusers_train_%j.out
#SBATCH --error=diffusers_train_%j.err

# Load required modules (Singularity and CUDA)
module purge
Module load singularity
module load cuda

# Execute the container with NVIDIA GPU support and bind your home (or other needed) directories.
singularity exec --nv \
    --bind /scrfs/storage/sunandad/home/Text-to-Image-V4:/home/sunandad/Text-to-Image-V4 \
    /scrfs/storage/sunandad/home/Text-to-Image-V4/diffusers.sif \
    accelerate launch --num_processes=4 --multi_gpu \
    /opt/diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
    --dataset_name="taefnadim/my-image-parquet-dataset8" \
    --validation_prompt="A watchmaker repairs an intricate timepiece in his tiny shop, while the bustling city street outside fades into evening" \
    --num_validation_images=4 \
    --validation_epochs=1 \
    --output_dir="./text-to-image-gpu" \
    --resolution=1024 \
    --center_crop \
    --random_flip \
    --train_text_encoder \
    --train_batch_size=1 \
    --num_train_epochs=250 \
    --checkpointing_steps=250 \
    --gradient_accumulation_steps=4 \
    --learning_rate=1e-04 \
    --lr_warmup_steps=0 \
    --report_to="wandb" \
    --dataloader_num_workers=8 \
    --allow_tf32 \
    --mixed_precision="fp16" \
    --push_to_hub \
    --hub_model_id="new-text-to-image-v6"


