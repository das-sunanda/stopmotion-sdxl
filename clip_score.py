import torch
from datasets import load_dataset
from diffusers import DiffusionPipeline
from transformers import CLIPModel, CLIPProcessor

def compute_clip_scores(images, texts, model, processor, device):
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
    outputs    = model(**inputs)
    img_emb    = outputs.image_embeds
    txt_emb    = outputs.text_embeds
    img_emb    = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
    txt_emb    = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)
    cosines    = (img_emb * txt_emb).sum(dim=-1)
    return torch.clamp(cosines * 100, min=0)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load & sample your dataset
    ds      = load_dataset("taefnadim/my-image-parquet-dataset8", split="train")
    sampled = ds.train_test_split(train_size=0.2, seed=45)["train"]
    prompts = sampled["text"]

    # 2) Load base+LoRA pipeline
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to(device)
    base.enable_model_cpu_offload()
    base.load_lora_weights(
        "/home/sunandad/Text-to-Image-V7/Test",
        weight_name="pytorch_lora_weights.safetensors",
        adapter_name="my_lora"
    )

    # 3) Load refiner, reusing text_encoder_2 and VAE from base
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    ).to(device)
    refiner.enable_model_cpu_offload()

    print("Pipelines loaded&\n")

    # 4) Generate + refine images
    images = []
    print("Generating and refining images for prompts...\n")
    for prompt in prompts:
        init_out = base(
            prompt=prompt,
            num_inference_steps=50,
            height=1024,
            width=1024,
            guidance_scale=7.0
        )
        init_img = init_out.images[0]  

        ref_out = refiner(
            prompt=prompt,
            image=init_img,
            num_inference_steps=40,      
            guidance_scale=6.0           
        )
        images.append(ref_out.images[0])

    print(f"Generated + refined {len(images)} images....\n")

    # 5) Load CLIP for scoring
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 6) Compute CLIPScore
    print("Computing CLIPScore...\n")
    scores = compute_clip_scores(images, prompts, clip_model, clip_proc, device)

    print(f"Evaluated {len(prompts)} samples.")
    print(f"Mean CLIPScore: {scores.mean().item():.4f}")
    print(f" Std CLIPScore: {scores.std().item():.4f}")

if __name__ == "__main__":
    main()

