import torch
import lpips
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
from diffusers import DiffusionPipeline
from tqdm import tqdm

def compute_lpips_scores(generated_images, reference_images, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])

    loss_fn = lpips.LPIPS(net='vgg').to(device)  # or 'vgg', alex

    scores = []
    print("Computing LPIPS scores...")
    for gen_img, ref_img in tqdm(zip(generated_images, reference_images), total=len(generated_images)):
        img1 = transform(gen_img).unsqueeze(0).to(device)
        img2 = transform(ref_img).unsqueeze(0).to(device)
        with torch.no_grad():
            score = loss_fn(img1, img2)
        scores.append(score.item())

    return torch.tensor(scores)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step: Load and sample the dataset
    ds = load_dataset("taefnadim/my-image-parquet-dataset8", split="train")
    sampled = ds.train_test_split(train_size=.2, seed=45)["train"]

    subset = sampled
    prompts = subset["text"]
    reference_images = [Image.open(img) if isinstance(img, str) else img for img in subset["image"]]

    # Step: Load Stable Diffusion XL base + LoRA
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

    # Step: Load refiner pipeline
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    ).to(device)
    refiner.enable_model_cpu_offload()

    print("Pipelines loaded successfully.\n")

    # Step: Generate and refine images
    generated_images = []
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
        generated_images.append(ref_out.images[0])

    print(f"Generated {len(generated_images)} images.\n")

    # Step: Compute LPIPS
    lpips_scores = compute_lpips_scores(generated_images, reference_images, device)
    print(f"\nEvaluated {len(lpips_scores)} image pairs.")
    print(f"Mean LPIPS: {lpips_scores.mean().item():.4f}")
    print(f" Std LPIPS: {lpips_scores.std().item():.4f}")

if __name__ == "__main__":
    main()
