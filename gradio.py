import os
os.environ["GRADIO_TEMP_DIR"] = "/home/sunandad/gradio_tmp"

import gradio as gr
import torch, gc, re
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import DiffusionPipeline

def clear_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# Load OpenChat (keep on CPU)
openchat_tokenizer = AutoTokenizer.from_pretrained("openchat/openchat-3.5-1210")
openchat_model = AutoModelForCausalLM.from_pretrained(
    "openchat/openchat-3.5-1210", torch_dtype=torch.float16
).to("cpu")

# Load SDXL base + LoRA (initially on CPU)
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
    variant="fp16", use_safetensors=True
).to("cpu")
base.load_lora_weights("/home/sunandad/Text-to-Image-V7/Test2/pytorch_lora_weights.safetensors")

# Load refiner (on CPU initially)
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
).to("cpu")

# === Suggestion Generator ===
def get_prompt_suggestions(prompt):
    clear_cuda()

    base.to("cpu"); refiner.to("cpu"); clear_cuda()
    openchat_model.to("cuda")
    
    input_text = (
        "You are a creative assistant for prompt generation.\n"
        f"User: Extend the '{prompt}' and generate four similar sentence maintaining contextuality. Number them 1 to 4.\nAssistant:"
    )
    input_ids = openchat_tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

    with torch.no_grad():
        output_ids = openchat_model.generate(
            input_ids, max_new_tokens=300, temperature=0.9, top_p=0.95,
            do_sample=True, eos_token_id=openchat_tokenizer.eos_token_id
        )

    reply = openchat_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    openchat_model.to("cpu"); clear_cuda()

    suggestions = re.findall(r"\d[\.\)]\s*(.+)", reply.split("Assistant:")[-1])
    if len(suggestions) < 4:
        suggestions = [line.strip("0123456789. -") for line in reply.split("\n") if line.strip()]
    
    # Append original user prompt as 5th suggestion
    suggestions = suggestions[:4] + [prompt]
    return suggestions

# === Image Generator ===
def generate_images(prompt, steps, guidance, high_noise_frac):
    base.to("cuda"); refiner.to("cuda"); clear_cuda()

    images = []
    for _ in range(4):
        latent = base(
            prompt=prompt,
            num_inference_steps=steps,
            denoising_end=high_noise_frac,
            output_type="latent"
        ).images
        img = refiner(
            prompt=prompt,
            num_inference_steps=steps,
            denoising_start=high_noise_frac,
            image=latent
        ).images[0]
        images.append(img)

    return images

with gr.Blocks() as demo:
    gr.Markdown("## Stable Diffusion + LoRA: Text to Image Generation")

    with gr.Row():
        input_prompt = gr.Textbox(label="Your idea", placeholder="e.g., A lion standing on a cliff at sunset")
        suggest_btn = gr.Button("Suggest Prompts")

    suggestions = gr.Radio(choices=[], label="Choose a Suggested Prompt", interactive=True)

    with gr.Row():
        steps_slider = gr.Slider(10, 100, value=50, step=1, label="Number of Steps")
        guidance_slider = gr.Slider(1.0, 20.0, value=7.0, step=0.5, label="Guidance Scale")
        noise_slider = gr.Slider(0.0, 1.0, value=0.8, step=0.1, label="High Noise Fraction")

    generate_btn = gr.Button("Generate Images")
    gallery = gr.Gallery(label="Generated Images", columns=2)

    # Suggestion trigger
    def handle_suggestions(user_prompt):
        return gr.update(choices=get_prompt_suggestions(user_prompt), value=None)

    suggest_btn.click(fn=handle_suggestions, inputs=input_prompt, outputs=suggestions)

    generate_btn.click(
        fn=generate_images,
        inputs=[suggestions, steps_slider, guidance_slider, noise_slider],
        outputs=gallery
    )

# === Launch ===
demo.launch(server_name="0.0.0.0", server_port=7891, share=True)
