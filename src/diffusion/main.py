# First install required packages
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import argparse

def setup_stable_diffusion(model_id="runwayml/stable-diffusion-v1-5", output_dir="./generated_images"):
    """
    Setup and run Stable Diffusion locally
    Args:
        model_id: HuggingFace model ID to use
        output_dir: Directory to save generated images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if CUDA is available and set device

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (NVIDIA GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

   
    # Load the pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None  # Optional: disable safety checker for speed
    )
    pipe = pipe.to(device)
    
    # Enable memory efficient attention if using CUDA
    if device == "cuda":
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    
    return pipe

def generate_image(pipe, prompt, output_dir="./generated_images", num_inference_steps=50):
    """
    Generate an image using the provided pipeline
    """
    # Generate the image
    image = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=7.5,
    ).images[0]
    
    # Save the image
    output_path = os.path.join(output_dir, f"{prompt[:30].replace(' ', '_')}.png")
    image.save(output_path)
    print(f"Image saved to: {output_path}")
    return image


def generate_ghibli_image(
    pipe,
    subject_prompt,
    output_dir="./ghibli_art",
    num_inference_steps=30,
    guidance_scale=7.5,
    size=(512, 512)
):
    """
    Generate a Ghibli-style image with enhanced prompting
    """
    # Construct a detailed Ghibli-style prompt
    ghibli_style_prompt = (
        f"{subject_prompt}, "
        "studio ghibli style, "
        "hayao miyazaki, "
        "soft lighting, "
        "gentle colors, "
        "hand-drawn animation style, "
        "detailed background, "
        "2d animation, "
        "anime key animation, "
        "film grain, "
        "painted texture"
    )
    
    # Negative prompt to avoid common issues
    negative_prompt = (
        "3d, cgi, photorealistic, photo, photography, "
        "ugly, deformed, noisy, blurry, low quality, "
        "deep fried, oversaturated"
    )
    
    # Generate image
    image = pipe(
        prompt=ghibli_style_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=size[0],
        height=size[1]
    ).images[0]
    
    # Save the image
    output_path = os.path.join(output_dir, f"ghibli_{subject_prompt[:30].replace(' ', '_')}.png")
    image.save(output_path)
    print(f"Image saved to: {output_path}")
    return image

# Example usage
if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser(description="Generate images using stable diffusion")
    parser.add_argument("--scene", action="store_true", help="Enable scene mode")
    parser.add_argument("--ghibli", action="store_true", help="Enable ghibli mode")
    args = parser.parse_args()
    print("Arguments:", vars(args))
    if args.ghibli:
        model_id = "nitrosocke/Ghibli-Diffusion"
    else:
        model_id = "runwayml/stable-diffusion-v1-5"

    # Setup the model
    pipe = setup_stable_diffusion(model_id=model_id, output_dir="./ghibli_art")
    SCENE = args.scene
    SIMPLE_PROMPT = not args.scene
    if SIMPLE_PROMPT:
        # Generate an image
        prompt = "a beautiful sunset over mountains, hyperrealistic"
        print(prompt)
        generate_image(pipe, prompt)
    elif SCENE:
        scenes = [
        "a young witch flying on a broomstick over a peaceful coastal town",
        "a magical forest spirit sitting under a giant tree",
        "a cozy countryside house with a vegetable garden",
        "a cat sleeping on a windowsill of a bakery"
        ]
    
        # Generate multiple scenes
        for scene in scenes:
            generate_ghibli_image(pipe, scene)