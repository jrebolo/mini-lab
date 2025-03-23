import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import cv2
import numpy as np
import os
from tqdm import tqdm

class AnimationGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", output_dir="./animation"):
        self.output_dir = output_dir
        self.frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        
        # Setup device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)
        
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
            self.pipe.enable_vae_slicing()

    def generate_interpolated_prompts(self, start_prompt, end_prompt, num_frames):
        """Generate interpolated prompts between start and end prompts"""
        prompts = []
        for i in range(num_frames):
            t = i / (num_frames - 1)
            # For simplicity, we're just doing linear interpolation of the text
            # You could implement more sophisticated prompt interpolation here
            if t < 0.5:
                prompt = start_prompt
            else:
                prompt = end_prompt
            prompts.append(prompt)
        return prompts

    def generate_frames(
        self,
        prompts,
        seed=42,
        num_inference_steps=30,
        guidance_scale=7.5,
        size=(512, 512)
    ):
        """Generate frames using the provided prompts"""
        # Set seed for reproducibility
        torch.manual_seed(seed)
        
        for i, prompt in enumerate(tqdm(prompts, desc="Generating frames")):
            # Generate image
            image = self.pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=size[0],
                height=size[1]
            ).images[0]
            
            # Save frame
            frame_path = os.path.join(self.frames_dir, f"frame_{i:04d}.png")
            image.save(frame_path)

    def create_video(self, fps=24, output_filename="animation.mp4"):
        """Combine frames into a video"""
        frames = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.png')])
        if not frames:
            raise ValueError("No frames found to create video")

        # Read first frame to get dimensions
        first_frame = cv2.imread(os.path.join(self.frames_dir, frames[0]))
        height, width, _ = first_frame.shape

        # Initialize video writer
        output_path = os.path.join(self.output_dir, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Write frames to video
        for frame_name in tqdm(frames, desc="Creating video"):
            frame_path = os.path.join(self.frames_dir, frame_name)
            frame = cv2.imread(frame_path)
            video_writer.write(frame)

        video_writer.release()
        print(f"Video saved to: {output_path}")

def create_animation(
    start_prompt,
    end_prompt,
    duration_seconds=3,
    fps=24,
    model_id="runwayml/stable-diffusion-v1-5",
    output_dir="./animation"
):
    """Main function to create an animation"""
    # Calculate number of frames
    num_frames = int(duration_seconds * fps)
    
    # Initialize generator
    generator = AnimationGenerator(model_id=model_id, output_dir=output_dir)
    
    # Generate prompts for each frame
    prompts = generator.generate_interpolated_prompts(start_prompt, end_prompt, num_frames)
    
    # Generate frames
    generator.generate_frames(prompts)
    
    # Create video
    generator.create_video(fps=fps)

# Example usage
if __name__ == "__main__":
    start_prompt = "a serene lake at sunrise, studio ghibli style"
    end_prompt = "a serene lake at sunset, studio ghibli style"
    
    create_animation(
        start_prompt=start_prompt,
        end_prompt=end_prompt,
        duration_seconds=3,
        fps=12
    )