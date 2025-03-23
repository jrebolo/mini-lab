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
        
        # Setup device for Mac MPS
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA (NVIDIA GPU)")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")
        
        # Initialize pipeline with float32 for MPS compatibility
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            safety_checker=None
        ).to(self.device)
        
        # Enable optimization
        self.pipe.enable_attention_slicing()
        
        print(f"Model loaded on: {self.device}")

    def generate_interpolated_prompts(self, start_prompt, end_prompt, num_frames):
        """Generate interpolated prompts between start and end prompts"""
        prompts = []
        for i in range(num_frames):
            t = i / (num_frames - 1)
            # Simple linear interpolation between prompts
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
        num_inference_steps=20,  # Reduced steps for MPS
        guidance_scale=7.5,
        size=(512, 512)
    ):
        """Generate frames using the provided prompts"""
        torch.manual_seed(seed)
        
        for i, prompt in enumerate(tqdm(prompts, desc="Generating frames")):
            try:
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
                
            except RuntimeError as e:
                print(f"Error generating frame {i}: {str(e)}")
                # Fallback to CPU if MPS encounters an error
                if str(self.device) == "mps":
                    print("Attempting CPU fallback for this frame...")
                    self.pipe = self.pipe.to("cpu")
                    image = self.pipe(
                        prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        width=size[0],
                        height=size[1]
                    ).images[0]
                    self.pipe = self.pipe.to(self.device)
                    frame_path = os.path.join(self.frames_dir, f"frame_{i:04d}.png")
                    image.save(frame_path)

    def create_video(self, fps=12, output_filename="animation.mp4"):
        """Combine frames into a video"""
        frames = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.png')])
        if not frames:
            raise ValueError("No frames found to create video")

        first_frame = cv2.imread(os.path.join(self.frames_dir, frames[0]))
        height, width, _ = first_frame.shape

        output_path = os.path.join(self.output_dir, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame_name in tqdm(frames, desc="Creating video"):
            frame_path = os.path.join(self.frames_dir, frame_name)
            frame = cv2.imread(frame_path)
            video_writer.write(frame)

        video_writer.release()
        print(f"Video saved to: {output_path}")
        
        # For Mac users: Convert video to be playable in QuickTime
        try:
            converted_path = output_path.replace('.mp4', '_converted.mp4')
            os.system(f"ffmpeg -i {output_path} -vcodec h264 {converted_path}")
            print(f"Converted video saved to: {converted_path}")
        except:
            print("Note: Install ffmpeg to create QuickTime-compatible videos")

def create_animation(
    start_prompt,
    end_prompt,
    duration_seconds=3,
    fps=12,  # Reduced FPS for faster generation
    model_id="runwayml/stable-diffusion-v1-5",
    output_dir="./animation"
):
    """Main function to create an animation"""
    num_frames = int(duration_seconds * fps)
    generator = AnimationGenerator(model_id=model_id, output_dir=output_dir)
    prompts = generator.generate_interpolated_prompts(start_prompt, end_prompt, num_frames)
    generator.generate_frames(prompts)
    generator.create_video(fps=fps)

# Example usage
if __name__ == "__main__":
    start_prompt = "a young witch flying on a broomstick over a peaceful coastal town"
    end_prompt = "a peaceful Japanese garden in autumn, studio ghibli style"
    
    create_animation(
        start_prompt=start_prompt,
        end_prompt=end_prompt,
        duration_seconds=3,
        fps=12,
        model_id="nitrosocke/Ghibli-Diffusion", 
        output_dir="./animation/ghibli_art"
    )