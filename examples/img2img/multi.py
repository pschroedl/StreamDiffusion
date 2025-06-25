import glob
import os
import sys
from typing import Literal, Dict, Optional
import torch
from PIL import Image

import fire


sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.wrapper import StreamDiffusionWrapper

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def main(
    input: str = os.path.join(CURRENT_DIR, "..", "..", "images", "inputs"),
    output: str = os.path.join(CURRENT_DIR, "..", "..", "images", "outputs"),
    model_id_or_path: str = "stabilityai/sd-turbo",
    lora_dict: Optional[Dict[str, float]] = None,
    prompt: str = "1girl with brown dog hair, thick glasses, smiling",
    negative_prompt: str = "low quality, bad quality, blurry, low resolution",
    width: int = 512,
    height: int = 512,
    acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
    use_denoising_batch: bool = True,
    guidance_scale: float = 1.2,
    cfg_type: Literal["none", "full", "self", "initialize"] = "self",
    seed: int = 2,
    delta: float = 0.5,
):
    """
    Initializes the StreamDiffusionWrapper.

    Parameters
    ----------
    input : str, optional
        The input directory to load images from.
    output : str, optional
        The output directory to save images to.
    model_id_or_path : str
        The model id or path to load.
    lora_dict : Optional[Dict[str, float]], optional
        The lora_dict to load, by default None.
        Keys are the LoRA names and values are the LoRA scales.
        Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
    prompt : str
        The prompt to generate images from.
    negative_prompt : str, optional
        The negative prompt to use.
    width : int, optional
        The width of the image, by default 512.
    height : int, optional
        The height of the image, by default 512.
    acceleration : Literal["none", "xformers", "tensorrt"], optional
        The acceleration method, by default "tensorrt".
    use_denoising_batch : bool, optional
        Whether to use denoising batch or not, by default True.
    guidance_scale : float, optional
        The CFG scale, by default 1.2.
    cfg_type : Literal["none", "full", "self", "initialize"],
    optional
        The cfg_type for img2img mode, by default "self".
        You cannot use anything other than "none" for txt2img mode.
    seed : int, optional
        The seed, by default 2. if -1, use random seed.
    delta : float, optional
        The delta multiplier of virtual residual noise,
        by default 1.0.
    """

    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)

    if guidance_scale <= 1.0:
        cfg_type = "none"

    frame_buffer_size = 2  # Set desired frame buffer size
    
    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=lora_dict,
        t_index_list=[45],  # Single timestep to simplify
        frame_buffer_size=frame_buffer_size,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        mode="img2img",
        use_denoising_batch=use_denoising_batch,
        cfg_type=cfg_type,
        seed=seed,
    )

    stream.prepare(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=guidance_scale,
        delta=delta,
    )

    images = glob.glob(os.path.join(input, "*"))
    
    # Process images in batches of frame_buffer_size
    def preprocess_batch(image_paths):
        """Preprocess a batch of images into a single tensor"""
        batch_tensors = []
        for path in image_paths:
            img = Image.open(path).convert("RGB").resize((width, height))
            tensor = stream.stream.image_processor.preprocess(img, height, width)
            batch_tensors.append(tensor)
        return torch.cat(batch_tensors, dim=0).to(device=stream.device, dtype=stream.dtype)
    
    # Pad images if needed to make batches
    while len(images) % frame_buffer_size != 0:
        images.append(images[-1])  # Duplicate last image
    
    # Process images in batches
    for i in range(0, len(images), frame_buffer_size):
        batch_paths = images[i:i + frame_buffer_size]
        
        try:
            # Create batched input tensor
            batch_tensor = preprocess_batch(batch_paths)
            
            # Process the batch
            output_tensor = stream.stream(batch_tensor)
            
            # Post-process and save results
            output_images = stream.postprocess_image(output_tensor, output_type="pil")
            
            # Save each image in the batch
            for j, (path, output_img) in enumerate(zip(batch_paths, output_images)):
                basename = os.path.splitext(os.path.basename(path))[0]
                output_filename = f"{basename}_batch{i//frame_buffer_size}_frame{j}.png"
                output_img.save(os.path.join(output, output_filename))
                print(f"Saved: {output_filename}")
                
        except Exception as e:
            print(f"Error processing batch {i//frame_buffer_size}: {e}")
            continue


if __name__ == "__main__":
    fire.Fire(main)
