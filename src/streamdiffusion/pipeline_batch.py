import time
from typing import List, Optional, Union, Any, Dict, Tuple, Literal

import numpy as np
import PIL.Image
import torch
from diffusers import LCMScheduler, StableDiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)

from streamdiffusion.image_filter import SimilarImageFilter
from streamdiffusion.stream_parameter_updater import StreamParameterUpdater

class StreamDiffusion:
    def __init__(
        self,
        pipe: StableDiffusionPipeline,
        t_index_list: List[int],
        torch_dtype: torch.dtype = torch.float16,
        width: int = 512,
        height: int = 512,
        do_add_noise: bool = True,
        use_denoising_batch: bool = True,
        frame_buffer_size: int = 1,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
    ) -> None:
        self.device = pipe.device
        self.dtype = torch_dtype
        self.generator = None

        self.height = height
        self.width = width

        self.latent_height = int(height // pipe.vae_scale_factor)
        self.latent_width = int(width // pipe.vae_scale_factor)

        self.frame_bff_size = frame_buffer_size
        self.denoising_steps_num = len(t_index_list)

        self.cfg_type = cfg_type

        if use_denoising_batch:
            self.batch_size = self.denoising_steps_num * frame_buffer_size
            if self.cfg_type == "initialize":
                self.trt_unet_batch_size = (
                    self.denoising_steps_num + 1
                ) * self.frame_bff_size
            elif self.cfg_type == "full":
                self.trt_unet_batch_size = (
                    2 * self.denoising_steps_num * self.frame_bff_size
                )
            else:
                self.trt_unet_batch_size = self.denoising_steps_num * frame_buffer_size
        else:
            self.trt_unet_batch_size = self.frame_bff_size
            self.batch_size = frame_buffer_size

        self.t_list = t_index_list

        self.do_add_noise = do_add_noise
        self.use_denoising_batch = use_denoising_batch

        self.similar_image_filter = False
        self.similar_filter = SimilarImageFilter()
        self.prev_image_result = None

        self.pipe = pipe
        self.image_processor = VaeImageProcessor(pipe.vae_scale_factor)

        self.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.vae = pipe.vae
        
        self.x_t_latent_buffer = None

        self.inference_time_ema = 0
        self._param_updater = StreamParameterUpdater(self)

    def get_stream_status(self):
        """Debug method to get current stream processing status"""
        return {
            'images_in_pipeline': len(self.x_t_latent_buffer) if self.x_t_latent_buffer is not None else 0,
            'total_capacity': self.frame_bff_size,
            'denoising_steps': self.denoising_steps_num,
            'stream_initialized': self.x_t_latent_buffer is not None
        }

    def load_lcm_lora(
        self,
        pretrained_model_name_or_path_or_dict: Union[
            str, Dict[str, torch.Tensor]
        ] = "latent-consistency/lcm-lora-sdv1-5",
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:
        self.pipe.load_lora_weights(
            pretrained_model_name_or_path_or_dict, adapter_name, **kwargs
        )

    def load_lora(
        self,
        pretrained_lora_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:
        self.pipe.load_lora_weights(
            pretrained_lora_model_name_or_path_or_dict, adapter_name, **kwargs
        )

    def fuse_lora(
        self,
        fuse_unet: bool = True,
        fuse_text_encoder: bool = True,
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
    ) -> None:
        self.pipe.fuse_lora(
            fuse_unet=fuse_unet,
            fuse_text_encoder=fuse_text_encoder,
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
        )

    def enable_similar_image_filter(self, threshold: float = 0.98, max_skip_frame: float = 10) -> None:
        self.similar_image_filter = True
        self.similar_filter.set_threshold(threshold)
        self.similar_filter.set_max_skip_frame(max_skip_frame)

    def disable_similar_image_filter(self) -> None:
        self.similar_image_filter = False

    @torch.no_grad()
    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
        generator: Optional[torch.Generator] = torch.Generator(),
        seed: int = 2,
    ) -> None:
        self.generator = generator
        self.generator.manual_seed(seed)
        
        # initialize x_t_latent buffer for multi-stream processing
        if self.denoising_steps_num > 1:
            self.x_t_latent_buffer = torch.zeros(
                (
                    (self.denoising_steps_num - 1) * self.frame_bff_size,
                    4,
                    self.latent_height,
                    self.latent_width,
                ),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.x_t_latent_buffer = None

        if self.cfg_type == "none":
            self.guidance_scale = 1.0
        else:
            self.guidance_scale = guidance_scale
        self.delta = delta

        do_classifier_free_guidance = self.guidance_scale > 1.0

        encoder_output = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )
        self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)

        if self.use_denoising_batch and self.cfg_type == "full":
            uncond_prompt_embeds = encoder_output[1].repeat(self.batch_size, 1, 1)
        elif self.cfg_type == "initialize":
            uncond_prompt_embeds = encoder_output[1].repeat(self.frame_bff_size, 1, 1)

        if self.guidance_scale > 1.0 and (self.cfg_type in ["initialize", "full"]):
            self.prompt_embeds = torch.cat(
                [uncond_prompt_embeds, self.prompt_embeds], dim=0
            )

        self.scheduler.set_timesteps(num_inference_steps, self.device)
        self.timesteps = self.scheduler.timesteps.to(self.device)

        self.sub_timesteps = [self.timesteps[t] for t in self.t_list]
        sub_timesteps_tensor = torch.tensor(self.sub_timesteps, dtype=torch.long, device=self.device)
        self.sub_timesteps_tensor = torch.repeat_interleave(
            sub_timesteps_tensor,
            repeats=self.frame_bff_size,
            dim=0,
        )

        self.init_noise = torch.randn(
            (self.batch_size, 4, self.latent_height, self.latent_width),
            generator=generator,
        ).to(device=self.device, dtype=self.dtype)
        self.stock_noise = torch.zeros_like(self.init_noise)

        c_skip_list, c_out_list = [], []
        for timestep in self.sub_timesteps:
            c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(timestep)
            c_skip_list.append(c_skip)
            c_out_list.append(c_out)

        self.c_skip = torch.stack(c_skip_list).view(len(self.t_list), 1, 1, 1).to(dtype=self.dtype, device=self.device)
        self.c_out = torch.stack(c_out_list).view(len(self.t_list), 1, 1, 1).to(dtype=self.dtype, device=self.device)
        
        self.c_skip = torch.repeat_interleave(self.c_skip, repeats=self.frame_bff_size, dim=0)
        self.c_out = torch.repeat_interleave(self.c_out, repeats=self.frame_bff_size, dim=0)

        alpha_prod_t_sqrt_list, beta_prod_t_sqrt_list = [], []
        for timestep in self.sub_timesteps:
            alpha_prod_t_sqrt = self.scheduler.alphas_cumprod[timestep].sqrt()
            beta_prod_t_sqrt = (1 - self.scheduler.alphas_cumprod[timestep]).sqrt()
            alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
            beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)
            
        alpha_prod_t_sqrt = torch.stack(alpha_prod_t_sqrt_list).view(len(self.t_list), 1, 1, 1).to(dtype=self.dtype, device=self.device)
        beta_prod_t_sqrt = torch.stack(beta_prod_t_sqrt_list).view(len(self.t_list), 1, 1, 1).to(dtype=self.dtype, device=self.device)

        self.alpha_prod_t_sqrt = torch.repeat_interleave(alpha_prod_t_sqrt, repeats=self.frame_bff_size, dim=0)
        self.beta_prod_t_sqrt = torch.repeat_interleave(beta_prod_t_sqrt, repeats=self.frame_bff_size, dim=0)

    @torch.no_grad()
    def update_prompt(self, prompt: str) -> None:
        encoder_output = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)

    @torch.no_grad()
    def update_stream_params(
        self,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        delta: Optional[float] = None,
        t_index_list: Optional[List[int]] = None,
    ) -> None:
        """
        Update streaming parameters efficiently in a single call.
        
        Parameters
        ----------
        num_inference_steps : Optional[int]
            The number of inference steps to perform.
        guidance_scale : Optional[float]
            The guidance scale to use for CFG.
        delta : Optional[float]
            The delta multiplier of virtual residual noise.
        t_index_list : Optional[List[int]]
            The t_index_list to use for inference.
        """
        self._param_updater.update_stream_params(
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            delta=delta,
            t_index_list=t_index_list,
        )

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, t_index: int) -> torch.Tensor:
        return self.alpha_prod_t_sqrt[t_index] * original_samples + self.beta_prod_t_sqrt[t_index] * noise

    def scheduler_step_batch(self, model_pred_batch: torch.Tensor, x_t_latent_batch: torch.Tensor, idx: Optional[int] = None) -> torch.Tensor:
        if idx is None:
            F_theta = (x_t_latent_batch - self.beta_prod_t_sqrt * model_pred_batch) / self.alpha_prod_t_sqrt
            denoised_batch = self.c_out * F_theta + self.c_skip * x_t_latent_batch
        else:
            F_theta = (x_t_latent_batch - self.beta_prod_t_sqrt[idx] * model_pred_batch) / self.alpha_prod_t_sqrt[idx]
            denoised_batch = self.c_out[idx] * F_theta + self.c_skip[idx] * x_t_latent_batch
        return denoised_batch

    def unet_step(self, x_t_latent: torch.Tensor, t_list: Union[torch.Tensor, list[int]], idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x_t_latent_plus_uc = x_t_latent
        if self.guidance_scale > 1.0:
            if self.cfg_type == "initialize":
                x_t_latent_plus_uc = torch.cat([x_t_latent[0:self.frame_bff_size], x_t_latent], dim=0)
                t_list = torch.cat([t_list[0:self.frame_bff_size], t_list], dim=0)
            elif self.cfg_type == "full":
                x_t_latent_plus_uc = torch.cat([x_t_latent, x_t_latent], dim=0)
                t_list = torch.cat([t_list, t_list], dim=0)
        
        model_pred = self.unet(x_t_latent_plus_uc, t_list, encoder_hidden_states=self.prompt_embeds, return_dict=False)[0]

        if self.guidance_scale > 1.0:
            if self.cfg_type == "initialize":
                noise_pred_text = model_pred[self.frame_bff_size:]
                self.stock_noise = torch.cat([model_pred[:self.frame_bff_size], self.stock_noise[self.frame_bff_size:]], dim=0)
            elif self.cfg_type == "full":
                noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
            else: # self
                noise_pred_text = model_pred
            
            if self.cfg_type in ["self", "initialize"]:
                noise_pred_uncond = self.stock_noise * self.delta
            
            model_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)

        if self.use_denoising_batch and self.cfg_type in ["self", "initialize"]:
            scaled_noise = self.beta_prod_t_sqrt * self.stock_noise
            delta_x = self.scheduler_step_batch(model_pred, scaled_noise, idx)
            alpha_next = torch.cat([self.alpha_prod_t_sqrt[self.frame_bff_size:], torch.ones_like(self.alpha_prod_t_sqrt[:self.frame_bff_size])], dim=0)
            delta_x = alpha_next * delta_x
            beta_next = torch.cat([self.beta_prod_t_sqrt[self.frame_bff_size:], torch.ones_like(self.beta_prod_t_sqrt[:self.frame_bff_size])], dim=0)
            delta_x = delta_x / beta_next
            init_noise = torch.cat([self.init_noise[self.frame_bff_size:], self.init_noise[:self.frame_bff_size]], dim=0)
            self.stock_noise = init_noise + delta_x
            
        return denoised_batch, model_pred

    def encode_image(self, image_tensors: torch.Tensor) -> torch.Tensor:
        image_tensors = image_tensors.to(device=self.device, dtype=self.vae.dtype)
        img_latent = retrieve_latents(self.vae.encode(image_tensors), self.generator)
        img_latent = img_latent * self.vae.config.scaling_factor
        x_t_latent = self.add_noise(img_latent, self.init_noise[:image_tensors.shape[0]], 0)
        return x_t_latent

    def decode_image(self, x_0_pred_out: torch.Tensor) -> torch.Tensor:
        output_latent = self.vae.decode(x_0_pred_out / self.vae.config.scaling_factor, return_dict=False)[0]
        return output_latent

    def predict_x0_batch(self, x_t_latent: torch.Tensor) -> torch.Tensor:
        """
        Correct Stream Batch implementation as per the StreamDiffusion paper.
        
        The key insight: Each batch position represents a different image at a different 
        denoising stage. New images enter at stage 0, existing images advance through 
        the pipeline, and completed images exit from the final stage.
        """
        print(f"predict_x0_batch: input x_t_latent shape: {x_t_latent.shape}")
        if not self.use_denoising_batch or self.denoising_steps_num <= 1:
            # Fallback for single-step or non-batch mode
            t_list = self.sub_timesteps_tensor[:self.frame_bff_size]
            x_0_pred_batch, _ = self.unet_step(x_t_latent, t_list)
            return x_0_pred_batch

        # Concatenate new input frames (x_t_latent) with the existing pipeline buffer
        # x_t_latent: New images entering at stage 0
        # x_t_latent_buffer: Existing images at various intermediate stages
        x_t_latent_batch = torch.cat((x_t_latent, self.x_t_latent_buffer), dim=0)
        print(f"predict_x0_batch: UNet input batch shape: {x_t_latent_batch.shape}")
        
        # Shift the stock noise for RCFG (Residual Classifier-Free Guidance)
        self.stock_noise = torch.cat((self.init_noise[:self.frame_bff_size], self.stock_noise[:-self.frame_bff_size]), dim=0)

        # Execute the UNet step on the entire batch
        # This processes all images at their respective denoising stages simultaneously
        x_0_pred_batch, _ = self.unet_step(x_t_latent_batch, self.sub_timesteps_tensor)
        print(f"predict_x0_batch: UNet output batch shape: {x_0_pred_batch.shape}")

        # The last `frame_bff_size` elements are the fully denoised images (completed pipeline)
        x_0_pred_out = x_0_pred_batch[-self.frame_bff_size:]

        # The first `(denoising_steps_num - 1) * frame_bff_size` elements are the intermediate states
        # These become the new buffer for the next iteration (pipeline advance)
        self.x_t_latent_buffer = x_0_pred_batch[:-self.frame_bff_size]

        print(f"predict_x0_batch: final output shape: {x_0_pred_out.shape}")
        return x_0_pred_out

    @torch.no_grad()
    def __call__(self, x: Union[torch.Tensor, PIL.Image.Image, np.ndarray, List[PIL.Image.Image]]) -> torch.Tensor:
        if isinstance(x, list):
            # Batch of images for stream batching
            x = torch.cat([self.image_processor.preprocess(img, self.height, self.width) for img in x])
        elif not isinstance(x, torch.Tensor):
            x = self.image_processor.preprocess(x, self.height, self.width)
        
        x = x.to(device=self.device, dtype=self.dtype)

        x_t_latent = self.encode_image(x)
        x_0_pred_out = self.predict_x0_batch(x_t_latent)
        x_output = self.decode_image(x_0_pred_out)
        
        return x_output

    @torch.no_grad()
    def txt2img(self, batch_size: int = 1) -> torch.Tensor:
        x_t_latent = torch.randn((batch_size, 4, self.latent_height, self.latent_width), device=self.device, dtype=self.dtype)
        x_0_pred_out = self.predict_x0_batch(x_t_latent)
        return self.decode_image(x_0_pred_out)

    def txt2img_sd_turbo(self, batch_size: int = 1) -> torch.Tensor:
        x_t_latent = torch.randn(
            (batch_size, 4, self.latent_height, self.latent_width),
            device=self.device,
            dtype=self.dtype,
        )
        model_pred = self.unet(
            x_t_latent,
            self.sub_timesteps_tensor,
            encoder_hidden_states=self.prompt_embeds,
            return_dict=False,
        )[0]
        x_0_pred_out = (
            x_t_latent - self.beta_prod_t_sqrt * model_pred
        ) / self.alpha_prod_t_sqrt
        return self.decode_image(x_0_pred_out)