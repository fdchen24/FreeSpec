# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified for FreeSpec: single-GPU Wan2.1-T2V-1.3B inference.
import gc
from pathlib import Path

import torch
from tqdm import tqdm

from .configs import t2v_1_3B
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class FreeSpecT2V:
    def __init__(self, checkpoint_dir, *, t5_cpu=True, offload=True):
        checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device('cuda')
        self.t5_cpu = t5_cpu
        self.offload = offload
        self.config = t2v_1_3B
        self.text_encoder = T5EncoderModel(
            text_len=self.config.text_len,
            dtype=self.config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=str(checkpoint_dir / self.config.t5_checkpoint),
            tokenizer_path=str(checkpoint_dir / self.config.t5_tokenizer),
        )
        self.vae = WanVAE(vae_pth=str(checkpoint_dir / self.config.vae_checkpoint), device=self.device)
        self.model = WanModel.from_pretrained(str(checkpoint_dir)).eval().requires_grad_(False)

    @torch.no_grad()
    def generate(self, prompt, *, seed=42, width=832, height=480, frames=321,
                 steps=50, shift=8.0, guidance=6.0, local_window=21, negative_prompt=None):
        if frames < 1 or (frames - 1) % 4:
            raise ValueError('frames must be a positive 4n+1 value.')
        if width <= 0 or height <= 0 or width % 16 or height % 16:
            raise ValueError('width and height must be positive multiples of 16.')
        if steps <= 0 or shift <= 0 or local_window <= 0 or seed < 0:
            raise ValueError('steps, shift and local_window must be positive; seed must be non-negative.')
        text_device = torch.device('cpu') if self.t5_cpu else self.device
        self.text_encoder.model.to(text_device)
        negative_prompt = self.config.sample_neg_prompt if negative_prompt is None else negative_prompt
        context = [x.to(self.device) for x in self.text_encoder([prompt], text_device)]
        context_null = [x.to(self.device) for x in self.text_encoder([negative_prompt], text_device)]
        if self.offload:
            self.text_encoder.model.cpu()

        shape = (16, (frames - 1) // 4 + 1, height // 8, width // 8)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = [torch.randn(shape, dtype=torch.float32, device=self.device, generator=generator)]
        seq_len = shape[1] * (shape[2] // 2) * (shape[3] // 2)
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000, shift=1, use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(steps, device=self.device, shift=shift)
        self.model.to(self.device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for t in tqdm(scheduler.timesteps, desc='FreeSpec'):
                timestep = torch.stack([t])
                conditional = self.model(latents, timestep, context, seq_len, local_window)[0]
                unconditional = self.model(latents, timestep, context_null, seq_len, local_window)[0]
                flow = unconditional + guidance * (conditional - unconditional)
                latents = [scheduler.step(
                    flow.unsqueeze(0), t, latents[0].unsqueeze(0),
                    return_dict=False, generator=generator,
                )[0].squeeze(0)]
        del conditional, unconditional, flow, context, context_null
        if self.offload:
            self.model.cpu()
            gc.collect()
            torch.cuda.empty_cache()
        # Keep the causal decoder cache; move decoded chunks to CPU immediately.
        return self.vae.decode(latents, temporal_chunk_size=1, offload_to_cpu=True)[0]
