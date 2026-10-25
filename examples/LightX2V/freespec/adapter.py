"""Single-GPU FreeSpec integration using LightX2V's native Wan runtime.

Only self-attention, distilled checkpoint loading and runner construction are
specialized. T5, VAE, weight operators and the four-step scheduler stay upstream.
"""
import torch
from flash_attn import flash_attn_func

from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer
from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
from lightx2v.models.networks.wan.weights.post_weights import WanPostWeights
from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.models.schedulers.wan.step_distill.scheduler import WanStepDistillScheduler
from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE

from .checkpoint import normalize_checkpoint
from .spectral import SPLIT_TIMESTEP, spectral_fusion


class FreeSpecTransformerInfer(WanTransformerInfer):
    def infer_self_attn(self, phase, x, shift_msa, scale_msa, grid_sizes=None):
        normalized = phase.norm1.apply(x).to(self.sensitive_layer_dtype)
        normalized = self.modulate_func(normalized, scale=scale_msa, shift=shift_msa).reshape(x.shape)
        normalized = normalized.to(self.infer_dtype)
        length, heads, dim = normalized.shape[0], self.num_heads, self.head_dim
        q = phase.self_attn_norm_q.apply(phase.self_attn_q.apply(normalized)).view(length, heads, dim)
        k = phase.self_attn_norm_k.apply(phase.self_attn_k.apply(normalized)).view(length, heads, dim)
        v = phase.self_attn_v.apply(normalized).view(length, heads, dim)
        q, k = phase.rope.apply(q, k, self.cos_sin)
        q, k, v = (tensor.unsqueeze(0) for tensor in (q, k, v))
        timestep = float(self.scheduler.timesteps[self.scheduler.step_index].item())
        tokens_per_frame = int(grid_sizes[1] * grid_sizes[2])
        radius = self.config['local_window'] * tokens_per_frame
        global_out = None
        if timestep > SPLIT_TIMESTEP:
            global_out = flash_attn_func(q, k, v).reshape(length, -1)
        local_out = flash_attn_func(q, k, v, window_size=(radius, radius)).reshape(length, -1)
        return phase.self_attn_o.apply(spectral_fusion(global_out, local_out, timestep))


class FreeSpecDistillModel(WanModel):
    post_weight_class = WanPostWeights

    def _init_infer_class(self):
        self.pre_infer_class = WanPreInfer
        self.post_infer_class = WanPostInfer
        self.transformer_infer_class = FreeSpecTransformerInfer

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        # Load the explicitly selected .pt file instead of a convention-based fallback.
        state = normalize_checkpoint(torch.load(
            self.config['dit_original_ckpt'], map_location='cpu', weights_only=True,
        ))
        return {
            key: value.to(device=self.device, dtype=(
                GET_DTYPE() if unified_dtype or all(name not in key for name in sensitive_layer)
                else GET_SENSITIVE_DTYPE()
            ))
            for key, value in state.items()
        }


class FreeSpecRunner(WanRunner):
    def load_transformer(self):
        return FreeSpecDistillModel(
            model_path=self.config['model_path'], config=self.config, device=self.init_device,
        )

    def init_scheduler(self):
        self.scheduler = WanStepDistillScheduler(self.config)
