"""Checkpoint-format adapters extracted from the local LightX2V implementation."""


def normalize_checkpoint(state_dict):
    """Unwrap the native or Diffusers training checkpoint before dtype conversion."""
    import torch

    containers = ('state_dict', 'model_ema', 'ema', 'model', 'module')
    while isinstance(state_dict, dict):
        nested = next((key for key in containers if isinstance(state_dict.get(key), dict)), None)
        if nested is None:
            break
        state_dict = state_dict[nested]
    if not isinstance(state_dict, dict):
        raise TypeError('Expected a state dictionary in the distilled checkpoint.')
    state_dict = {key: value for key, value in state_dict.items()
                  if isinstance(key, str) and isinstance(value, torch.Tensor)}
    if not state_dict:
        raise ValueError('The distilled checkpoint contains no tensors.')
    for prefix in ('module.model.diffusion_model.', 'model.diffusion_model.', 'diffusion_model.', 'module.'):
        if all(key.startswith(prefix) for key in state_dict):
            state_dict = {key[len(prefix):]: value for key, value in state_dict.items()}
            break
    return convert_diffusers_wan_state_dict(state_dict)


_DIFFUSERS_TOP_LEVEL_RENAMES = {
    "condition_embedder.text_embedder.linear_1.": "text_embedding.0.",
    "condition_embedder.text_embedder.linear_2.": "text_embedding.2.",
    "condition_embedder.time_embedder.linear_1.": "time_embedding.0.",
    "condition_embedder.time_embedder.linear_2.": "time_embedding.2.",
    "condition_embedder.time_proj.": "time_projection.1.",
    "proj_out.": "head.head.",
}

_DIFFUSERS_BLOCK_RENAMES = (
    (".attn1.to_q.", ".self_attn.q."),
    (".attn1.to_k.", ".self_attn.k."),
    (".attn1.to_v.", ".self_attn.v."),
    (".attn1.to_out.0.", ".self_attn.o."),
    (".attn1.norm_q.", ".self_attn.norm_q."),
    (".attn1.norm_k.", ".self_attn.norm_k."),
    (".attn2.to_q.", ".cross_attn.q."),
    (".attn2.to_k.", ".cross_attn.k."),
    (".attn2.to_v.", ".cross_attn.v."),
    (".attn2.to_out.0.", ".cross_attn.o."),
    (".attn2.norm_q.", ".cross_attn.norm_q."),
    (".attn2.norm_k.", ".cross_attn.norm_k."),
    (".ffn.net.0.proj.", ".ffn.0."),
    (".ffn.net.2.", ".ffn.2."),
    (".scale_shift_table", ".modulation"),
    (".norm2.", ".norm3."),
)


def is_diffusers_wan_state_dict(state_dict):
    """Return whether a state dict uses Diffusers WanTransformer3DModel names."""
    return (
        "condition_embedder.text_embedder.linear_1.weight" in state_dict
        and "blocks.0.attn1.to_q.weight" in state_dict
    )


def convert_diffusers_wan_state_dict(state_dict):
    """Convert Diffusers WanTransformer3DModel names to native Wan names."""
    if not is_diffusers_wan_state_dict(state_dict):
        return state_dict

    converted = {}
    for key, value in state_dict.items():
        new_key = key
        if key == "scale_shift_table":
            new_key = "head.modulation"
        else:
            for source, target in _DIFFUSERS_TOP_LEVEL_RENAMES.items():
                if key.startswith(source):
                    new_key = target + key[len(source) :]
                    break
            if key.startswith("blocks."):
                for source, target in _DIFFUSERS_BLOCK_RENAMES:
                    if source in new_key:
                        new_key = new_key.replace(source, target, 1)
                        break

        if new_key in converted:
            raise ValueError(f"Wan checkpoint key conversion collision: {key!r} -> {new_key!r}")
        converted[new_key] = value

    return converted
