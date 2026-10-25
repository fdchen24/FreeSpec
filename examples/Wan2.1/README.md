# FreeSpec with Wan2.1-T2V-1.3B

This example contains the T2V backbone, T5 tokenizer/encoder, causal VAE and UniPC
scheduler needed by the original 1.3B model. The FreeSpec attention path is
extracted from `LongWan2.1`'s `v145` implementation.

## Environment

Use your existing Wan2.1 environment, or create one with Python 3.10 or newer
(the source environment used Python 3.12). Install a CUDA build of PyTorch 2.4+
compatible with your GPU before installing the remaining dependencies. Linux or
WSL2 with an NVIDIA GPU is the intended runtime.

```bash
conda create -n freespec-wan python=3.12 -y
conda activate freespec-wan
# Install the CUDA PyTorch build appropriate for your machine first.
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

The requirements retain the source Diffusers/Transformers versions and omit the
research environment's unrelated web, plotting and evaluation packages.

## Weights

Download the native [Wan2.1-T2V-1.3B checkpoint](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B):

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir models/Wan2.1-T2V-1.3B
```

Pass the directory containing `config.json`, `diffusion_pytorch_model.safetensors`,
`models_t5_umt5-xxl-enc-bf16.pth`, `Wan2.1_VAE.pth` and `google/umt5-xxl/`.

## Generate

Run from this example directory:

```bash
python generate.py --checkpoint-dir models/Wan2.1-T2V-1.3B \
  --prompt "A sailboat crossing a calm sea at golden hour." \
  --seed 42 --frames 321 --output outputs/sailboat.mp4
```

Pass `--prompt`, `--frames` and `--output` for each run.
Defaults: 832 × 480, 16 fps, seed 42, 50 UniPC steps, shift 8, CFG 6,
local-window radius 21, matching the base model's pretraining latent frame count.
The original English negative prompt is retained and
recorded in each output JSON. Frame counts must be `4n+1`; width and height must
be divisible by 16.

T5 runs on CPU by default, the transformer is offloaded before decoding, and
decoded VAE chunks are immediately moved to CPU. Use `--t5-on-gpu` and
`--keep-model-on-gpu` when memory permits. This release supports one GPU; its
321-frame memory demand should not be inferred from the original 81-frame model
card. Lower `--frames` or resolution if necessary.

FreeSpec's global/local computation is in [wan/modules/model.py](wan/modules/model.py),
and the fixed spectral reconstruction is in [wan/modules/freespec.py](wan/modules/freespec.py).
The public entry point has no legacy FreeLong version or ablation flags.
