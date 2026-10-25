# FreeSpec with LightX2V's distilled Wan2.1 1.3B

This example extracts the FreeSpec path and checkpoint adapter from the local
LightX2V implementation. It uses the native LightX2V runtime for model weights,
T5, VAE and the four-step scheduler. The released integration lives entirely in
`freespec/`; it does not modify the installed framework.

## Environment

Reuse a working LightX2V CUDA environment with FlashAttention 2. No additional
Conda environment or FreeSpec installation is required. The adapter targets
[LightX2V revision d658c11](https://github.com/ModelTC/LightX2V/tree/d658c11edb77124387f275c1c080fcf9612ba8e1),
the revision of the extraction source. Install that revision in a fresh native
LightX2V environment if needed:

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

See the [native installation instructions](https://github.com/ModelTC/LightX2V/tree/d658c11edb77124387f275c1c080fcf9612ba8e1#installation-from-git)
for the CUDA/PyTorch environment. This example uses a single NVIDIA GPU and
unquantized BF16 transformer weights. The original accelerated sample keeps the
model on GPU; multi-GPU, quantization, LoRA, feature caching and CPU-offload modes
are outside this example's scope.

## Weights

Use two downloads: the base Wan files for the text encoder/VAE/configuration,
and the distilled transformer checkpoint.

```bash
pip install "huggingface-hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir models/Wan2.1-T2V-1.3B
huggingface-cli download lightx2v/Wan2.1-T2V-1.3B-Distill-Models \
  model_iter6000.pt --local-dir models/distill
```

The base transformer is not used by this example. An existing native Wan
checkpoint directory can be reused. The adapter loads the explicitly supplied
`model_iter6000.pt`, unwraps training-state containers, and maps Diffusers Wan
tensor names to the native Wan names when needed.

## Generate

Run from this example directory:

```bash
python generate.py --checkpoint-dir models/Wan2.1-T2V-1.3B \
  --distill-checkpoint models/distill/model_iter6000.pt \
  --prompt "A sailboat crossing a calm sea at golden hour." \
  --seed 42 --frames 321 --output outputs/sailboat.mp4
```

Pass `--prompt`, `--frames` and `--output` for each run.
Defaults: 832 × 480, 16 fps, seed 42, four steps, shift 5, CFG disabled,
local-window radius 21, matching the base model's pretraining latent frame count.

The scheduler's input step list is `[1000, 750, 500, 250]`. With shift 5, its
actual model timesteps are approximately `[1000, 937.5, 833.33, 625]`. FreeSpec
therefore evaluates both attention branches on the first two steps and local
attention on the last two. The actual timesteps are saved in the output JSON.
