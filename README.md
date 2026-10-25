<h1 align="center">
  🎬 FreeSpec: Training-Free Long Video Generation<br>
  via Singular-Spectrum Reconstruction
</h1>

<p align="center">
  <strong>Fangda Chen</strong> ·
  <strong>Shanshan Zhao</strong> ·
  <strong>Longrong Yang</strong> ·
  <strong>Chuanfu Xu</strong> ·
  <strong>Zhigang Luo</strong> ·
  <strong>Long Lan</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.06509"><img src="https://img.shields.io/badge/arXiv-2605.06509-b31b1b.svg" alt="arXiv:2605.06509"></a>
  <a href="https://fdchen24.github.io/FreeSpec-Website/"><img src="https://img.shields.io/badge/Project_Page-FreeSpec-blue" alt="FreeSpec Project Page"></a>
</p>

## 🎯 Overview

Official implementation of **FreeSpec**, a training-free framework for long video
generation through singular-spectrum reconstruction.
FreeSpec transfers the global attention spectrum into the local attention basis,
modulates it by diffusion timestep and singular-value rank, and adds a lightweight
global residual.

## 📁 Examples

This repository contains two Wan2.1-T2V-1.3B examples:

| Example | Transformer weights | Sampling | Instructions |
| --- | --- | --- | --- |
| Wan2.1 | [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | 50-step UniPC, CFG 6, shift 8 | [Standard example](examples/Wan2.1/README.md) |
| LightX2V | [model_iter6000.pt](https://huggingface.co/lightx2v/Wan2.1-T2V-1.3B-Distill-Models/blob/main/model_iter6000.pt) | 4 distilled steps, no CFG, shift 5 | [Accelerated example](examples/LightX2V/README.md) |

Each example uses its framework's own CUDA environment. There is no shared Conda
environment or additional FreeSpec package to install. The LightX2V example also
uses the original Wan tokenizer, T5 encoder, VAE and model configuration.

## ⚡ Quick Start

```bash
git clone https://github.com/fdchen24/FreeSpec.git
cd FreeSpec
```

Install the dependencies and download the weights using the instructions for
your chosen example, then pass a prompt, frame count and output path.

### Wan2.1

```bash
cd examples/Wan2.1
python generate.py \
  --checkpoint-dir /path/to/Wan2.1-T2V-1.3B \
  --prompt "A sailboat crossing a calm sea at golden hour." \
  --frames 321 \
  --output outputs/sailboat.mp4 \
  --seed 42
```

### LightX2V

From the repository root:

```bash
cd examples/LightX2V
python generate.py \
  --checkpoint-dir /path/to/Wan2.1-T2V-1.3B \
  --distill-checkpoint /path/to/model_iter6000.pt \
  --prompt "A sailboat crossing a calm sea at golden hour." \
  --frames 321 \
  --output outputs/sailboat.mp4 \
  --seed 42
```

Both examples default to 832 × 480, 16 fps, seed 42 and `--local-window 21`.
The local-window default matches the base model's 21 pretraining latent frames.
Frame counts must be `4n+1`; width and height must be divisible by 16.
Each generated video is saved with a neighboring JSON file containing the prompt,
seed and sampling settings. Use `python generate.py --help` for all options.

## 📚 Citation

If you find FreeSpec useful in your research, please cite:

```bibtex
@article{chen2026freespec,
  title={FreeSpec: Training-Free Long Video Generation via Singular-Spectrum Reconstruction},
  author={Fangda Chen and Shanshan Zhao and Longrong Yang and Chuanfu Xu and Zhigang Luo and Long Lan},
  year={2026},
  journal={arXiv preprint arXiv:2605.06509},
}
```

## 🙏 Acknowledgements

This implementation builds on [Wan2.1](https://github.com/Wan-Video/Wan2.1) and
[LightX2V](https://github.com/ModelTC/LightX2V).
