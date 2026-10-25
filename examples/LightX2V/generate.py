"""Generate a FreeSpec video with LightX2V's four-step Wan2.1 1.3B model."""
import argparse
import json
import os
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, required=True, help='Original Wan T5, tokenizer, VAE and config directory.')
    parser.add_argument('--distill-checkpoint', type=Path, required=True, help='Path to model_iter6000.pt.')
    parser.add_argument('--prompt', required=True, help='Text prompt for the generated video.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True, help='Output MP4 path.')
    parser.add_argument('--frames', type=int, required=True, help='Output frame count (4n+1).')
    parser.add_argument('--width', type=int, default=832)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--local-window', type=int, default=21, help='Latent-frame radius; defaults to 21, the base model pretraining latent frame count.')
    args = parser.parse_args(argv)
    if args.frames < 1 or (args.frames - 1) % 4:
        parser.error('--frames must be a positive 4n+1 value, e.g. 81 or 321.')
    if min(args.width, args.height) <= 0 or args.width % 16 or args.height % 16:
        parser.error('--width and --height must be positive multiples of 16.')
    if args.local_window <= 0 or args.seed < 0:
        parser.error('local-window must be positive; seed must be non-negative.')
    if not args.prompt.strip():
        parser.error('The prompt must not be empty.')
    if args.output.suffix.lower() != '.mp4':
        parser.error('--output must end in .mp4.')
    return args


def build_config(args, set_config):
    config = set_config(argparse.Namespace(
        model_cls='wan2.1_distill', task='t2v', support_tasks=[],
        model_path=str(args.checkpoint_dir.resolve()),
        config_json=str(Path(__file__).with_name('config.json')),
        use_prompt_enhancer=False, warmup=False,
    ))
    config.update(
        dit_original_ckpt=str(args.distill_checkpoint.resolve()),
        target_video_length=args.frames, target_height=args.height, target_width=args.width,
        local_window=args.local_window,
    )
    if config.get('dim') != 1536 or config.get('num_layers') != 30:
        raise ValueError('--checkpoint-dir must contain the Wan2.1-T2V-1.3B config.')
    return config


def main():
    args = parse_args()
    for name in ['config.json', 'models_t5_umt5-xxl-enc-bf16.pth', 'Wan2.1_VAE.pth', 'google/umt5-xxl']:
        if not (args.checkpoint_dir / name).exists():
            raise FileNotFoundError(args.checkpoint_dir / name)
    if not args.distill_checkpoint.is_file():
        raise FileNotFoundError(args.distill_checkpoint)
    # Match the original LightX2V sample precision and single-GPU path.
    os.environ['DTYPE'] = 'BF16'
    os.environ['SENSITIVE_LAYER_DTYPE'] = 'None'
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('PROFILING_DEBUG_LEVEL', '0')

    import torch
    from lightx2v.utils.set_config import set_config
    from lightx2v.utils.input_info import T2VInputInfo
    from lightx2v.utils.utils import seed_all
    from freespec.adapter import FreeSpecRunner

    if not torch.cuda.is_available():
        raise RuntimeError('Generation requires an NVIDIA CUDA GPU and FlashAttention 2.')
    torch.set_grad_enabled(False)
    seed_all(args.seed)
    config = build_config(args, set_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runner = FreeSpecRunner(config)
    runner.init_modules()
    runner.run_pipeline(T2VInputInfo(
        seed=args.seed, prompt=args.prompt, save_result_path=str(args.output.resolve()),
    ))
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(backend='LightX2V', method='FreeSpec', steps=4, fps=16,
                    shift=5, enable_cfg=False, split_timestep=900,
                    denoising_step_list=[1000, 750, 500, 250],
                    timesteps=runner.scheduler.timesteps.detach().cpu().tolist())
    args.output.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Saved {args.output} (seed={args.seed})')


if __name__ == '__main__':
    main()
