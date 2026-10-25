"""Generate a long video with FreeSpec and the original Wan2.1 1.3B weights."""
import argparse
import json
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--prompt', required=True, help='Text prompt for the generated video.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True, help='Output MP4 path.')
    parser.add_argument('--frames', type=int, required=True, help='Output frame count (4n+1).')
    parser.add_argument('--width', type=int, default=832)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--shift', type=float, default=8.0)
    parser.add_argument('--guidance', type=float, default=6.0)
    parser.add_argument('--local-window', type=int, default=21, help='Latent-frame radius; defaults to 21, the base model pretraining latent frame count.')
    parser.add_argument('--t5-on-gpu', action='store_true', help='Run the text encoder on GPU instead of CPU.')
    parser.add_argument('--keep-model-on-gpu', action='store_true', help='Keep the transformer on GPU during VAE decode.')
    args = parser.parse_args(argv)
    if args.frames < 1 or (args.frames - 1) % 4:
        parser.error('--frames must be a positive 4n+1 value, e.g. 81 or 321.')
    if min(args.width, args.height) <= 0 or args.width % 16 or args.height % 16:
        parser.error('--width and --height must be positive multiples of 16.')
    if args.steps <= 0 or args.shift <= 0 or args.local_window <= 0 or args.seed < 0:
        parser.error('steps, shift and local-window must be positive; seed must be non-negative.')
    if not args.prompt.strip():
        parser.error('The prompt must not be empty.')
    if args.output.suffix.lower() != '.mp4':
        parser.error('--output must end in .mp4.')
    return args


def main():
    args = parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('Generation requires an NVIDIA CUDA GPU and FlashAttention 2.')
    from wan.text2video import FreeSpecT2V
    from wan.configs import t2v_1_3B
    import imageio.v2 as imageio

    for name in ['config.json', t2v_1_3B.t5_checkpoint, t2v_1_3B.vae_checkpoint, t2v_1_3B.t5_tokenizer]:
        if not (args.checkpoint_dir / name).exists():
            raise FileNotFoundError(args.checkpoint_dir / name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pipe = FreeSpecT2V(args.checkpoint_dir, t5_cpu=not args.t5_on_gpu, offload=not args.keep_model_on_gpu)
    video = pipe.generate(
        args.prompt, seed=args.seed, width=args.width, height=args.height,
        frames=args.frames, steps=args.steps, shift=args.shift,
        guidance=args.guidance, local_window=args.local_window,
    )
    # Convert frame by frame to avoid allocating a second full float video.
    with imageio.get_writer(str(args.output), fps=16, codec='libx264', quality=8, macro_block_size=None) as writer:
        for frame in video.permute(1, 2, 3, 0):
            writer.append_data(((frame.float().clamp(-1, 1) + 1) * 127.5).round().byte().numpy())
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(backend='Wan2.1', method='FreeSpec', fps=16,
                    split_timestep=900, negative_prompt=t2v_1_3B.sample_neg_prompt)
    args.output.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Saved {args.output} (seed={args.seed})')


if __name__ == '__main__':
    main()
