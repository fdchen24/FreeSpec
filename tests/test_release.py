"""CPU regression checks. Run: python -m unittest discover -s tests -v."""
import ast
import contextlib
import importlib.util
import inspect
import io
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CliTests(unittest.TestCase):
    def test_direct_inputs_defaults_and_overrides_for_both_backends(self):
        for backend in ['Wan2.1', 'LightX2V']:
            cli = load_module('cli', f'examples/{backend}/generate.py')
            base = ['--checkpoint-dir', 'unused', '--prompt', 'A sailboat at sea.',
                    '--frames', '321', '--output', 'outputs/sailboat.mp4']
            if backend == 'LightX2V':
                base += ['--distill-checkpoint', 'unused.pt']
            args = cli.parse_args(base)
            self.assertEqual(args.prompt, 'A sailboat at sea.')
            self.assertEqual(args.frames, 321)
            self.assertEqual(args.output, Path('outputs/sailboat.mp4'))
            self.assertEqual((args.seed, args.local_window), (42, 21))
            self.assertFalse(hasattr(args, 'sample'))
            args = cli.parse_args(base + ['--seed', '123', '--local-window', '7'])
            self.assertEqual((args.seed, args.local_window), (123, 7))

    def test_required_inputs_and_invalid_frames(self):
        for backend in ['Wan2.1', 'LightX2V']:
            cli = load_module('cli', f'examples/{backend}/generate.py')
            base = ['--checkpoint-dir', 'unused']
            if backend == 'LightX2V':
                base += ['--distill-checkpoint', 'unused.pt']
            for flag in ['--prompt', '--frames', '--output']:
                inputs = {'--prompt': 'custom', '--frames': '321', '--output': 'test.mp4'}
                del inputs[flag]
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    cli.parse_args(base + [item for pair in inputs.items() for item in pair])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.parse_args(base + ['--prompt', 'custom', '--frames', '320', '--output', 'test.mp4'])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.parse_args(base + ['--prompt', 'custom', '--frames', '321', '--output', 'test.mp4', '--sample', '10'])


class SpectrumTests(unittest.TestCase):
    def test_reference_v145_outputs(self):
        # Fixture captured from the original v145 implementation before extraction.
        reference = json.loads((ROOT / 'tests/fixtures/v145.json').read_text())
        g, l = torch.tensor(reference['global']), torch.tensor(reference['local'])
        for backend, path in [
            ('Wan2.1', 'examples/Wan2.1/wan/modules/freespec.py'),
            ('LightX2V', 'examples/LightX2V/freespec/spectral.py'),
        ]:
            core = load_module('spectral', path)
            for record in reference['outputs']:
                with self.subTest(backend=backend, timestep=record['timestep'], autocast=record['autocast']):
                    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=record['autocast']):
                        output = core.spectral_fusion(g, l, record['timestep'])
                    torch.testing.assert_close(output, torch.tensor(record['values']), atol=2e-6, rtol=2e-6)
            self.assertIs(core.spectral_fusion(None, l, 899), l)
            zeros = torch.zeros_like(l)
            torch.testing.assert_close(core.spectral_fusion(zeros, zeros, 950), zeros)
            bf16 = core.spectral_fusion(g.bfloat16(), l.bfloat16(), 950)
            self.assertEqual(bf16.dtype, torch.bfloat16)
            self.assertTrue(torch.isfinite(bf16).all())


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = load_module('checkpoint', 'examples/LightX2V/freespec/checkpoint.py')

    def test_wrapped_prefixed_native_weights(self):
        weight = torch.arange(4.)
        result = self.checkpoint.normalize_checkpoint({'model_ema': {
            'module.model.diffusion_model.blocks.0.self_attn.q.weight': weight,
        }})
        self.assertIs(result['blocks.0.self_attn.q.weight'], weight)

    def test_diffusers_names_and_affine_norm(self):
        weight = torch.ones(1)
        source = {
            'condition_embedder.text_embedder.linear_1.weight': weight,
            'blocks.0.attn1.to_q.weight': weight,
            'blocks.0.norm2.weight': weight,
            'blocks.0.scale_shift_table': weight,
            'proj_out.weight': weight,
            'scale_shift_table': weight,
        }
        converted = self.checkpoint.normalize_checkpoint({'state_dict': source})
        self.assertEqual(set(converted), {
            'text_embedding.0.weight', 'blocks.0.self_attn.q.weight',
            'blocks.0.norm3.weight', 'blocks.0.modulation', 'head.head.weight', 'head.modulation',
        })

    def test_conversion_collision_is_rejected(self):
        weight = torch.ones(1)
        source = {
            'condition_embedder.text_embedder.linear_1.weight': weight,
            'blocks.0.attn1.to_q.weight': weight,
            'blocks.0.self_attn.q.weight': weight,
        }
        with self.assertRaises(ValueError):
            self.checkpoint.normalize_checkpoint(source)


class WanIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / 'examples/Wan2.1'))
        from wan.modules import model
        cls.model = model

    def test_pipeline_import_and_api_defaults(self):
        with patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('Unexpected CUDA initialization')):
            from wan.text2video import FreeSpecT2V
        self.assertEqual(inspect.signature(FreeSpecT2V.generate).parameters['local_window'].default, 21)
        self.assertEqual(inspect.signature(self.model.WanModel.forward).parameters['local_window'].default, 21)

    def test_attention_windows_and_timestep_boundary(self):
        attention = self.model.WanSelfAttention(8, 1)
        inputs = torch.randn(1, 12, 8)
        freqs = torch.cat([self.model.rope_params(16, 4), self.model.rope_params(16, 2),
                           self.model.rope_params(16, 2)], dim=1)
        calls = []

        def fake_attention(q, k, v, **kwargs):
            calls.append(kwargs.get('window_size', (-1, -1)))
            return v.clone()

        for timestep, expected in [(1000, [(-1, -1), (8, 8)]), (900, [(8, 8)]), (899, [(8, 8)])]:
            calls.clear()
            with patch.object(self.model, 'flash_attention', fake_attention):
                output = attention(inputs, torch.tensor([12]), torch.tensor([[3, 2, 2]]),
                                   freqs, timestep, local_window=2)
            self.assertEqual(calls, expected)
            self.assertEqual(output.shape, inputs.shape)

    def test_small_backbone_forward_and_checkpoint_reload(self):
        model = self.model.WanModel(
            patch_size=(1, 2, 2), in_dim=2, out_dim=2, dim=16, ffn_dim=32,
            num_heads=2, num_layers=1, freq_dim=8, text_dim=8, text_len=5,
        ).eval()

        def cpu_attention(q, k, v, k_lens=None, window_size=(-1, -1)):
            length = q.shape[1]
            mask = None
            if window_size != (-1, -1):
                row = torch.arange(length).unsqueeze(1)
                col = torch.arange(k.shape[1]).unsqueeze(0)
                mask = (col >= row - window_size[0]) & (col <= row + window_size[1])
            return torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask,
            ).transpose(1, 2).contiguous()

        model.load_state_dict(model.state_dict(), strict=True)
        latent = torch.randn(2, 3, 4, 4)
        context = [torch.randn(4, 8)]
        with torch.no_grad(), patch.object(self.model, 'flash_attention', cpu_attention):
            for t in [1000., 900., 500.]:
                result = model([latent], torch.tensor([t]), context, 12, local_window=1)
                self.assertEqual(result[0].shape, latent.shape)
                self.assertTrue(torch.isfinite(result[0]).all())


class LightIntegrationTests(unittest.TestCase):
    def test_adapter_window_calls_with_native_phase_interface(self):
        # Isolate the adapter class from the CUDA-only upstream imports. This
        # exercises its actual method with the native weight-phase interface.
        source = ast.parse((ROOT / 'examples/LightX2V/freespec/adapter.py').read_text())
        node = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'FreeSpecTransformerInfer')
        calls = []

        def attention(q, k, v, window_size=(-1, -1)):
            calls.append(window_size)
            return v.clone()

        spectral = load_module('spectral', 'examples/LightX2V/freespec/spectral.py')
        namespace = dict(torch=torch, WanTransformerInfer=object, flash_attn_func=attention,
                         SPLIT_TIMESTEP=900, spectral_fusion=spectral.spectral_fusion)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<adapter>', 'exec'), namespace)
        adapter = namespace['FreeSpecTransformerInfer']()
        adapter.num_heads, adapter.head_dim = 2, 4
        adapter.infer_dtype = adapter.sensitive_layer_dtype = torch.float32
        adapter.config = {'local_window': 2}
        adapter.cos_sin = None
        adapter.modulate_func = lambda x, scale, shift: x * (1 + scale.squeeze()) + shift.squeeze()
        operator = types.SimpleNamespace(apply=lambda x: x)
        phase = types.SimpleNamespace(**{name: operator for name in [
            'norm1', 'self_attn_q', 'self_attn_k', 'self_attn_v',
            'self_attn_norm_q', 'self_attn_norm_k', 'self_attn_o',
        ]})
        phase.rope = types.SimpleNamespace(apply=lambda q, k, _: (q, k))
        # The shifted four-step schedule crosses 900 between the second and third steps.
        timesteps = torch.tensor([1000., 937.5, 833.3333, 625.])
        adapter.scheduler = types.SimpleNamespace(timesteps=timesteps, step_index=0)
        x = torch.randn(12, 8)
        for index in range(4):
            calls.clear()
            adapter.scheduler.step_index = index
            output = adapter.infer_self_attn(phase, x, torch.zeros(1, 8), torch.zeros(1, 8), [3, 2, 2])
            self.assertEqual(calls, [(-1, -1), (8, 8)] if index < 2 else [(8, 8)])
            self.assertEqual(output.shape, x.shape)


if __name__ == '__main__':
    unittest.main()
