"""FreeSpec singular-spectrum reconstruction, extracted from v145."""
import math
import torch

SPLIT_TIMESTEP = 900.0


def spectral_fusion(global_out, local_out, timestep):
    """Reconstruct [tokens, channels] in the local singular basis.

    The global branch is evaluated only for t > 900. The remaining steps use
    local attention directly. SVD uses FP32; reconstruction follows runtime autocast.
    """
    if global_out is None:
        return local_out
    if global_out.shape != local_out.shape or local_out.ndim != 2:
        raise ValueError('Expected matching [tokens, channels] attention outputs.')
    t = max(0.0, min(1000.0, float(timestep)))
    progress = max(0.0, min(1.0, (1000.0 - t) / (1000.0 - SPLIT_TIMESTEP)))
    local_weight = (1.0 - math.exp(-5.0 * progress)) / (1.0 - math.exp(-5.0))
    global_weight = 1.0 - local_weight
    g, l = global_out.float(), local_out.float()
    _, sg, _ = torch.linalg.svd(g, full_matrices=False)
    ul, sl, vhl = torch.linalg.svd(l, full_matrices=False)
    ranks = torch.linspace(0.0, 1.0, sg.numel(), device=sg.device, dtype=sg.dtype)
    coefficients = global_weight * torch.exp(-5.0 * ranks)
    spectrum = coefficients * sg + (1.0 - coefficients) * sl
    reconstructed = ul @ torch.diag(spectrum) @ vhl
    anchor = 0.15 + 0.20 * global_weight
    result = (1.0 - anchor) * reconstructed + anchor * g
    return result.to(local_out.dtype)
