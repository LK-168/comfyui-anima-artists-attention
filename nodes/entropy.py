"""
Cross-Attention Entropy 分析节点。

Quantifies how well the DiT "recognizes" each artist by measuring
the entropy of cross-attention maps over artist token positions.

- Low entropy → attention is spatially focused → artist features extracted
- High entropy → attention is uniformly dispersed → artist signal blurred
"""

import logging
import types
import torch
import torch.nn.functional as F
from contextlib import contextmanager

from ..utils import (
    split_artist_chain,
    parse_artist_weights,
    get_raw_tokenizer,
    encode_prompt,
    locate_artist_ranges,
)

logger = logging.getLogger(__name__)


# ── Entropy computation ─────────────────────────────


# ── Hook context manager ────────────────────────────
@contextmanager
def _patch_cross_attn_for_entropy(diffusion_model, artist_ranges, n_artists):
    """
    Monkey-patch Block.cross_attn.compute_attention.
    Accumulates entropy into a GPU tensor [n_layers, n_artists],
    only transferring to CPU once at context exit.
    """
    blocks = diffusion_model.blocks
    n_layers = len(blocks)
    device = next(diffusion_model.parameters()).device
    saved = {}
    accum = torch.zeros(n_layers, n_artists, device=device, dtype=torch.float32)

    for i, block in enumerate(blocks):
        attn = block.cross_attn
        orig = attn.compute_attention
        hd = attn.head_dim
        nh = attn.n_heads
        layer_idx = i

        def _make_patched(original_fn, accum_tensor, head_dim, n_heads, ranges, lidx):
            def patched(self, q, k, v, transformer_options=None):
                if transformer_options is None:
                    transformer_options = {}
                if lidx == 0:
                    logger.warning("[Entropy] q device=%s dtype=%s k device=%s dtype=%s",
                                q.device, q.dtype, k.device, k.dtype)
                scale = head_dim ** -0.5
                layer_ent = torch.zeros(len(ranges), device=accum_tensor.device,
                                         dtype=torch.float32)
                for h in range(n_heads):
                    # matmul stays bfloat16 (fast), cast to float32 for stable log
                    q_h = q[..., h, :].reshape(-1, head_dim)
                    k_h = k[..., h, :].reshape(-1, head_dim)
                    attn_w = F.softmax(torch.matmul(q_h, k_h.T) * scale, dim=-1).float()
                    for ri, (sr, er, art_idx) in enumerate(ranges):
                        if sr >= attn_w.shape[-1]:
                            continue
                        e = min(er, attn_w.shape[-1])
                        artist_attn = attn_w[:, sr:e]
                        ent = -(artist_attn * artist_attn.clamp(min=1e-8).log()).sum(dim=1).mean()
                        layer_ent[ri] += ent * (1.0 / n_heads)
                        ent = -(artist_attn * artist_attn.clamp(min=1e-8).log()).sum(dim=1).mean()
                        layer_ent[ri] += ent * (1.0 / n_heads)
                accum_tensor[lidx] = layer_ent
                return original_fn(q, k, v, transformer_options=transformer_options)
            return patched

        attn.compute_attention = types.MethodType(
            _make_patched(orig, accum, hd, nh, artist_ranges, layer_idx), attn)
        saved[i] = orig

    try:
        yield accum
    finally:
        for i, block in enumerate(blocks):
            if i in saved:
                block.cross_attn.compute_attention = saved[i]


class AnimaCrossAttnEntropy:
    """
    Cross-Attention Entropy Analysis Node.

    Runs one forward pass of the DiT with a random latent at a given
    timestep, captures cross-attention weights, and computes per-artist
    entropy values layer by layer.

    Usage: compare single-artist vs multi-artist entropy — a sharp
    entropy increase under mixing quantitatively proves contamination.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "artist_chain": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "画师串，逗号/换行分隔，支持 (name:weight)"
                }),
                "base_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "主词条（不含画师名）"
                }),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "timestep": ("FLOAT", {
                    "default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "去噪时间步。0.5=中间步"
                }),
            },
        }

    RETURN_TYPES = ("ENTROPY_DATA", "STRING")
    RETURN_NAMES = ("entropy_data", "summary")
    FUNCTION = "analyze"
    CATEGORY = "Anima/Analysis"

    def analyze(self, model, clip, artist_chain, base_prompt, width, height, timestep):
        names = split_artist_chain(artist_chain)
        base = (base_prompt or "").strip()

        if not names:
            return ({}, "No artists provided.")

        artist_names, artist_weights = parse_artist_weights(names)
        n_artists = len(artist_names)

        # Build full prompt
        weighted_parts = [f"({artist_names[i]}:{artist_weights[i]})"
                          for i in range(n_artists)]
        full_text = ", ".join(weighted_parts)
        if base:
            full_text = f"{full_text}, {base}"

        logger.warning("[Entropy] encoding: %r", full_text)
        positive = encode_prompt(clip, full_text)

        # Locate artist token ranges
        artist_ranges = locate_artist_ranges(clip, artist_names, full_text)
        artist_ranges.sort(key=lambda x: x[0])

        if not artist_ranges:
            return ({}, "Failed to locate artist tokens in encoded prompt.")

        # Prepare latent — MiniTrainDIT expects [B, C, T, H, W] (5D video format)
        latent_h = height // 8
        latent_w = width // 8
        x = torch.randn(1, 16, 1, latent_h, latent_w)  # T=1 for image

        base_model = model.model
        diffusion_model = base_model.diffusion_model
        inference_dtype = base_model.get_dtype_inference()

        x = x.to(dtype=inference_dtype)

        # Unpack conditioning
        cond_data = positive[0]
        if isinstance(cond_data, (list, tuple)):
            context_tensor = cond_data[0]
            cond_dict = cond_data[1] if len(cond_data) >= 2 else {}
        else:
            context_tensor = cond_data
            cond_dict = {}

        context_tensor = context_tensor.to(dtype=inference_dtype)

        extra_kwargs = {}
        for k in ("t5xxl_ids", "t5xxl_weights"):
            if k in cond_dict and cond_dict[k] is not None:
                val = cond_dict[k]
                # t5xxl_ids must stay integer (embedding indices), not be cast
                if val.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                    pass  # integer — keep as-is
                else:
                    val = val.to(dtype=inference_dtype)
                # Ensure batch dim: LLMAdapter expects [B, seq]
                if val.dim() == 1:
                    val = val.unsqueeze(0)
                # t5xxl_weights needs channel dim: [B, seq] → [B, seq, 1]
                if k == "t5xxl_weights" and val.dim() == 2:
                    val = val.unsqueeze(-1)
                extra_kwargs[k] = val

        t_tensor = torch.tensor([float(timestep)])

        # Force model to GPU — ModelPatcher may have offloaded after encoding
        import comfy.model_management
        load_device = comfy.model_management.get_torch_device()
        if hasattr(model, 'model') and hasattr(model, 'load_model'):
            model.load_model()
        diffusion_model.to(load_device)

        x = x.to(load_device)
        context_tensor = context_tensor.to(load_device)
        t_tensor = t_tensor.to(load_device)
        for k in extra_kwargs:
            extra_kwargs[k] = extra_kwargs[k].to(load_device)

        # Hook + forward
        with _patch_cross_attn_for_entropy(diffusion_model, artist_ranges, n_artists) as accum:
            with torch.no_grad():
                _ = diffusion_model(x, t_tensor, context=context_tensor,
                                    **extra_kwargs)

        # Single GPU→CPU transfer — all layers at once
        entropy_gpu = accum.cpu()

        # Build result dict
        result = {}
        n_layers = entropy_gpu.shape[0]
        for lidx in range(n_layers):
            layer_vals = {}
            for ri, (_, _, art_idx) in enumerate(artist_ranges):
                layer_vals[art_idx] = float(entropy_gpu[lidx, ri])
            if layer_vals:
                result[lidx] = layer_vals

        # Build summary
        summary_lines = ["=== Cross-Attn Entropy Analysis ==="]
        summary_lines.append(f"Artists: {artist_names}")
        summary_lines.append(f"Timestep: {timestep:.2f}")
        summary_lines.append(f"Layers: {len(result)}")
        summary_lines.append("")

        for layer_idx in sorted(result):
            layer_entropy = result[layer_idx]
            for art_idx in sorted(layer_entropy):
                if art_idx < len(artist_names):
                    summary_lines.append(
                        f"  L{layer_idx:02d} | {artist_names[art_idx]:30s} | H={layer_entropy[art_idx]:.4f}")

        # Per-artist averages
        import math
        h_max = math.log(latent_h * latent_w)  # uniform distribution entropy
        summary_lines.append("")
        summary_lines.append(f"── Per-artist average (H_max = ln({latent_h}×{latent_w}) = {h_max:.1f}) ──")
        for art_idx in range(n_artists):
            vals = [result[l][art_idx] for l in result if art_idx in result[l]]
            avg = sum(vals) / max(1, len(vals)) if vals else 0.0
            if avg < h_max * 0.15:
                label = "← FOCUSED"
            elif avg < h_max * 0.4:
                label = "← MODERATE"
            else:
                label = "← DISPERSED"
            summary_lines.append(
                f"  {artist_names[art_idx]:30s} | avg_H={avg:.4f}  {label}")

        return (result, "\n".join(summary_lines))
