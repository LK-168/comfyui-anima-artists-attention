"""
LLM-level attention mask for multi-artist isolation.

Patches the Qwen3 self-attention during CLIP encoding so each artist's
tokens cannot see other artists' tokens — eliminating contamination in
the non-linear LLM text space.
"""

import logging
import types
import torch
import torch.nn.functional as F
from contextlib import contextmanager

from ..utils import (
    MAX_ARTISTS,
    split_artist_chain,
    parse_artist_weights,
    get_raw_tokenizer,
    find_llm_transformer,
    build_llm_mask,
)

logger = logging.getLogger(__name__)


class AnimaArtistLLMMask:
    """
    LLM-level attention mask for multi-artist mixing.

    Patches the Qwen3-0.6B self-attention with an isolation mask during
    a single CLIP encoding pass. No LLMAdapter modification needed.

    Workflow:
      1. Build combined prompt: "(artist1:w1), (artist2:w2), base"
      2. Locate each artist's token positions via word_id tokenization
      3. Build a 4D attention mask: artists can't see each other
      4. Patch LLM forward to inject our mask
      5. CLIPTextEncode once → conditioning
      6. Restore LLM
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "artist_chain": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "画师串。逗号或换行分隔，支持权重语法 (name:1.2)"
                }),
                "base_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "主词条。画出什么内容（不含画师名）"
                }),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否启用画师注入"
                }),
                "patch_position": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "画师插入标记（在 base_prompt 中的位置）"
                })
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "positive")
    FUNCTION = "patch"
    CATEGORY = "Anima/CrossAttn"

    @contextmanager
    def _patch_llm_attention(self, transformer, mask):
        """Temporarily patch all self-attention layers to inject artist mask."""
        saved = {}
        for i, layer in enumerate(transformer.layers):
            attn = layer.self_attn
            saved[i] = attn.forward

            orig = attn.forward

            def _mk_patched(o, m):
                def fwd(_self, hidden_states, attention_mask=None, position_ids=None,
                        past_key_value=None, output_attentions=False,
                        use_cache=False, **kwargs):
                    target_dtype = hidden_states.dtype
                    m_dev = m.to(device=hidden_states.device, dtype=target_dtype)
                    if attention_mask is not None:
                        am = attention_mask.to(dtype=target_dtype)
                        if m_dev.shape[-1] < am.shape[-1]:
                            pad_len = am.shape[-1] - m_dev.shape[-1]
                            m_dev = F.pad(m_dev, (0, pad_len, 0, pad_len), value=float("-inf"))
                            m_dev = m_dev.to(dtype=target_dtype)
                        elif m_dev.shape[-1] > am.shape[-1]:
                            m_dev = m_dev[:, :, :am.shape[-2], :am.shape[-1]]
                        merged = am + m_dev
                    else:
                        merged = m_dev
                    return o(hidden_states, attention_mask=merged,
                            past_key_value=past_key_value, **kwargs)
                return fwd

            layer.self_attn.forward = types.MethodType(
                _mk_patched(orig, mask.clone()), layer.self_attn)

        try:
            yield
        finally:
            for i, layer in enumerate(transformer.layers):
                if i in saved:
                    layer.self_attn.forward = saved[i]

    def patch(self, model, clip, artist_chain, base_prompt, enabled, patch_position=""):
        names = split_artist_chain(artist_chain)
        base = (base_prompt or "").strip()

        if not names or not enabled:
            import nodes
            enc = nodes.CLIPTextEncode()
            full_text = f"{', '.join(names)}, {base}" if base else ", ".join(names) if names else base
            return (model, enc.encode(clip, full_text)[0])

        if len(names) > MAX_ARTISTS:
            logger.warning("[LLMMask] 画师数 %d 超过上限 %d，截断", len(names), MAX_ARTISTS)
            names = names[:MAX_ARTISTS]

        artist_names, artist_weights = parse_artist_weights(names)
        n_artists = len(artist_names)

        # All artists use (name:weight) — escape_important handles \( correctly
        weighted_parts = [f"({artist_names[i]}:{artist_weights[i]})" for i in range(n_artists)]

        # Handle patch_position: insert artists at marker location in base
        pos = (patch_position or "").strip()
        if pos and pos in base:
            idx = base.index(pos)
            base_before = base[:idx].strip(", ")
            base_after = base[idx + len(pos):].strip(", ")
            if base_before and base_after:
                full_text = f"{base_before}, {', '.join(weighted_parts)}, {base_after}"
            elif base_before:
                full_text = f"{base_before}, {', '.join(weighted_parts)}"
            else:
                full_text = f"{', '.join(weighted_parts)}, {base_after}"
        else:
            if base:
                full_text = ", ".join(weighted_parts) + f", {base}"
            else:
                full_text = ", ".join(weighted_parts)

        logger.warning("[LLMMask] encoding: %r", full_text)

        # ── Tokenize & locate artist ranges ──
        raw_tok = get_raw_tokenizer(clip)
        full_tok = clip.tokenize(full_text, return_word_ids=True)
        qwen_batch = full_tok["qwen3_06b"][0]

        wid_ranges = {}
        for idx, (_, _, wid) in enumerate(qwen_batch):
            if wid > 0:
                if wid not in wid_ranges:
                    wid_ranges[wid] = [idx, idx]
                wid_ranges[wid][1] = idx

        # Match artists via word_id → decode → name substring
        name_lower_map = {}
        for i in range(n_artists):
            n = artist_names[i].lower()
            name_lower_map[n] = i
            nu = n.replace("\\)", ")").replace("\\(", "(")
            if nu != n:
                name_lower_map[nu] = i

        artist_ranges = []
        for wid in sorted(wid_ranges.keys()):
            sr, er = wid_ranges[wid]
            try:
                decoded = raw_tok.decode(
                    [int(t) for t, _, _ in qwen_batch[sr:er + 1]])
            except Exception:
                decoded = ""
            dlower = decoded.lower().strip()
            for nl, idx in name_lower_map.items():
                if nl in dlower:
                    artist_ranges.append((sr, er + 1, idx))
                    break

        artist_ranges.sort(key=lambda x: x[0])
        artist_ranges_pos = [(s, e) for s, e, _ in artist_ranges]

        all_ids = [int(t) for t, _, _ in qwen_batch]

        # Verify
        for si, ei, idx in artist_ranges:
            name = artist_names[idx]
            try:
                decoded = raw_tok.decode(all_ids[si:ei])
            except Exception:
                decoded = "<err>"
            logger.warning("[LLMMask] artist '%s': [%d:%d] decoded=%r",
                        name, si, ei, decoded)

        logger.warning("[LLMMask] ranges=%s full_len=%d",
                    artist_ranges_pos, len(all_ids))

        # ── Build mask ──
        device = next(model.model.parameters()).device if hasattr(model, 'model') else torch.device("cpu")
        mask = build_llm_mask(all_ids, artist_ranges_pos, device)
        total = mask.shape[-1] * mask.shape[-2]
        blocked = (mask < 0).sum().item()
        logger.warning("[LLMMask] mask %dx%d, blocked %d/%d (%.1f%%)",
                    mask.shape[-1], mask.shape[-2], blocked, total,
                    100.0 * blocked / total if total > 0 else 0)

        # Find LLM transformer and encode with patched attention
        transformer = find_llm_transformer(clip)
        import nodes
        encoder = nodes.CLIPTextEncode()
        with self._patch_llm_attention(transformer, mask):
            positive = encoder.encode(clip, full_text)[0]

        return (model, positive)
