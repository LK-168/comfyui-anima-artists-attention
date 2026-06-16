"""
LLM-level attention mask for multi-artist isolation.

Unlike nodes.py (LLMAdapter mask), this approach patches the LLM's own
self-attention during CLIP encoding, so the entire pipeline remains
unchanged — normal sequence length, normal LLMAdapter, normal DiT.
"""

import logging
import torch
import torch.nn.functional as F
import types
from contextlib import contextmanager

logger = logging.getLogger(__name__)

MAX_ARTISTS = 32


def _split_artist_chain(chain):
    if not chain:
        return []
    s = str(chain).replace("，", ",").replace("\n", ",").replace("\r", ",")
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def _parse_artist_weights(names):
    """Parse artist names, handling both 'name' and '(name:weight)' formats."""
    parsed, weights = [], []
    for name in names:
        if ":" in name and name.rfind(":") > 0:
            parts = name.rsplit(":", 1)
            try:
                # Strip parens from (name:weight) syntax
                w = float(parts[1].strip().rstrip(")"))
                n = parts[0].strip().lstrip("(")
                parsed.append(n)
                weights.append(w)
            except ValueError:
                parsed.append(name)
                weights.append(1.0)
        else:
            parsed.append(name)
            weights.append(1.0)
    return parsed, weights


def _find_llm_transformer(clip):
    """Navigate CLIP object to find the Qwen3-0.6B transformer."""
    model = clip.cond_stage_model if hasattr(clip, "cond_stage_model") else clip

    # Try common paths
    for path in [
        "transformer", "model", "llm", "text_model",
        "transformer.text_model", "model.model",
    ]:
        obj = model
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                obj = None
                break
        if obj is not None and hasattr(obj, "layers"):
            return obj

    # Deep search: look for any attribute with 'layers'
    def _search(obj, depth=0):
        if depth > 3:
            return None
        if hasattr(obj, "layers") and hasattr(obj.layers, "__len__") and len(obj.layers) > 0:
            return obj
        for attr in dir(obj):
            if attr.startswith("_"):
                continue
            try:
                child = getattr(obj, attr)
            except Exception:
                continue
            if isinstance(child, torch.nn.Module):
                result = _search(child, depth + 1)
                if result is not None:
                    return result
        return None

    result = _search(model)
    if result is not None:
        logger.warning("[LLMMask] found transformer: %s", type(result).__name__)
        return result
    raise RuntimeError("[LLMMask] 无法找到 Qwen3 LLM transformer")


def _get_raw_tokenizer(clip):
    """Get the raw HuggingFace tokenizer from any ComfyUI CLIP model.

    Auto-detects model type (Anima, SD1, SDXL, etc.) and returns
    the underlying HuggingFace tokenizer for decode/verification.
    """
    tokenizer_obj = clip.tokenizer

    # Anima: AnimaTokenizer wraps qwen3_06b + t5xxl sub-tokenizers
    te_name = getattr(clip.cond_stage_model, 'clip_name', '') if hasattr(clip, 'cond_stage_model') else ''
    if te_name == 'qwen3_06b' and hasattr(tokenizer_obj, 'qwen3_06b'):
        sub = tokenizer_obj.qwen3_06b
        if hasattr(sub, 'tokenizer'):
            return sub.tokenizer

    # SD1/SDXL: SDTokenizer has .tokenizer directly
    if hasattr(tokenizer_obj, 'tokenizer'):
        return tokenizer_obj.tokenizer

    # Fallback: search tokenizer object for any HuggingFace tokenizer
    import transformers
    for attr in dir(tokenizer_obj):
        if attr.startswith('_'):
            continue
        try:
            obj = getattr(tokenizer_obj, attr)
            if isinstance(obj, transformers.PreTrainedTokenizerBase):
                return obj
        except Exception:
            continue

    raise RuntimeError("[LLMMask] 无法自动检测 tokenizer，模型类型: %s", type(tokenizer_obj).__name__)


def _build_llm_mask(full_ids, artist_ranges, device):
    """Build 4D additive mask [1,1,seq,seq] with artist isolation only.

    The model provides its own causal mask. We only add -inf blocks
    for artist-to-artist positions. All other positions are 0.0,
    preserving the model's original attention pattern.
    """
    seq_len = len(full_ids)
    # Start with all zeros — no causal mask, model handles that
    mask = torch.zeros(1, 1, seq_len, seq_len, device=device, dtype=torch.float)

    # Block artist-to-artist cross-attention
    for i, (si, ei) in enumerate(artist_ranges):
        for j, (sj, ej) in enumerate(artist_ranges):
            if i != j:
                mask[:, :, si:ei, sj:ej] = float("-inf")

    return mask


class AnimaArtistLLMMask:
    """
    LLM-level attention mask for multi-artist mixing.

    Patches the Qwen3-0.6B self-attention with an isolation mask during
    a single CLIP encoding pass. No LLMAdapter modification needed.

    Workflow:
      1. Build combined prompt: "artist1, artist2, base"
      2. Locate each artist's token positions in the tokenized prompt
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
                    "tooltip": "画师串。逗号或换行分隔。"
                }),
                "base_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "主词条。"
                }),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否启用画师注入"
                }),
                "patch_position": ("STRING", {
                    "multiline": False,
                    "default": ""
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
                    # Merge artist mask with whatever mask the model provides
                    m_dev = m.to(device=hidden_states.device, dtype=torch.float)
                    if attention_mask is not None:
                        am = attention_mask.to(dtype=torch.float)
                        if m_dev.shape[-1] < am.shape[-1]:
                            pad_len = am.shape[-1] - m_dev.shape[-1]
                            m_dev = F.pad(m_dev, (0, pad_len, 0, pad_len), value=float("-inf"))
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
        names = _split_artist_chain(artist_chain)
        base = (base_prompt or "").strip()

        if not names or not enabled:
            import nodes
            enc = nodes.CLIPTextEncode()
            full_text = f"{', '.join(names)}, {base}" if base else ", ".join(names) if names else base
            return (model, enc.encode(clip, full_text)[0])

        if len(names) > MAX_ARTISTS:
            logger.warning("[LLMMask] 画师数 %d 超过上限 %d，截断", len(names), MAX_ARTISTS)
            names = names[:MAX_ARTISTS]

        artist_names, artist_weights = _parse_artist_weights(names)
        n_artists = len(artist_names)

        # Build weighted artist parts — all names use (name:weight) syntax.
        # escape_important runs BEFORE parse_parentheses, converting \( → \0\2
        # so parenthesized names like \(cyoroama\) don't break nesting.
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

        # ── Precise ranges via word_id + name matching ──
        # Tokenize full text. With (name:1.0) syntax each artist gets a
        # unique word_id.  We locate artists by decoding each word_id's
        # token span and matching against known artist names — this
        # works regardless of where in the prompt artists are inserted.
        full_tok = clip.tokenize(full_text, return_word_ids=True)
        qwen_batch = full_tok["qwen3_06b"][0]

        wid_ranges = {}
        for idx, (_, _, wid) in enumerate(qwen_batch):
            if wid > 0:
                if wid not in wid_ranges:
                    wid_ranges[wid] = [idx, idx]
                wid_ranges[wid][1] = idx

        raw_tok = _get_raw_tokenizer(clip)
        artist_ranges = []

        # Match via word_id: each (name:weight) gets its own word_id.
        # decode → match against artist name → get precise token range.
        # Build map with both escaped and unescaped variants — unescape_important
        # strips \( → ( before tokenization, so decoded text may differ.
        name_lower_map = {}
        for i in range(n_artists):
            n = artist_names[i].lower()
            name_lower_map[n] = i
            n_unesc = n.replace("\\)", ")").replace("\\(", "(")
            if n_unesc != n:
                name_lower_map[n_unesc] = i
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

        # ── Verify ──
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
        mask = _build_llm_mask(all_ids, artist_ranges_pos, device)
        total = mask.shape[-1] * mask.shape[-2]
        blocked = (mask < 0).sum().item()
        logger.warning("[LLMMask] mask %dx%d, blocked %d/%d (%.1f%%)",
                    mask.shape[-1], mask.shape[-2], blocked, total,
                    100.0 * blocked / total if total > 0 else 0)

        # Find LLM transformer and encode with patched attention
        transformer = _find_llm_transformer(clip)
        import nodes
        encoder = nodes.CLIPTextEncode()
        with self._patch_llm_attention(transformer, mask):
            positive = encoder.encode(clip, full_text)[0]

        return (model, positive)


NODE_CLASS_MAPPINGS = {
    "AnimaArtistLLMMask": AnimaArtistLLMMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaArtistLLMMask": "Anima Artist LLM Mask (Qwen3 Attention)",
}
