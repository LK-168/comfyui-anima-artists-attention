import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

MAX_ARTISTS = 32


# ─── helpers (adapted from Artist-mixer) ───

def _split_artist_chain(chain):
    if not chain:
        return []
    s = str(chain).replace("，", ",").replace("\n", ",").replace("\r", ",")
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def _extract(conditioning):
    """Extract raw tensor, t5xxl_ids, t5xxl_weights from a conditioning output."""
    if conditioning is None:
        return None, None, None
    if not isinstance(conditioning, (list, tuple)) or len(conditioning) == 0:
        return None, None, None
    first = conditioning[0]
    if not isinstance(first, (list, tuple)) or len(first) == 0:
        return None, None, None
    raw = first[0] if torch.is_tensor(first[0]) else None
    extra = first[1] if len(first) > 1 and isinstance(first[1], dict) else {}
    return raw, extra.get("t5xxl_ids"), extra.get("t5xxl_weights")


def _normalize_weights(weights):
    """Normalize weights to sum to 1.0."""
    total = sum(abs(w) for w in weights)
    if total <= 1e-8:
        return [1.0 / len(weights)] * len(weights)
    return [w / total for w in weights]


def _pad_to_512(tensor):
    """Pad LLMAdapter output to Anima's expected 512-token length."""
    if tensor.shape[1] < 512:
        tensor = F.pad(tensor, (0, 0, 0, 512 - tensor.shape[1]))
    return tensor


# ─── Main node ───

class AnimaArtistMaskAttn:
    """
    Mask-based multi-artist mixing for Anima.

    Unlike the Artist-mixer approach (AnimaArtistCrossAttn) which patches
    DiT cross-attention blocks and feeds isolated 2-token artist embeddings,
    the Mask approach:

    1. Encodes each artist separately through CLIP to get raw LLM outputs
    2. Merges all LLM outputs and t5xxl_ids into a single sequence
    3. Builds a target_attention_mask that isolates artist blocks in LLMAdapter
    4. Passes the merged sequence + mask through LLMAdapter in a single pass
    5. Uses model_function_wrapper to inject precomputed cond context per CFG row

    This way:
    - Cross-attention receives a full-length, normally-distributed context
    - Artist embeddings are isolated via LLMAdapter self-attention mask
    - CFG works correctly (cond rows get artists, uncond rows get base only)
    - No adaLN issues (Anima has none)
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
                    "tooltip": "主词条。画出什么内容（不含画师名），例如 'a cat sitting'"
                }),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否启用画师注入"
                }),
                "normalize_weights": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "True: 多个画师的权重归一化。推荐保持 True"
                }),
            },
        }
    
    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "base_prompt")
    FUNCTION = "patch"
    CATEGORY = "Anima/CrossAttn"
    
    def _parse_artist_weights(self, names):
        """Parse artist names and optional weight syntax (name:weight)."""
        parsed = []
        weights = []
        for name in names:
            if ":" in name and name.rfind(":") > 0:
                # Check if the part after : looks like a number
                parts = name.rsplit(":", 1)
                try:
                    w = float(parts[1])
                    parsed.append(parts[0].strip())
                    weights.append(w)
                except ValueError:
                    parsed.append(name)
                    weights.append(1.0)
            else:
                parsed.append(name)
                weights.append(1.0)
        return parsed, weights
    
    def _encode_artist(self, clip, name, base_prompt):
        """Encode a single artist (with optional base context) through CLIP."""
        text = f"{name}, {base_prompt}" if base_prompt else name
        tokens = clip.tokenize(text)
        cond = clip.encode_from_tokens_scheduled(tokens)
        return cond

    def _encode_text(self, clip, text):
        """Encode plain text through CLIP (no artist prefix)."""
        tokens = clip.tokenize(text) if text else clip.tokenize("")
        return clip.encode_from_tokens_scheduled(tokens)
    
    def _build_mask(self, all_ids, num_artists, device):
        """Build target_attention_mask for LLMAdapter self-attention.
        
        Shape: [1, 1, total_len, total_len] with bool dtype.
        
        Rules:
        - Artists[i] can attend to: itself
        - Artists[i] CANNOT attend to: other artists[j] (j != i)
        - Base tokens (if any, after all artists) can attend to: everything
        
        Returns None if ids are missing (mask not applicable).
        """
        lengths = []
        for ids in all_ids:
            if ids is not None:
                lengths.append(ids.shape[-1])
            else:
                return None  # Cannot build mask without t5xxl_ids

        total_len = sum(lengths)
        # Start with all-to-all allowed
        mask = torch.ones(1, 1, total_len, total_len, device=device, dtype=torch.bool)

        # Compute token ranges for each block
        offset = 0
        ranges = []
        for length in lengths:
            ranges.append((offset, offset + length))
            offset += length

        # Block artist-to-artist attention
        for i in range(num_artists):
            si, ei = ranges[i]
            for j in range(num_artists):
                if i != j:
                    sj, ej = ranges[j]
                    mask[:, :, si:ei, sj:ej] = False

        return mask

    def patch(self, model, clip, artist_chain, base_prompt,
              enabled, normalize_weights):
        names = _split_artist_chain(artist_chain)
        base = (base_prompt or "").strip()
        if not base:
            pass  # base stays empty string

        if not names:
            return (model, self._encode_text(clip, base))

        if len(names) > MAX_ARTISTS:
            logger.warning("[MaskAttn] 画师数 %d 超过上限 %d，截断", len(names), MAX_ARTISTS)
            names = names[:MAX_ARTISTS]

        # Parse artist weights (supports name:weight syntax)
        artist_names, artist_weights = self._parse_artist_weights(names)
        n_artists = len(artist_names)

        if normalize_weights and n_artists > 1:
            artist_weights = _normalize_weights(artist_weights)

        # ─── Step 1: Encode each artist WITH base context ───
        # The LLM needs art-related context (base prompt) to encode the artist
        # name as a style reference. Without context, it's just an isolated name.
        raws, all_ids, all_weights = [], [], []
        for i, name in enumerate(artist_names):
            cond = self._encode_artist(clip, name, base)  # f"{name}\n{base}"
            raw, ids, w = _extract(cond)
            if raw is None:
                raise ValueError(f"[MaskAttn] 画师 '{name}' 编码失败")
            raw = raw * float(artist_weights[i])
            raws.append(raw)
            all_ids.append(ids)
            all_weights.append(w)

        # ─── Step 2: Encode base separately (for KSampler output only) ───
        base_cond = self._encode_text(clip, base)  # for KSampler positive output
        base_raw, base_ids, base_w = _extract(base_cond)
        has_base = base_raw is not None

        # Build merged sequence for LLMAdapter: artists only
        # (each already includes base context, no separate base entry needed)
        mask_raws = list(raws)
        mask_ids = list(all_ids)
        mask_ws = list(all_weights)

        num_artists_for_mask = n_artists  # only artists, not base

        # ─── Step 3: Get diffusion model and validate ───
        try:
            dm = model.get_model_object("diffusion_model")
        except Exception:
            dm = model.model.diffusion_model

        if not hasattr(dm, "preprocess_text_embeds"):
            raise ValueError("[MaskAttn] 模型没有 preprocess_text_embeds，不是 Anima")
        if not hasattr(dm, "llm_adapter"):
            raise ValueError("[MaskAttn] 模型没有 llm_adapter，不是 Anima")

        # ─── Step 4: Merge raw LLM outputs ───
        merged_source = torch.cat(
            [r.unsqueeze(0) if r.dim() == 2 else r for r in mask_raws], dim=1)

        valid_ids = [ids for ids in mask_ids if ids is not None]
        merged_ids = torch.cat(valid_ids, dim=-1) if valid_ids else None
        # LLMAdapter.forward expects 2D [B, N] not 1D [N]
        if merged_ids is not None and merged_ids.dim() == 1:
            merged_ids = merged_ids.unsqueeze(0)
        valid_ws = [w for w in mask_ws if w is not None]
        merged_w = torch.cat(valid_ws, dim=-1) if valid_ws else None
        if merged_w is not None and merged_w.dim() == 1:
            merged_w = merged_w.unsqueeze(0)

        # ─── Step 5: Build attention mask ───
        target_mask = self._build_mask(mask_ids, num_artists_for_mask, merged_source.device)

        # ─── Step 6: Pre-compute LLMAdapter outputs ───
        # Get the correct device/dtype from LLMAdapter weights
        first_param = next(dm.llm_adapter.parameters())
        adapter_dtype = first_param.dtype
        adapter_device = first_param.device

        merged_source = merged_source.to(device=adapter_device, dtype=adapter_dtype)
        if merged_ids is not None:
            merged_ids = merged_ids.to(device=adapter_device)
        if merged_w is not None:
            merged_w = merged_w.to(device=adapter_device, dtype=adapter_dtype)
        if target_mask is not None:
            target_mask = target_mask.to(device=adapter_device)

        with torch.inference_mode():
            # ── Single artist shortcut ──
            # No mask needed; use preprocess_text_embeds (same as normal path)
            if n_artists == 1 and has_base:
                raw = mask_raws[0]
                ids = mask_ids[0]
                w = mask_ws[0]
                if raw.dim() == 2:
                    raw = raw.unsqueeze(0)
                if ids is not None and ids.dim() == 1:
                    ids = ids.unsqueeze(0)
                if w is not None and w.dim() == 1:
                    w = w.unsqueeze(0)
                # t5xxl_weights must be [B, N, 1] for broadcasting with [B, M, dim]
                if w is not None and w.dim() == 2:
                    w = w.unsqueeze(-1)

                precomputed_artist = dm.preprocess_text_embeds(
                    raw.to(device=adapter_device, dtype=adapter_dtype),
                    ids.to(device=adapter_device) if ids is not None else None,
                    t5xxl_weights=(
                        w.to(device=adapter_device, dtype=adapter_dtype)
                        if w is not None else None
                    ),
                )
            else:
                # Multi-artist: use llm_adapter with mask isolation
                precomputed_artist = dm.llm_adapter(
                    merged_source, merged_ids,
                    target_attention_mask=target_mask,
                )
                if merged_w is not None:
                    precomputed_artist = precomputed_artist * merged_w.unsqueeze(-1)
                precomputed_artist = _pad_to_512(precomputed_artist)

        # ─── Step 7: Clone model and patch preprocess_text_embeds ───
        # preprocess_text_embeds is called once for positive cond, once for negative.
        # We use a counter: first call = artist version, second = original.
        m = model.clone()
        patched_dm = m.get_model_object("diffusion_model")

        if enabled:
            artist_out = precomputed_artist  # captured by closure
            original_pp = patched_dm.preprocess_text_embeds

            # Store positive input length: base prompt's token count
            # (positive input comes from base_cond, not artist encoding)
            pos_len = base_raw.shape[1] if has_base and base_raw.dim() >= 2 else None

            def patched_pp(text_embeds, text_ids, t5xxl_weights=None):
                # Differentiate positive vs negative by input length
                is_pos = (pos_len is not None and text_embeds.shape[1] == pos_len)
                if is_pos:
                    out = artist_out.to(device=text_embeds.device, dtype=text_embeds.dtype)
                    if out.shape[0] == 1 and text_embeds.shape[0] > 1:
                        out = out.expand(text_embeds.shape[0], -1, -1)
                    return out
                else:
                    return original_pp(text_embeds, text_ids, t5xxl_weights)

            patched_dm.preprocess_text_embeds = patched_pp

        return (m, base_cond)


# ─── Registration ───

NODE_CLASS_MAPPINGS = {
    "AnimaArtistMaskAttn": AnimaArtistMaskAttn,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaArtistMaskAttn": "Anima Artist Mask Attn (LLMAdapter Mask)",
}
