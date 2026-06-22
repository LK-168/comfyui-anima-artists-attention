"""Attention / Mask / LLM 相关工具。"""

import torch
import logging

logger = logging.getLogger(__name__)


def find_llm_transformer(clip):
    """
    在 CLIP 对象中搜索 LLM transformer（用于 attention patch）。
    支持 Anima (Qwen3-0.6B) 等。
    """
    model = clip.cond_stage_model if hasattr(clip, "cond_stage_model") else clip

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
    raise RuntimeError("[LLMMask] 无法找到 LLM transformer")


def build_llm_mask(full_ids, artist_ranges, device):
    """
    构建 4D additive mask [1, 1, seq, seq] — 仅隔离画师之间的 attention。
    模型自带 causal mask，这里只叠加 -inf 阻断 artist→artist 互看。
    """
    seq_len = len(full_ids)
    mask = torch.zeros(1, 1, seq_len, seq_len, device=device, dtype=torch.float)

    for i, (si, ei) in enumerate(artist_ranges):
        for j, (sj, ej) in enumerate(artist_ranges):
            if i != j:
                mask[:, :, si:ei, sj:ej] = float("-inf")

    return mask
