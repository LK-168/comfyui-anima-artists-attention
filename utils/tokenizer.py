"""Tokenizer 相关工具：自动检测、编码、画师定位。"""

import torch
import logging

logger = logging.getLogger(__name__)


def get_raw_tokenizer(clip):
    """
    从任意 ComfyUI CLIP 模型中自动检测并返回底层 HuggingFace tokenizer。
    支持 Anima (Qwen3)、SD1/SDXL 等。
    """
    tokenizer_obj = clip.tokenizer
    te_name = getattr(clip.cond_stage_model, 'clip_name', '') if hasattr(clip, 'cond_stage_model') else ''
    if te_name == 'qwen3_06b' and hasattr(tokenizer_obj, 'qwen3_06b'):
        sub = tokenizer_obj.qwen3_06b
        if hasattr(sub, 'tokenizer'):
            return sub.tokenizer
    if hasattr(tokenizer_obj, 'tokenizer'):
        return tokenizer_obj.tokenizer
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
    raise RuntimeError("无法自动检测 tokenizer，模型类型: %s",
                       type(tokenizer_obj).__name__)


def encode_prompt(clip, text):
    """便捷的 CLIPTextEncode 封装。"""
    import nodes
    enc = nodes.CLIPTextEncode()
    return enc.encode(clip, text)[0]


def locate_artist_ranges(clip, artist_names, full_text):
    """
    在完整 prompt 的 token 序列中精确定位每位画师的 token 范围。

    \u539f\u7406\uff1a
    1. \u7528 (name:weight) \u8bed\u6cd5\u7ed9\u6bcf\u4f4d\u753b\u5e08\u5206\u914d\u72ec\u7acb word_id
    2. \u89e3\u7801\u6bcf\u4e2a word_id \u7684 token span\uff0c\u4e0e\u753b\u5e08\u540d\u505a\u5b50\u4e32\u5339\u914d
    3. \u540c\u65f6\u5339\u914d\u8f6c\u4e49/\u975e\u8f6c\u4e49\u7248\u672c\u7684\u753b\u5e08\u540d\uff08unescape_important \u5904\u7406\uff09
    \u8fd4\u56de: [(start, end, artist_index), ...]
    """
    raw_tok = get_raw_tokenizer(clip)
    full_tok = clip.tokenize(full_text, return_word_ids=True)
    qwen_batch = full_tok["qwen3_06b"][0]

    wid_ranges = {}
    for idx, (_, _, wid) in enumerate(qwen_batch):
        if wid > 0:
            if wid not in wid_ranges:
                wid_ranges[wid] = [idx, idx]
            wid_ranges[wid][1] = idx

    name_lower_map = {}
    for i, n in enumerate(artist_names):
        nl = n.lower()
        name_lower_map[nl] = i
        nu = nl.replace("\\)", ")").replace("\\(", "(")
        if nu != nl:
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

    return artist_ranges
