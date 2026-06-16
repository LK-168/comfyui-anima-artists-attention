"""
Artist Mask Attn - Mask-based multi-artist mixing for Anima.
Provides two approaches:
  - nodes.py: LLMAdapter-level mask (patches preprocess_text_embeds)
  - nodes_llm_mask.py: LLM-level mask (patches Qwen3 self-attention during CLIP encode)
"""

# from .nodes import NODE_CLASS_MAPPINGS as _MASK, NODE_DISPLAY_NAME_MAPPINGS as _DISPLAY_MASK
from .nodes_llm_mask import NODE_CLASS_MAPPINGS as _LLM, NODE_DISPLAY_NAME_MAPPINGS as _DISPLAY_LLM

# NODE_CLASS_MAPPINGS = {**_MASK, **_LLM}
NODE_CLASS_MAPPINGS = {**_LLM}
# NODE_DISPLAY_NAME_MAPPINGS = {**_DISPLAY_MASK, **_DISPLAY_LLM}
NODE_DISPLAY_NAME_MAPPINGS = {**_DISPLAY_LLM}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
