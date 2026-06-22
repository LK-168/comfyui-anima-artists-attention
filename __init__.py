"""
Artist Mask Attn - Mask-based multi-artist mixing for Anima.
Provides two approaches:
  - nodes.py: LLMAdapter-level mask (patches preprocess_text_embeds)
  - nodes_llm_mask.py: LLM-level mask (patches Qwen3 self-attention during CLIP encode)
"""

# from .nodes import NODE_CLASS_MAPPINGS as _MASK, NODE_DISPLAY_NAME_MAPPINGS as _DISPLAY_MASK
"""
Artist Mask Attn — Mask-based multi-artist mixing + quantitative analysis for Anima.

Modules:
  nodes/llm_mask.py — LLM-level attention mask (patches Qwen3 self-attention)
  nodes/entropy.py — Cross-attention entropy analysis
  utils/          — Shared utilities (artist parsing, tokenizer, attention)
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
