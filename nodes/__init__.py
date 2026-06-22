"""节点模块。"""

from .llm_mask import AnimaArtistLLMMask
from .entropy import AnimaCrossAttnEntropy

NODE_CLASS_MAPPINGS = {
    "AnimaArtistLLMMask": AnimaArtistLLMMask,
    "AnimaCrossAttnEntropy": AnimaCrossAttnEntropy,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaArtistLLMMask": "Anima Artist LLM Mask (Qwen3 Attention)",
    "AnimaCrossAttnEntropy": "Anima Cross-Attn Entropy Analysis",
}

__all__ = [
    "AnimaArtistLLMMask",
    "AnimaCrossAttnEntropy",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
