# ComfyUI Anima Artists Attention

Mask-based multi-artist mixing for Anima, preventing artist style cross-contamination at the attention level.

## Nodes

### Anima Artist LLM Mask (Qwen3 Attention)

Patches Qwen3-0.6B self-attention layers during CLIP encoding to isolate artist token ranges. Each artist can only attend to itself and the base prompt — not to other artists.

**Inputs:**
| Name | Type | Description |
|---|---|---|
| `model` | MODEL | Anima UNet model |
| `clip` | CLIP | Anima CLIP model (Qwen3-0.6B) |
| `artist_chain` | STRING | Comma-separated artist names. Supports Danbooru `\(` escapes and `(name:weight)` syntax |
| `base_prompt` | STRING | Content description (without artist names) |
| `enabled` | BOOLEAN | Enable/disable artist injection |
| `patch_position` | STRING | Optional marker text in base_prompt where artists are inserted (empty = prepend) |

**Outputs:** MODEL, CONDITIONING (positive)

### Example

```
artist_chain: @hatsushiro mamimu, (@uenomigi:0.8), @yanggaengwang, (@naguru \(cyoroama\):0.5)
base_prompt:  masterpiece, best quality, 1girl, sky, clouds
```

## How It Works

1. Builds combined prompt inserting artists at the specified position
2. Tokenizes with `(name:weight)` syntax to assign unique word_ids per artist
3. Locates each artist's token range via word_id + name matching
4. Builds a 4D attention mask blocking artist-to-artist cross-attention
5. Patches Qwen3 self-attention layers during a single CLIP encode pass
6. Restores the original attention forward after encoding

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/LK-168/comfyui-anima-artists-attention.git
```

Requires ComfyUI with Anima model support.

## Limitations

- Only tested with Anima (Qwen3-0.6B text encoder)
- Artist count limited to 32
