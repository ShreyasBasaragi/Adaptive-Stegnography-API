"""
stego_pipeline.py
─────────────────
Single-file bridge between the CamouflageNet AI model and the AdaptiveEngine.
Handles:
  - Loading the trained PyTorch weights
  - Generating per-pixel capacity heatmaps
  - Encoding text → bits (with EOF marker) and embedding into image pixels
  - Extracting bits from pixels and decoding back to text
"""

import os
import io
import random
import numpy as np
import torch
from PIL import Image, ImageOps

from models.camouflage_net import CamouflageNet
from stego.adaptive_engine import AdaptiveEngine

# ── Constants ────────────────────────────────────────────────────────
EOF_MARKER   = "1111111111111110"   # 16-bit sentinel written after text bits
COVER_SIZE   = (256, 256)
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "models", "weights", "camou_net_adversarial.pth")
COVERS_DIR   = os.path.join(os.path.dirname(__file__), "input_dataset")

# ── Lazy model singleton (loaded once, reused across requests) ────────
_model: CamouflageNet | None = None
_engine = AdaptiveEngine()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_model() -> CamouflageNet:
    global _model
    if _model is None:
        model = CamouflageNet()
        if os.path.exists(WEIGHTS_PATH):
            state = torch.load(WEIGHTS_PATH, map_location=_device)
            model.load_state_dict(state)
        else:
            # No weights found — model is untrained but structurally fine for testing
            pass
        model.eval()
        model.to(_device)
        _model = model
    return _model


# ── Bit conversion helpers ────────────────────────────────────────────

def text_to_bits(text: str) -> str:
    """Convert UTF-8 text to binary string, appending EOF marker."""
    raw = "".join(format(b, "08b") for b in text.encode("utf-8"))
    return raw + EOF_MARKER


def bits_to_text(bitstream: str) -> str:
    """Extract text from a bitstream, stopping at the EOF marker."""
    idx = bitstream.find(EOF_MARKER)
    if idx == -1:
        raise ValueError("No hidden data found in this image (EOF marker missing).")
    payload = bitstream[:idx]

    # Pad to byte boundary
    remainder = len(payload) % 8
    if remainder:
        payload = payload[: len(payload) - remainder]

    chars = []
    for i in range(0, len(payload), 8):
        byte = payload[i : i + 8]
        if len(byte) == 8:
            chars.append(chr(int(byte, 2)))
    return "".join(chars)


# ── Heatmap generation ────────────────────────────────────────────────

def _generate_heatmap(img_np: np.ndarray) -> np.ndarray:
    """Run CamouflageNet to produce a (H, W) float heatmap in [0,1]."""
    model = _get_model()

    # PIL → normalised tensor [1, 3, H, W]
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(_device)

    with torch.no_grad():
        cap_map = model(tensor)  # shape: [1, 1, H, W], values in (0,1) via sigmoid

    heatmap = cap_map.squeeze().cpu().numpy()  # (H, W)
    return heatmap


# ── Image helpers ─────────────────────────────────────────────────────

def _load_cover(image_bytes: bytes | None) -> np.ndarray:
    """
    Load and normalize a cover image to COVER_SIZE (256×256 RGB numpy array).
    If image_bytes is None, pick a random image from COVERS_DIR.
    """
    if image_bytes is not None:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    else:
        if not os.path.isdir(COVERS_DIR) or not os.listdir(COVERS_DIR):
            raise FileNotFoundError(
                f"No cover images found in '{COVERS_DIR}'. "
                "Run `python get_data.py` first to download default covers."
            )
        candidates = [
            f for f in os.listdir(COVERS_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not candidates:
            raise FileNotFoundError(f"No PNG/JPG files found in '{COVERS_DIR}'.")
        chosen = os.path.join(COVERS_DIR, random.choice(candidates))
        img = Image.open(chosen).convert("RGB")

    # Centre-crop to square, then resize
    img = ImageOps.fit(img, COVER_SIZE, Image.Resampling.LANCZOS)
    return np.array(img)


def _np_to_png_bytes(img_np: np.ndarray) -> bytes:
    """Convert numpy (H, W, 3) uint8 array to lossless PNG bytes."""
    pil = Image.fromarray(img_np.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


# ── Public API ────────────────────────────────────────────────────────

def embed_text(text: str, image_bytes: bytes | None = None) -> bytes:
    """
    Embed `text` into a cover image using AI-guided adaptive LSB.

    Parameters
    ----------
    text        : The secret message to hide.
    image_bytes : Raw bytes of a user-uploaded image, or None to use a random cover.

    Returns
    -------
    PNG bytes of the stego image (lossless, preserves hidden data).
    """
    cover_np = _load_cover(image_bytes)
    heatmap  = _generate_heatmap(cover_np)

    bits = text_to_bits(text)

    # Capacity check
    total_capacity = sum(
        _engine._capacity(heatmap[y, x]) * 3
        for y in range(heatmap.shape[0])
        for x in range(heatmap.shape[1])
    )
    if len(bits) > total_capacity:
        raise ValueError(
            f"Text too long: needs {len(bits)} bits, image capacity is {total_capacity} bits "
            f"(~{total_capacity // 8} bytes)."
        )

    stego_np, _ = _engine.embed(cover_np, heatmap, bits)
    return _np_to_png_bytes(stego_np)


def extract_text(image_bytes: bytes) -> str:
    """
    Extract a hidden message from a stego image.

    Parameters
    ----------
    image_bytes : Raw bytes of the stego PNG.

    Returns
    -------
    The recovered secret text string.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = ImageOps.fit(img, COVER_SIZE, Image.Resampling.LANCZOS)
    stego_np = np.array(img)

    heatmap  = _generate_heatmap(stego_np)
    bitstream = _engine.extract(stego_np, heatmap)
    return bits_to_text(bitstream)
