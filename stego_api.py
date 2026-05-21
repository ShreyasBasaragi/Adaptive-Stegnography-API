"""
stego_api.py
────────────
FastAPI application exposing the Neural Steganography Engine.

Endpoints:
  POST /embed   — Hide text inside an image (returns PNG)
  POST /extract — Recover hidden text from a stego image
  GET  /health  — Liveness probe

Run locally:
  uvicorn stego_api:app --reload --port 8000
"""

import io
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse

from stego_pipeline import embed_text, extract_text

app = FastAPI(
    title="Adaptive Steganography API",
    description=(
        "Embed secret text into images using an AI-guided adaptive LSB engine.\n\n"
        "- **POST /embed** — supply `text` and optionally an `image`; receive a PNG with the message hidden inside.\n"
        "- **POST /extract** — supply a stego PNG; receive the hidden text."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── POST /embed ───────────────────────────────────────────────────────

@app.post(
    "/embed",
    summary="Embed text into an image",
    response_class=Response,
    responses={
        200: {"content": {"image/png": {}}, "description": "Stego PNG with hidden text"},
        400: {"description": "Text too long for the image capacity"},
        500: {"description": "Internal server error"},
    },
)
async def embed(
    text: str = Form(..., description="The secret message to hide"),
    image: UploadFile | None = File(None, description="Optional cover image (PNG/JPG). If omitted, a random default is used."),
):
    """
    Embed `text` into a cover image using AI-guided adaptive LSB steganography.

    - The CamouflageNet analyses the image texture to build a per-pixel capacity heatmap.
    - Textured regions hide **3 bits** per channel; medium regions **2 bits**; flat regions **1 bit**.
    - The stego image is visually identical to the cover but carries the hidden payload.
    """
    image_bytes = await image.read() if image else None

    try:
        png_bytes = embed_text(text, image_bytes)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=stego.png"},
    )


# ── POST /extract ─────────────────────────────────────────────────────

@app.post(
    "/extract",
    summary="Extract hidden text from a stego image",
    response_class=JSONResponse,
)
async def extract(
    image: UploadFile = File(..., description="Stego PNG produced by /embed"),
):
    """
    Recover the secret message hidden inside a stego image.

    The same CamouflageNet heatmap is regenerated from the image to retrace
    exactly which pixels were used and how many bits each held.
    """
    image_bytes = await image.read()

    try:
        recovered = extract_text(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

    return {"text": recovered}


# ── GET /health ───────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "version": "1.0.0"}
