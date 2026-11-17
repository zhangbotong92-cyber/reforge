"""REFORGE — reverse-engineer any AI image/video into a reproducible recipe.

The name of the game is *reproducibility*, not just captioning:

    1. forensics  -> the file may already contain its own recipe (PNG/EXIF/ComfyUI/C2PA)
    2. caption    -> structured, dialect-ready description (pluggable VLM backends)
    3. controls   -> structure/identity signals that text alone cannot carry
    4. recipe     -> prompt + controls + pins, stamped back into the file
    5. score      -> closed-loop verification so "reproducible" is measurable
"""

from __future__ import annotations

__version__ = "0.3.1"
__all__ = ["forensics", "metrics", "providers", "schema", "recipe", "controls", "video", "invert", "refine"]
