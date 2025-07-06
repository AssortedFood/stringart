# stringart_app/preprocessing.py

from PIL import Image, ImageOps
import numpy as np
import logging
from typing import Optional, Tuple, Union, BinaryIO
import os

# === Configuration ===
# How many distinct gray‐levels to quantize to.
# Increase or decrease this to control the number of overlapping string layers.
DEFAULT_LEVELS = 8


def load_image_to_pixels(
    path: Union[str, bytes, os.PathLike, BinaryIO],  # Accept file-like objects
    size: Optional[Tuple[int, int]] = None,
    levels: int = DEFAULT_LEVELS,
    gamma: float = 1.0,
    autocontrast: bool = True,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Load an image from `path`, convert to grayscale, optionally resize,
    autocontrast, apply gamma correction, then quantize to `levels` gray values.
    Returns a NumPy array (H, W) dtype=uint8 with values in the set
    {0, 255/(levels-1), 2*255/(levels-1), …, 255}.

    :param path: file path, file-like object, or bytes for PIL to open
    :param size: optional (width, height) to resize the image to
    :param levels: number of gray levels to quantize to
    :param gamma: gamma correction exponent
    :param autocontrast: whether to apply PIL.ImageOps.autocontrast
    :param logger: optional logger to receive debug messages
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.debug("Loading image to grayscale")
    img = Image.open(path).convert('L')

    if size:
        logger.debug(f"Resizing image to {size}")
        img = img.resize(size, Image.Resampling.LANCZOS)

    if autocontrast:
        logger.debug("Applying autocontrast")
        img = ImageOps.autocontrast(img, cutoff=1)

    if gamma != 1.0:
        logger.debug(f"Applying gamma correction (gamma={gamma})")
        # build LUT: new = 255 * (old/255)**gamma
        lut = [round((i / 255) ** gamma * 255) for i in range(256)]
        img = img.point(lut)

    logger.debug(f"Quantizing to {levels} gray levels")
    arr = np.array(img, dtype=np.float32)
    scale = (levels - 1) / 255.0
    quantized = np.round(arr * scale) / scale

    result = quantized.astype(np.uint8)
    logger.debug("Finished preprocessing image")
    return result
