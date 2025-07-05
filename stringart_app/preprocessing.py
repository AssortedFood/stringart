# stringart_app/preprocessing.py

from PIL import Image, ImageOps
import numpy as np

def load_image_to_pixels(path, size=None, levels=8, gamma=1.0, autocontrast=True):
    """
    Load an image from `path`, convert to grayscale, optionally resize,
    autocontrast, apply gamma correction, then quantize to `levels` gray values.
    Returns a NumPy array (H, W) dtype=uint8 with values in the set
    {0, 255/(levels-1), 2*255/(levels-1), …, 255}.
    """
    # 1. Load & convert to L mode
    img = Image.open(path).convert('L')
    if size:
        img = img.resize(size, Image.Resampling.LANCZOS)

    # 2. Autocontrast to push whites whiter (drop 1% extremes by default)
    if autocontrast:
        img = ImageOps.autocontrast(img, cutoff=1)

    # 3. Gamma correction via lookup table
    if gamma != 1.0:
        # build LUT: new = 255 * (old/255)**gamma
        lut = [round((i / 255) ** gamma * 255) for i in range(256)]
        img = img.point(lut)

    # 4. Quantize to `levels` equally spaced gray values
    arr = np.array(img, dtype=np.float32)
    scale = (levels - 1) / 255.0
    quantized = np.round(arr * scale) / scale

    return quantized.astype(np.uint8)
