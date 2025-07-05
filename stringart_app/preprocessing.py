# stringart_app/preprocessing.py

from PIL import Image
import numpy as np

def load_image_to_pixels(path, size=None):
    """
    Load an image from `path` (PNG/JPG/JPEG),
    convert to grayscale, optionally resize,
    and return a NumPy array of shape (H, W) dtype=uint8.
    """
    img = Image.open(path).convert('L')  # grayscale
    if size:
        img = img.resize(size)
    return np.array(img)
