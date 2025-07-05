# stringart_app/tests/test_preprocessing.py

import io
import numpy as np
from PIL import Image
from django.test import SimpleTestCase

from stringart_app.preprocessing import load_image_to_pixels


class PreprocessingTests(SimpleTestCase):
    def test_load_image_to_pixels_accepts_various_formats(self):
        """Ensure PNG, JPEG, BMP all load without error."""
        gradient = np.tile(np.linspace(0, 255, 50, dtype=np.uint8), (50, 1))
        pil = Image.fromarray(gradient)

        for fmt in ("PNG", "JPEG", "BMP"):
            buf = io.BytesIO()
            pil.save(buf, format=fmt)
            buf.seek(0)

            out = load_image_to_pixels(buf, size=(50, 50), levels=8)
            # basic shape & dtype checks
            self.assertIsInstance(out, np.ndarray)
            self.assertEqual(out.dtype, np.uint8)
            self.assertEqual(out.shape, (50, 50))
            # values stay in [0,255]
            self.assertGreaterEqual(out.min(), 0)
            self.assertLessEqual(out.max(), 255)

    def test_load_image_to_pixels_levels_quantization(self):
        """Quantizing to N levels yields exactly N distinct gray values."""
        # create & save a gradient file
        gradient = np.tile(np.arange(256, dtype=np.uint8), (10, 1))
        img = Image.fromarray(gradient)
        path = io.BytesIO()
        img.save(path, format="PNG")
        path.seek(0)

        arr = load_image_to_pixels(path, size=(10, 256), levels=4,
                                   gamma=1.0, autocontrast=False)
        unique = sorted(set(arr.ravel().tolist()))
        self.assertEqual(len(unique), 4)

        # expected quantized levels
        expected = [round(255 * i / 3) for i in range(4)]
        for v in unique:
            self.assertIn(v, expected)
