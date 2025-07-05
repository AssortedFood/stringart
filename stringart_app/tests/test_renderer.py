# stringart_app/tests/test_renderer.py

import numpy as np
from django.test import SimpleTestCase
from PIL import Image

from stringart_app.renderer import render_vector_list


class RendererTests(SimpleTestCase):
    def test_render_vector_list_outputs_correct_image(self):
        """Render a single diagonal string on a 4-anchor square."""
        vectors = [{"from": 0, "to": 2}]
        size = (100, 100)

        img = render_vector_list(vectors, size=size, n_anchors=4, line_width=3)

        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.mode, "L")
        self.assertEqual(img.size, size)

        data = np.array(img)
        # Expect at least one non-white pixel
        self.assertLess(data.min(), 255)
        # And average remains mostly white
        self.assertGreater(data.mean(), 200)
