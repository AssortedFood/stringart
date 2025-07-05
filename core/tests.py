# core/tests.py

import os
import numpy as np
from django.test import TestCase
from core.preprocessing import load_image_to_pixels  # absolute import

class PreprocessingTests(TestCase):
    def test_load_image_to_pixels(self):
        """
        Given our downloaded test.png, load_image_to_pixels should
        return a 2D NumPy array of dtype uint8 with non-zero dimensions.
        """
        # Path to your test image
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        img_path = os.path.join(project_root, 'test_images', 'test.png')
        
        # Load it
        pixels = load_image_to_pixels(img_path)
        
        # Assertions
        self.assertTrue(isinstance(pixels, np.ndarray), "Should return a NumPy array")
        self.assertEqual(pixels.ndim, 2, "Output should be a 2D array (grayscale)")
        self.assertGreater(pixels.shape[0], 0, "Height should be > 0")
        self.assertGreater(pixels.shape[1], 0, "Width should be > 0")
        self.assertEqual(pixels.dtype.name, 'uint8', "Array dtype should be uint8")
