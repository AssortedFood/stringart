# stringart_app/image_to_vector_algorithms/coverage.py

import numpy as np
from typing import List, Dict
from PIL import Image, ImageDraw

from .base import StringArtAlgorithm
from ..renderer import generate_radial_anchors

class CoverageMulticoverAlgorithm(StringArtAlgorithm):
    """
    At each step pick the anchor-pair whose line covers the most
    of the *remaining* darkness map.
    """

    def generate(
        self,
        pixels: np.ndarray,
        n_anchors: int = 180,
        n_strings: int = 200,
        line_thickness: int = 1,
        sample_pairs: int = 1000  # unused here, but kept for signature compatibility
    ) -> List[Dict[str, int]]:
        height, width = pixels.shape

        # 1. Linearize to float so that dark → large (0 = white, 1 = black)
        target = (255.0 - pixels.astype(np.float32)) / 255.0

        # 2. Generate anchor coords
        anchors = generate_radial_anchors(n_anchors, width, height)

        # 3. Precompute a binary mask (H×W) for each possible line ℓ = (i,j)
        all_pairs = [(i, j) for i in range(n_anchors) for j in range(i+1, n_anchors)]
        line_masks = []
        for (i, j) in all_pairs:
            img = Image.new('L', (width, height), color=0)
            draw = ImageDraw.Draw(img)
            draw.line([anchors[i], anchors[j]], fill=255, width=line_thickness)
            mask = np.array(img, dtype=bool)  # True where the string would lie
            line_masks.append(mask)

        # 4. Flatten masks and target once for dot-product use
        target_flat = target.ravel()
        masks_flat  = [mask.ravel() for mask in line_masks]

        # 5. Initial coverage score for each line = inner(mask_flat, target_flat)
        coverage = np.array([m.dot(target_flat) for m in masks_flat], dtype=np.float32)

        # 6. Iteratively pick the best line, subtract its mask from the residual map
        vectors: List[Dict[str, int]] = []
        residual = target_flat.copy()

        for _ in range(n_strings):
            # 6a. Compute score = inner(mask_flat, residual)
            scores = np.array([m.dot(residual) for m in masks_flat], dtype=np.float32)
            best_idx = int(np.argmax(scores))
            if scores[best_idx] <= 0:
                break  # nothing left to cover

            # 6b. Record the chosen vector
            i, j = all_pairs[best_idx]
            vectors.append({"from": i, "to": j})

            # 6c. Subtract that mask’s contribution, clamped to ≥0
            subtraction = masks_flat[best_idx] * (scores[best_idx] / coverage[best_idx])
            residual = np.maximum(residual - subtraction, 0.0)

        return vectors
