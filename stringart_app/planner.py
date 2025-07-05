# stringart_app/planner.py

import numpy as np
from .renderer import generate_radial_anchors
from PIL import Image, ImageDraw
from typing import List, Dict

def generate_string_vectors(
    pixels: np.ndarray,
    n_anchors: int = 180,
    n_strings: int = 200,
    line_thickness: int = 1,
    sample_pairs: int = 1000
) -> List[Dict[str, int]]:
    """
    Given a small grayscale pixel map `pixels` (2D uint8 array),
    produce up to `n_strings` string-art vectors (anchor index pairs).
    Uses a greedy algorithm sampling `sample_pairs` candidates per iteration.
    """
    height, width = pixels.shape
    canvas: np.ndarray = np.full_like(pixels, 255, dtype=np.int16)
    anchors = generate_radial_anchors(n_anchors, width, height)
    vectors: List[Dict[str, int]] = []

    # Precompute all possible pairs once
    all_pairs = [(i, j)
                 for i in range(n_anchors)
                 for j in range(i + 1, n_anchors)]

    for _ in range(n_strings):
        best_improvement = 0
        best_pair = None
        best_canvas = None

        before_error = np.sum((canvas - pixels) ** 2)

        # sample a subset of pairs each iteration
        if sample_pairs < len(all_pairs):
            idxs = np.random.choice(len(all_pairs), size=sample_pairs, replace=False)
            candidates = [all_pairs[i] for i in idxs]
        else:
            candidates = all_pairs

        for a_idx, b_idx in candidates:
            # Assert to narrow type: canvas is definitely an ndarray here
            assert isinstance(canvas, np.ndarray)
            temp_img = Image.fromarray(canvas.astype(np.uint8), mode='L')
            draw = ImageDraw.Draw(temp_img)
            draw.line([anchors[a_idx], anchors[b_idx]],
                      fill=0, width=line_thickness)
            temp_canvas = np.array(temp_img, dtype=np.int16)

            after_error = np.sum((temp_canvas - pixels) ** 2)
            improvement = before_error - after_error

            if improvement > best_improvement:
                best_improvement = improvement
                best_pair = (a_idx, b_idx)
                best_canvas = temp_canvas

        if best_pair is None:
            break

        # best_canvas was set whenever best_pair is not None
        canvas = best_canvas  # type: ignore[assignment]
        vectors.append({"from": best_pair[0], "to": best_pair[1]})

    return vectors
