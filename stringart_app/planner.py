# stringart_app/planner.py

import time
import numpy as np
from .renderer import generate_radial_anchors
from PIL import Image, ImageDraw
from typing import List, Dict

DEBUG = True

def _log(msg: str):
    if DEBUG:
        print(msg)

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
    _log(f"Starting generate_string_vectors: n_anchors={n_anchors}, "
         f"n_strings={n_strings}, line_thickness={line_thickness}, sample_pairs={sample_pairs}")
    start_time = time.time()

    height, width = pixels.shape
    canvas: np.ndarray = np.full_like(pixels, 255, dtype=np.int16)
    anchors = generate_radial_anchors(n_anchors, width, height)
    vectors: List[Dict[str, int]] = []

    # Precompute all possible pairs once
    all_pairs = [(i, j)
                 for i in range(n_anchors)
                 for j in range(i + 1, n_anchors)]
    _log(f"Precomputed total possible anchor pairs: {len(all_pairs)}")

    before_error = np.sum((canvas - pixels) ** 2)
    _log(f"Initial error (canvas vs. target): {before_error}")

    for i in range(n_strings):
        _log(f"Iteration {i+1}/{n_strings} started")
        best_improvement = 0
        best_pair = None
        best_canvas = None

        # sample a subset of pairs each iteration
        if sample_pairs < len(all_pairs):
            idxs = np.random.choice(len(all_pairs), size=sample_pairs, replace=False)
            candidates = [all_pairs[k] for k in idxs]
            _log(f"  Sampled {len(candidates)} pairs from {len(all_pairs)} total")
        else:
            candidates = all_pairs
            _log(f"  Using all {len(candidates)} candidate pairs")

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
            _log(f"  No improvement found in iteration {i+1}; breaking early")
            break

        # Apply best found
        canvas = best_canvas  # type: ignore[assignment]
        vectors.append({"from": best_pair[0], "to": best_pair[1]})
        _log(f"  Iteration {i+1}: chose pair {best_pair} with improvement {best_improvement}")
        before_error -= best_improvement  # update error for next round

    elapsed = time.time() - start_time
    _log(f"Completed generate_string_vectors in {elapsed:.2f}s; total vectors: {len(vectors)}")

    return vectors
