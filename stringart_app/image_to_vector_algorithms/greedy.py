# stringart_app/image_to_vector_algorithms/greedy.py

import time
from typing import List, Dict

import numpy as np
from PIL import Image, ImageDraw

from .base import StringArtAlgorithm
from ..renderer import generate_radial_anchors

DEBUG = True

def _log(msg: str):
    if DEBUG:
        print(msg)

class GreedyAlgorithm(StringArtAlgorithm):
    def generate(
        self,
        pixels: np.ndarray,
        n_anchors: int = 180,
        n_strings: int = 200,
        line_thickness: int = 1,
        sample_pairs: int = 1000
    ) -> List[Dict[str, int]]:
        _log(f"Starting GreedyAlgorithm: n_anchors={n_anchors}, "
             f"n_strings={n_strings}, line_thickness={line_thickness}, sample_pairs={sample_pairs}")
        start_time = time.time()

        height, width = pixels.shape
        canvas: np.ndarray = np.full_like(pixels, 255, dtype=np.int16)
        anchors = generate_radial_anchors(n_anchors, width, height)
        vectors: List[Dict[str, int]] = []

        all_pairs = [
            (i, j)
            for i in range(n_anchors)
            for j in range(i + 1, n_anchors)
        ]
        _log(f"Precomputed total possible anchor pairs: {len(all_pairs)}")

        before_error = np.sum((canvas - pixels) ** 2)
        _log(f"Initial error (canvas vs. target): {before_error}")

        for i in range(n_strings):
            _log(f"Iteration {i+1}/{n_strings} started")
            best_improvement = 0
            best_pair = None
            best_canvas: np.ndarray = None  # type: ignore

            if sample_pairs < len(all_pairs):
                idxs = np.random.choice(len(all_pairs), size=sample_pairs, replace=False)
                candidates = [all_pairs[k] for k in idxs]
            else:
                candidates = all_pairs

            for a_idx, b_idx in candidates:
                # canvas is definitely an ndarray here:
                temp_img = Image.fromarray(canvas.astype(np.uint8), mode='L')
                draw = ImageDraw.Draw(temp_img)
                draw.line(
                    [anchors[a_idx], anchors[b_idx]],
                    fill=0,
                    width=line_thickness
                )
                temp_canvas = np.array(temp_img, dtype=np.int16)

                after_error = np.sum((temp_canvas - pixels) ** 2)
                improvement = before_error - after_error

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_pair = (a_idx, b_idx)
                    best_canvas = temp_canvas

            if best_pair is None:
                _log(f"No improvement in iteration {i+1}; stopping early")
                break

            # mypy/Pylance now knows best_canvas is ndarray
            canvas = best_canvas  # type: ignore
            vectors.append({"from": best_pair[0], "to": best_pair[1]})
            before_error -= best_improvement
            _log(f"  Picked {best_pair} (Δerror={best_improvement})")

        elapsed = time.time() - start_time
        _log(f"GreedyAlgorithm done in {elapsed:.2f}s; got {len(vectors)} vectors")
        return vectors
