# stringart_app/image_to_vector_algorithms/hough_greedy.py

import logging
from typing import List, Dict, Tuple, Optional, Callable

import numpy as np
from PIL import Image, ImageDraw

# you need scikit-image on your PYTHONPATH for these:
# pip install scikit-image
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line

from .base import StringArtAlgorithm
from ..renderer import generate_radial_anchors


class HoughGreedyAlgorithm(StringArtAlgorithm):
    """
    1) Use a probabilistic Hough transform to select a small candidate set of lines.
    2) Greedily pick the line that most reduces squared‐error at each step.
    """

    def generate(
        self,
        pixels: np.ndarray,
        n_anchors: int = 180,
        n_strings: int = 200,
        line_thickness: int = 1,
        sample_pairs: int = 1000,  # unused here
        logger: Optional[logging.Logger] = None,
        *,
        vector_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, int]]:
        if logger is None:
            logger = logging.getLogger(__name__)

        height, width = pixels.shape
        logger.debug(f"[hough_greedy] Starting with {n_anchors} anchors and {n_strings} strings")

        # 1) Edge‐detect
        edges: np.ndarray = canny(pixels / 255.0, sigma=2.0)
        logger.debug(f"[hough_greedy] Detected edges")

        # 2) Hough → list of ((x0,y0),(x1,y1)) segments
        lines: List[Tuple[Tuple[float, float], Tuple[float, float]]] = \
            probabilistic_hough_line(edges,
                                     threshold=10,
                                     line_length=30,
                                     line_gap=5)
        logger.debug(f"[hough_greedy] Found {len(lines)} Hough line segments")

        # 3) Anchor positions, as an (n_anchors × 2) array
        anchors_list = generate_radial_anchors(n_anchors, width, height, logger=logger)
        anchors_arr: np.ndarray = np.array(anchors_list, dtype=float)

        # 4) Map each Hough segment to the nearest anchor‐pair
        pairs: set[Tuple[int, int]] = set()
        for (pt0, pt1) in lines:
            p0 = np.array(pt0, dtype=float)
            p1 = np.array(pt1, dtype=float)

            d0 = np.linalg.norm(anchors_arr - p0, axis=1)
            d1 = np.linalg.norm(anchors_arr - p1, axis=1)

            i = int(d0.argmin())
            j = int(d1.argmin())
            if i != j:
                pairs.add((min(i, j), max(i, j)))

        candidates: List[Tuple[int, int]] = list(pairs)
        logger.debug(f"[hough_greedy] Reduced to {len(candidates)} candidate anchor pairs")

        if not candidates:
            logger.warning("[hough_greedy] No candidates found; returning empty vector list")
            return []

        # 5) Precompute binary masks for all candidate pairs
        masks_flat = []
        for (i, j) in candidates:
            img = Image.new('L', (width, height), color=0)
            draw = ImageDraw.Draw(img)
            draw.line([anchors_list[i], anchors_list[j]], fill=255, width=line_thickness)
            masks_flat.append(np.array(img, dtype=bool).ravel())

        logger.debug(f"[hough_greedy] Precomputed {len(masks_flat)} masks")

        # 6) Prepare mutable canvas and residual
        canvas = np.full_like(pixels, 255, dtype=np.int16)
        residual = (canvas - pixels).astype(np.float32).ravel()

        vectors: List[Dict[str, int]] = []

        for iteration in range(n_strings):
            logger.debug(f"[hough_greedy] Iteration {iteration+1}/{n_strings}")

            # Compute scores for all candidates
            scores = np.array([mask.dot(residual) for mask in masks_flat], dtype=np.float32)

            best_idx = int(np.argmax(scores))
            best_score = scores[best_idx]

            if best_score <= 0:
                logger.debug(f"[hough_greedy] No positive score at iteration {iteration+1}; stopping")
                break

            i, j = candidates[best_idx]
            logger.debug(
                f"[hough_greedy] Pick {iteration+1}: chord ({i},{j}) score={best_score:.2f}"
            )

            # Append and callback
            vectors.append({"from": i, "to": j})
            if vector_callback:
                vector_callback(i, j)

            # Draw the selected line onto the canvas
            tmp_img = Image.fromarray(canvas.astype(np.uint8), mode='L')
            draw = ImageDraw.Draw(tmp_img)
            draw.line(
                [tuple(anchors_list[i]), tuple(anchors_list[j])],
                fill=0,
                width=line_thickness
            )
            canvas = np.array(tmp_img, dtype=np.int16)

            # Update residual
            residual = (canvas - pixels).astype(np.float32).clip(min=0).ravel()

        logger.debug(f"[hough_greedy] Completed with {len(vectors)} vectors")
        return vectors
