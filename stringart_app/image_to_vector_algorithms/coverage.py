# stringart_app/image_to_vector_algorithms/coverage.py

import numpy as np
from typing import List, Dict, Optional
from PIL import Image, ImageDraw

from skimage.feature import canny
from skimage.transform import probabilistic_hough_line

from .base import StringArtAlgorithm
from ..renderer import generate_radial_anchors
import logging


class CoverageMulticoverAlgorithm(StringArtAlgorithm):
    """
    1. Detect prominent line segments via Canny+Hough.
    2. Snap each segment’s endpoints to the nearest radial anchors.
    3. Run multicover on this reduced set of candidate chords.
    """

    # Hough parameters
    HOUGH_THRESHOLD   = 10
    HOUGH_LINE_LENGTH = 30
    HOUGH_LINE_GAP    = 5

    # length‐normalization exponent (0 = no normalization, 1 = full length penalty)
    ALPHA = 0.5

    def generate(
        self,
        pixels: np.ndarray,
        n_anchors: int = 180,
        n_strings: int = 200,
        line_thickness: int = 1,
        sample_pairs: int = 1000,  # unused here
        logger: Optional[logging.Logger] = None
    ) -> List[Dict[str, int]]:
        if logger is None:
            logger = logging.getLogger(__name__)

        height, width = pixels.shape
        logger.debug(f"[coverage] Starting generate — anchors={n_anchors}, strings={n_strings}")

        # 1. Build darkness map: (0=white → 1=black)
        target = (255.0 - pixels.astype(np.float32)) / 255.0
        target_flat = target.ravel()
        logger.debug("[coverage] Built darkness map")

        # 2. Generate our anchor coordinates
        anchors = generate_radial_anchors(n_anchors, width, height, logger=logger)
        anchors_arr = np.array(anchors, dtype=float)
        logger.debug(f"[coverage] Generated {len(anchors)} anchors")

        # 3. Edge-detect + Hough to get a small set of segments
        edges = canny(pixels / 255.0)
        segments = probabilistic_hough_line(
            edges,
            threshold=self.HOUGH_THRESHOLD,
            line_length=self.HOUGH_LINE_LENGTH,
            line_gap=self.HOUGH_LINE_GAP
        )
        logger.debug(f"[coverage] Detected {len(segments)} Hough segments")

        # 4. Snap each segment’s endpoints to nearest anchor index
        pair_set = set()
        for (p0, p1) in segments:
            p0, p1 = np.array(p0), np.array(p1)
            i = int(np.linalg.norm(anchors_arr - p0, axis=1).argmin())
            j = int(np.linalg.norm(anchors_arr - p1, axis=1).argmin())
            if i != j:
                pair_set.add((min(i, j), max(i, j)))
        all_pairs = list(pair_set)
        logger.debug(f"[coverage] Snapped segments → {len(all_pairs)} unique anchor-pairs")

        # If Hough gave too few candidates, fall back to full enumeration
        if len(all_pairs) < 100:
            all_pairs = [(i, j) for i in range(n_anchors) for j in range(i+1, n_anchors)]
            logger.debug(f"[coverage] Fallback to full enumeration: {len(all_pairs)} pairs")

        # 5. Precompute chord-lengths for normalization
        lengths = np.array([
            np.hypot(anchors[i][0] - anchors[j][0], anchors[i][1] - anchors[j][1])
            for i, j in all_pairs
        ], dtype=np.float32)
        norm_factors = lengths ** self.ALPHA + 1e-6

        # 6. Precompute binary masks only for this reduced set
        masks_flat = []
        for (i, j) in all_pairs:
            img = Image.new('L', (width, height), color=0)
            draw = ImageDraw.Draw(img)
            draw.line([anchors[i], anchors[j]], fill=255, width=line_thickness)
            masks_flat.append(np.array(img, dtype=bool).ravel())
        logger.debug(f"[coverage] Precomputed {len(masks_flat)} masks")

        # 7. Precompute raw coverage for subtraction
        raw_cov = np.array([m.dot(target_flat) for m in masks_flat], dtype=np.float32)

        # 8. Iteratively pick the best line (normalized), subtract from residual
        vectors: List[Dict[str, int]] = []
        residual = target_flat.copy()
        logger.debug("[coverage] Beginning iterative picks")

        for k in range(n_strings):
            raw_scores = np.array([m.dot(residual) for m in masks_flat], dtype=np.float32)
            scores = raw_scores / norm_factors
            best_idx = int(np.argmax(scores))
            if scores[best_idx] <= 0:
                logger.debug(f"[coverage] No positive score at iteration {k}; stopping")
                break

            i, j = all_pairs[best_idx]
            vectors.append({"from": i, "to": j})
            logger.debug(f"[coverage] Pick {k+1}: chord ({i},{j}) score={scores[best_idx]:.4f}")

            # subtract proportional to raw coverage
            residual = np.maximum(
                residual - masks_flat[best_idx] * (raw_scores[best_idx] / raw_cov[best_idx]),
                0.0
            )

        logger.debug(f"[coverage] Completed with {len(vectors)} vectors")
        return vectors
