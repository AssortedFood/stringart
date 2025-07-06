# stringart_app/image_to_vector_algorithms/simulated_annealing.py

import numpy as np
import random
import math
import logging
from typing import List, Dict, Optional
from PIL import Image, ImageDraw

from .base import StringArtAlgorithm
from ..renderer import generate_radial_anchors


class SimulatedAnnealingAlgorithm(StringArtAlgorithm):
    """
    Start from a random set of N strings, then swap in/out
    edges probabilistically to escape local minima.
    """

    def generate(
        self,
        pixels: np.ndarray,
        n_anchors: int = 180,
        n_strings: int = 200,
        line_thickness: int = 1,
        sample_pairs: int = 1000,  # unused
        logger: Optional[logging.Logger] = None
    ) -> List[Dict[str, int]]:
        if logger is None:
            logger = logging.getLogger(__name__)

        h, w = pixels.shape
        logger.debug(f"[annealing] Starting with anchors={n_anchors}, strings={n_strings}")

        anchors = generate_radial_anchors(n_anchors, w, h, logger=logger)

        # Precompute all pair coverage and masks
        all_pairs = [(i, j) for i in range(n_anchors) for j in range(i + 1, n_anchors)]
        coverage = {}
        target_flat = pixels.astype(np.float32).ravel()
        masks_flat = {}

        for (i, j) in all_pairs:
            img = Image.new('L', (w, h), color=255)
            draw = ImageDraw.Draw(img)
            draw.line([anchors[i], anchors[j]], fill=0, width=line_thickness)
            mask_flat = np.array(img, dtype=bool).ravel()
            masks_flat[(i, j)] = mask_flat
            diff = np.where(mask_flat, 0.0, 255.0) - target_flat
            coverage[(i, j)] = np.sum(diff ** 2)

        logger.debug(f"[annealing] Precomputed coverage for {len(all_pairs)} pairs")

        # Initialize random solution
        current = set(random.sample(all_pairs, n_strings))
        best = current.copy()

        T0, alpha = 1.0, 0.995
        T = T0
        restarts = 0

        def score(sol: set[tuple[int, int]]) -> float:
            return sum(coverage[e] for e in sol)

        current_score = score(current)
        best_score = current_score
        logger.debug(f"[annealing] Initial SSE={current_score:.2f}")

        for it in range(10000):
            # propose swap
            out = random.choice(list(current))
            inp = random.choice(all_pairs)
            if inp in current:
                continue

            new = current.copy()
            new.remove(out)
            new.add(inp)

            new_score = current_score - coverage[out] + coverage[inp]
            Δ = new_score - current_score

            # acceptance criterion
            if Δ < 0 or random.random() < math.exp(-Δ / T):
                current, current_score = new, new_score
                if current_score < best_score:
                    best, best_score = current.copy(), current_score
                    logger.debug(f"[annealing] Iter {it}: New best SSE={best_score:.2f}")

            T *= alpha

            # restart temperature if frozen
            if T < 1e-3:
                restarts += 1
                if restarts > 2:
                    logger.debug("[annealing] Temperature frozen, stopping")
                    break
                logger.debug(f"[annealing] Restarting temperature (restart #{restarts})")
                T = T0 * (0.5 ** restarts)  # lower starting temp each restart

        logger.debug(f"[annealing] Finished with SSE={best_score:.2f}")
        return [{"from": i, "to": j} for (i, j) in best]
