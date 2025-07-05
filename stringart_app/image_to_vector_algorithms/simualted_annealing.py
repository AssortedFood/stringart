# stringart_app/image_to_vector_algorithms/simulated_annealing.py
import numpy as np
import random
import math
from typing import List, Dict
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
        sample_pairs: int = 1000  # unused
    ) -> List[Dict[str, int]]:
        h, w = pixels.shape
        anchors = generate_radial_anchors(n_anchors, w, h)

        # Precompute all pair coverage
        all_pairs = [(i,j) for i in range(n_anchors) for j in range(i+1, n_anchors)]
        coverage = {}
        target = pixels.astype(np.int16)
        blank = np.full_like(pixels, 255, dtype=np.int16)
        for (i,j) in all_pairs:
            temp = Image.fromarray(blank.astype(np.uint8), 'L')
            ImageDraw.Draw(temp).line([anchors[i], anchors[j]], fill=0, width=line_thickness)
            mask = np.array(temp, dtype=np.int16)
            coverage[(i,j)] = np.sum((mask - target)**2)

        # Initialize random solution
        current = set(random.sample(all_pairs, n_strings))
        best = current.copy()
        T0, alpha = 1.0, 0.995
        T = T0
        def score(sol):
            return sum(coverage[e] for e in sol)

        current_score = score(current)
        best_score = current_score

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
            if Δ < 0 or random.random() < math.exp(-Δ / T):
                current, current_score = new, new_score
                if current_score < best_score:
                    best, best_score = current.copy(), current_score
            T *= alpha
            if T < 1e-3:
                break

        return [{"from":i, "to":j} for (i,j) in best]
