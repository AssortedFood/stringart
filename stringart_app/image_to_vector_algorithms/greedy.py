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
    # length‐normalization exponent (0 = no normalization, 1 = full length penalty)
    ALPHA = 0.5
    # prune every K picks
    PRUNE_K = 50
    # prune percentile (10th)
    PRUNE_PCT = 10

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
        # canvas initialized white (255)
        canvas: np.ndarray = np.full_like(pixels, 255, dtype=np.int16)
        anchors = generate_radial_anchors(n_anchors, width, height)

        # Precompute all anchor‐pairs and their chord lengths
        all_pairs = [(i, j) for i in range(n_anchors) for j in range(i + 1, n_anchors)]
        lengths = np.array([
            np.hypot(anchors[i][0] - anchors[j][0], anchors[i][1] - anchors[j][1])
            for i, j in all_pairs
        ], dtype=np.float32)
        norm_factors = lengths**self.ALPHA + 1e-6

        # Precompute binary masks for each possible line, flattened
        masks_flat = []
        for (i, j) in all_pairs:
            img = Image.new('L', (width, height), color=0)
            draw = ImageDraw.Draw(img)
            draw.line([anchors[i], anchors[j]], fill=255, width=line_thickness)
            masks_flat.append(np.array(img, dtype=bool).ravel())

        _log(f"Precomputed {len(all_pairs)} anchor‐pairs and masks")

        # Precompute static coverage (for pruning)
        target_flat = pixels.astype(np.float32).ravel()
        static_cover = np.array([m.dot(target_flat) for m in masks_flat], dtype=np.float32)

        before_error = np.sum((canvas - pixels) ** 2)
        _log(f"Initial error (canvas vs. target): {before_error}")

        vectors: List[Dict[str, int]] = []

        for iteration in range(n_strings):
            _log(f"Iteration {iteration+1}/{n_strings} started")

            # Build residual intensity map: positive where canvas > target
            residual_flat = (canvas - pixels.astype(np.int16)).clip(min=0).ravel()

            # Smart endpoint sampling: compute per-anchor darkness
            anchor_darkness = np.zeros(n_anchors, dtype=np.float32)
            for idx, (i, j) in enumerate(all_pairs):
                d = masks_flat[idx].dot(residual_flat)
                anchor_darkness[i] += d
                anchor_darkness[j] += d

            # probabilities over anchors
            probs = anchor_darkness / (anchor_darkness.sum() + 1e-6)

            # sample endpoints
            a_choices = np.random.choice(n_anchors, sample_pairs, p=probs)
            b_choices = np.random.choice(n_anchors, sample_pairs, p=probs)

            # build candidate list (unique indices)
            candidate_idxs = []
            seen = set()
            for a, b in zip(a_choices, b_choices):
                if a == b:
                    continue
                pair = (min(a, b), max(a, b))
                try:
                    k = all_pairs.index(pair)
                except ValueError:
                    continue
                if k not in seen:
                    seen.add(k)
                    candidate_idxs.append(k)

            # fallback to uniform sampling
            if not candidate_idxs:
                candidate_idxs = list(range(len(all_pairs)))

            best_norm_score = 0.0
            best_improvement = 0.0
            best_pair = None
            best_canvas = None  # type: ignore

            # evaluate sampled candidates
            for k in candidate_idxs:
                i_idx, j_idx = all_pairs[k]

                # draw the line on a temp canvas
                temp = Image.fromarray(canvas.astype(np.uint8))
                draw = ImageDraw.Draw(temp)
                draw.line([anchors[i_idx], anchors[j_idx]], fill=0, width=line_thickness)
                temp_canvas = np.array(temp, dtype=np.int16)

                # compute improvement
                after_error = np.sum((temp_canvas - pixels) ** 2)
                imp = before_error - after_error
                if imp <= 0:
                    continue

                # normalize by chord length
                norm_score = imp / norm_factors[k]

                if norm_score > best_norm_score:
                    best_norm_score = norm_score
                    best_improvement = imp
                    best_pair = (i_idx, j_idx)
                    best_canvas = temp_canvas

            if best_pair is None:
                _log(f"No further improvement at iteration {iteration+1}; stopping early")
                break

            # commit best line
            canvas = best_canvas  # type: ignore
            vectors.append({"from": best_pair[0], "to": best_pair[1]})
            before_error -= best_improvement
            _log(f"  Picked {best_pair} (Δerror={best_improvement:.1f}, norm_score={best_norm_score:.4f})")

            # Prune low-yield lines every PRUNE_K picks
            if len(vectors) % self.PRUNE_K == 0:
                thresh = np.percentile(static_cover, self.PRUNE_PCT)
                keep = static_cover >= thresh

                all_pairs    = [p for p, k in zip(all_pairs,    keep) if k]
                masks_flat   = [m for m, k in zip(masks_flat,   keep) if k]
                norm_factors = norm_factors[keep]
                static_cover = static_cover[keep]

                _log(f"  Pruned to {len(all_pairs)} candidates (threshold={thresh:.2f})")

        elapsed = time.time() - start_time
        _log(f"GreedyAlgorithm done in {elapsed:.2f}s; got {len(vectors)} vectors")
        return vectors
