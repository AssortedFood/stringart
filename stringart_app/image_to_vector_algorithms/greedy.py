# stringart_app/image_to_vector_algorithms/greedy.py

import time
import logging
from typing import List, Dict, Optional

import numpy as np
from PIL import Image, ImageDraw

from .base import StringArtAlgorithm
from ..renderer import generate_radial_anchors


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
        sample_pairs: int = 1000,
        logger: Optional[logging.Logger] = None
    ) -> List[Dict[str, int]]:
        if logger is None:
            logger = logging.getLogger(__name__)

        logger.debug(
            f"[greedy] Starting: anchors={n_anchors}, strings={n_strings}, "
            f"thickness={line_thickness}, samples={sample_pairs}"
        )
        start_time = time.time()

        height, width = pixels.shape
        # canvas initialized white (255)
        canvas: np.ndarray = np.full_like(pixels, 255, dtype=np.int16)

        # 1. Generate anchors
        anchors = generate_radial_anchors(n_anchors, width, height, logger=logger)
        logger.debug(f"[greedy] Generated {len(anchors)} anchors")

        # 2. Precompute all anchor‐pairs and their chord lengths
        all_pairs = [(i, j) for i in range(n_anchors) for j in range(i + 1, n_anchors)]
        lengths = np.array([
            np.hypot(anchors[i][0] - anchors[j][0], anchors[i][1] - anchors[j][1])
            for i, j in all_pairs
        ], dtype=np.float32)
        norm_factors = lengths ** self.ALPHA + 1e-6
        logger.debug(f"[greedy] Prepared {len(all_pairs)} candidate chords")

        # 3. Precompute binary masks for each line
        masks_flat = []
        for (i, j) in all_pairs:
            img = Image.new('L', (width, height), color=0)
            draw = ImageDraw.Draw(img)
            draw.line([anchors[i], anchors[j]], fill=255, width=line_thickness)
            masks_flat.append(np.array(img, dtype=bool).ravel())
        logger.debug(f"[greedy] Precomputed {len(masks_flat)} masks")

        # 4. Precompute static coverage (for pruning)
        target_flat = pixels.astype(np.float32).ravel()
        static_cover = np.array([m.dot(target_flat) for m in masks_flat], dtype=np.float32)

        before_error = np.sum((canvas - pixels) ** 2)
        logger.debug(f"[greedy] Initial SSE error: {before_error:.1f}")

        vectors: List[Dict[str, int]] = []

        # 5. Main greedy loop
        for iteration in range(n_strings):
            logger.debug(f"[greedy] Iteration {iteration+1}/{n_strings}")

            # residual: where canvas is brighter than target
            residual_flat = (canvas - pixels.astype(np.int16)).clip(min=0).ravel()

            # smart sampling: endpoint darkness
            anchor_darkness = np.zeros(n_anchors, dtype=np.float32)
            for idx, (i, j) in enumerate(all_pairs):
                d = masks_flat[idx].dot(residual_flat)
                anchor_darkness[i] += d
                anchor_darkness[j] += d

            probs = anchor_darkness / (anchor_darkness.sum() + 1e-6)

            a_choices = np.random.choice(n_anchors, sample_pairs, p=probs)
            b_choices = np.random.choice(n_anchors, sample_pairs, p=probs)

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

            if not candidate_idxs:
                candidate_idxs = list(range(len(all_pairs)))

            best_norm_score = 0.0
            best_improvement = 0.0
            best_pair = None
            best_canvas = None  # type: ignore

            # evaluate candidates
            for k in candidate_idxs:
                i_idx, j_idx = all_pairs[k]
                temp = Image.fromarray(canvas.astype(np.uint8))
                draw = ImageDraw.Draw(temp)
                draw.line([anchors[i_idx], anchors[j_idx]], fill=0, width=line_thickness)
                temp_canvas = np.array(temp, dtype=np.int16)

                after_error = np.sum((temp_canvas - pixels) ** 2)
                imp = before_error - after_error
                if imp <= 0:
                    continue

                norm_score = imp / norm_factors[k]
                if norm_score > best_norm_score:
                    best_norm_score = norm_score
                    best_improvement = imp
                    best_pair = (i_idx, j_idx)
                    best_canvas = temp_canvas

            if best_pair is None:
                logger.debug(f"[greedy] No further improvement; stopping at iteration {iteration+1}")
                break

            # commit best pick
            canvas = best_canvas  # type: ignore
            vectors.append({"from": best_pair[0], "to": best_pair[1]})
            before_error -= best_improvement
            logger.debug(
                f"[greedy] Picked chord {best_pair} ΔSSE={best_improvement:.1f} norm_score={best_norm_score:.4f}"
            )

            # pruning
            if len(vectors) % self.PRUNE_K == 0:
                thresh = np.percentile(static_cover, self.PRUNE_PCT)
                keep = static_cover >= thresh
                all_pairs = [p for p, k in zip(all_pairs, keep) if k]
                masks_flat = [m for m, k in zip(masks_flat, keep) if k]
                norm_factors = norm_factors[keep]
                static_cover = static_cover[keep]
                logger.debug(f"[greedy] Pruned to {len(all_pairs)} candidates (threshold={thresh:.2f})")

        elapsed = time.time() - start_time
        logger.debug(f"[greedy] Done in {elapsed:.2f}s; total picks={len(vectors)}")
        return vectors
