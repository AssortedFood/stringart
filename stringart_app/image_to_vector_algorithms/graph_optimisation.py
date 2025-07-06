# stringart_app/image_to_vector_algorithms/graph_optimisation.py

import numpy as np
from typing import List, Dict, Optional, Callable
from PIL import Image, ImageDraw

from .base import StringArtAlgorithm
from ..renderer import generate_radial_anchors
import logging

# This algorithm requires PuLP: pip install pulp
import pulp


class GraphOptimisationAlgorithm(StringArtAlgorithm):
    """
    Select exactly N strings (edges) to maximize total coverage of the darkness map,
    formulated as a 0/1 integer linear program:
        maximize   Σₗ coverageₗ ⋅ xₗ
        subject to Σₗ xₗ = n_strings
                   xₗ ∈ {0,1}
    """

    def generate(
        self,
        pixels: np.ndarray,
        n_anchors: int = 180,
        n_strings: int = 200,
        line_thickness: int = 1,
        sample_pairs: int = 1000,  # unused, kept for signature compatibility
        logger: Optional[logging.Logger] = None,
        *,
        vector_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, int]]:
        if logger is None:
            logger = logging.getLogger(__name__)

        height, width = pixels.shape
        logger.debug(f"[graph_optimisation] Starting generate — anchors={n_anchors}, strings={n_strings}")

        # 1. Build darkness map: 0 (white) → 1 (black)
        target = (255.0 - pixels.astype(np.float32)) / 255.0
        logger.debug("[graph_optimisation] Built darkness map")

        # 2. Generate anchors
        anchors = generate_radial_anchors(n_anchors, width, height, logger=logger)
        logger.debug(f"[graph_optimisation] Generated {len(anchors)} anchors")

        # 3. Enumerate all possible anchor-pairs
        all_pairs = [(i, j) for i in range(n_anchors) for j in range(i + 1, n_anchors)]
        n_pairs = len(all_pairs)
        logger.debug(f"[graph_optimisation] Enumerated {n_pairs} anchor-pairs")

        # 4. Precompute coverage for each pair
        masks = []
        coverage = []
        target_flat = target.ravel()
        logger.debug("[graph_optimisation] Precomputing coverage for each pair")
        for idx, (i, j) in enumerate(all_pairs, start=1):
            img = Image.new('L', (width, height), color=0)
            draw = ImageDraw.Draw(img)
            draw.line([anchors[i], anchors[j]], fill=255, width=line_thickness)
            mask = np.array(img, dtype=bool).ravel()
            masks.append(mask)
            cov = float(mask.dot(target_flat))
            coverage.append(cov)
            if idx % 5000 == 0:
                logger.debug(f"[graph_optimisation] Processed {idx}/{n_pairs} pairs")

        # 5. Set up the ILP
        logger.debug("[graph_optimisation] Setting up integer linear program (ILP)")
        prob = pulp.LpProblem("StringArt_MaxCoverage", pulp.LpMaximize)
        x = [pulp.LpVariable(f"x_{k}", cat="Binary") for k in range(n_pairs)]

        # Objective: maximize sum coverage[k] * x[k]
        prob += pulp.lpSum(coverage[k] * x[k] for k in range(n_pairs)), "TotalCoverage"

        # Constraint: pick exactly n_strings lines
        prob += pulp.lpSum(x) == n_strings, "NumStrings"

        # 6. Solve
        logger.debug("[graph_optimisation] Solving ILP...")
        solver = pulp.PULP_CBC_CMD(msg=False)  # silent CBC solver
        prob.solve(solver)
        logger.debug(f"[graph_optimisation] Solver status: {pulp.LpStatus[prob.status]}")

        # 7. Extract solution
        vectors: List[Dict[str, int]] = []
        for k, var in enumerate(x):
            val = pulp.value(var)
            if val is not None and val >= 0.5:
                i, j = all_pairs[k]
                vectors.append({"from": i, "to": j})
                if vector_callback:
                    vector_callback(i, j)

        logger.debug(f"[graph_optimisation] Completed with {len(vectors)} vectors")
        return vectors
