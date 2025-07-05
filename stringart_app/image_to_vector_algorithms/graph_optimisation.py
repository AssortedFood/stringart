# stringart_app/image_to_vector_algorithms/graph_optimisation.py

import numpy as np
from typing import List, Dict
from PIL import Image, ImageDraw

from .base import StringArtAlgorithm
from ..renderer import generate_radial_anchors

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
        sample_pairs: int = 1000  # unused, kept for signature compatibility
    ) -> List[Dict[str, int]]:
        height, width = pixels.shape

        # 1. Build darkness map: 0 (white) → 1 (black)
        target = (255.0 - pixels.astype(np.float32)) / 255.0

        # 2. Generate anchors
        anchors = generate_radial_anchors(n_anchors, width, height)

        # 3. Enumerate all possible anchor-pairs
        all_pairs = [(i, j) for i in range(n_anchors) for j in range(i + 1, n_anchors)]
        n_pairs = len(all_pairs)

        # 4. Precompute coverage for each pair
        masks = []
        coverage = []
        target_flat = target.ravel()
        for (i, j) in all_pairs:
            img = Image.new('L', (width, height), color=0)
            draw = ImageDraw.Draw(img)
            draw.line([anchors[i], anchors[j]], fill=255, width=line_thickness)
            mask = np.array(img, dtype=bool).ravel()
            masks.append(mask)
            coverage.append(float(mask.dot(target_flat)))

        # 5. Set up the ILP
        prob = pulp.LpProblem("StringArt_MaxCoverage", pulp.LpMaximize)
        # decision variables x_0 ... x_{n_pairs-1}
        x = [pulp.LpVariable(f"x_{k}", cat="Binary") for k in range(n_pairs)]

        # objective: maximize sum coverage[k] * x[k]
        prob += pulp.lpSum(coverage[k] * x[k] for k in range(n_pairs)), "TotalCoverage"

        # constraint: pick exactly n_strings lines
        prob += pulp.lpSum(x) == n_strings, "NumStrings"

        # 6. Solve
        solver = pulp.PULP_CBC_CMD(msg=False)  # silent CBC solver
        prob.solve(solver)

        # 7. Extract solution
        vectors: List[Dict[str, int]] = []
        for k, var in enumerate(x):
            val = pulp.value(var)
            if val is not None and val >= 0.5:
                i, j = all_pairs[k]
                vectors.append({"from": i, "to": j})

        return vectors
