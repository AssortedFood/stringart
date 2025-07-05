# stringart_app/image_to_vector_algorithms/__init__.py

from typing import List, Dict
import numpy as np

from .base import StringArtAlgorithm
from .greedy import GreedyAlgorithm
from .coverage import CoverageMulticoverAlgorithm
from .graph_optimisation import GraphOptimisationAlgorithm
# import other algorithms here as you add them:
# from .image_to_vector_algorithms.random import RandomPairAlgorithm

# === registry of available strategies ===
ALGORITHMS: Dict[str, StringArtAlgorithm] = {
    "greedy": GreedyAlgorithm(),
    "coverage": CoverageMulticoverAlgorithm(),
    "graph": GraphOptimisationAlgorithm(),
    # "random": RandomPairAlgorithm(),
    # add new ones here...
}

def generate_string_vectors(
    pixels: np.ndarray,
    n_anchors: int = 180,
    n_strings: int = 200,
    line_thickness: int = 1,
    sample_pairs: int = 1000,
    algorithm: str = "greedy",
) -> List[Dict[str, int]]:
    """
    Dispatch to whichever StringArtAlgorithm you’ve registered.
    """
    algo = ALGORITHMS.get(algorithm)
    if algo is None:
        raise ValueError(f"Unknown algorithm '{algorithm}', available: {list(ALGORITHMS)}")
    return algo.generate(
        pixels,
        n_anchors=n_anchors,
        n_strings=n_strings,
        line_thickness=line_thickness,
        sample_pairs=sample_pairs,
    )
