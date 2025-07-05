# stringart_app/planner.py

from typing import List, Dict
import numpy as np

# pull in your registry
from .image_to_vector_algorithms import ALGORITHMS

def generate_string_vectors(
    pixels: np.ndarray,
    n_anchors: int = 180,
    n_strings: int = 200,
    line_thickness: int = 1,
    sample_pairs: int = 1000,
    algorithm: str = "greedy",
) -> List[Dict[str, int]]:
    """
    Dispatch to whichever StringArtAlgorithm you've registered.

    :param pixels: grayscale pixel array
    :param n_anchors: how many nails around the circle
    :param n_strings: how many lines to draw
    :param line_thickness: thickness of each line
    :param sample_pairs: how many candidate pairs to sample per iteration
    :param algorithm: key into ALGORITHMS registry
    :returns: list of {"from": i, "to": j} dicts
    """
    algo = ALGORITHMS.get(algorithm)
    if algo is None:
        valid = ", ".join(ALGORITHMS.keys())
        raise ValueError(f"Unknown algorithm '{algorithm}'. Valid options are: {valid}")
    # delegate to the selected strategy
    return algo.generate(
        pixels,
        n_anchors=n_anchors,
        n_strings=n_strings,
        line_thickness=line_thickness,
        sample_pairs=sample_pairs,
    )
