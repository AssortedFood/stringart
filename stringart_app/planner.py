# stringart_app/planner.py

from typing import List, Dict, Optional, Callable
import numpy as np
import logging

# pull in your registry
from .image_to_vector_algorithms import ALGORITHMS

def generate_string_vectors(
    pixels: np.ndarray,
    n_anchors: int = 180,
    n_strings: int = 200,
    line_thickness: int = 1,
    sample_pairs: int = 1000,
    algorithm: str = "greedy",
    logger: Optional[logging.Logger] = None,
    *,
    vector_callback: Optional[Callable[[int, int], None]] = None
) -> List[Dict[str, int]]:
    """
    Dispatch to whichever StringArtAlgorithm you've registered, passing along
    an optional logger for SSE streaming and an optional vector_callback
    to receive each vector as it's generated.

    :param pixels: grayscale pixel array
    :param n_anchors: how many nails around the circle
    :param n_strings: how many lines to draw
    :param line_thickness: thickness of each line
    :param sample_pairs: how many candidate pairs to sample per iteration
    :param algorithm: key into ALGORITHMS registry
    :param logger: optional Logger to receive debug/info messages
    :param vector_callback: optional callable that will be called for each
                            generated vector as vector_callback(from_idx, to_idx)
    :returns: list of {"from": i, "to": j} dicts
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    algo = ALGORITHMS.get(algorithm)
    if algo is None:
        valid = ", ".join(ALGORITHMS.keys())
        logger.error(f"Unknown algorithm '{algorithm}'. Valid options: {valid}")
        raise ValueError(f"Unknown algorithm '{algorithm}'. Valid options: {valid}")

    # delegate to the selected strategy, providing the logger and callback
    return algo.generate(
        pixels,
        n_anchors=n_anchors,
        n_strings=n_strings,
        line_thickness=line_thickness,
        sample_pairs=sample_pairs,
        logger=logger,
        vector_callback=vector_callback,
    )
