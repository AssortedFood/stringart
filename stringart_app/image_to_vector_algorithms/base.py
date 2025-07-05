# stringart_app/image_to_vector_algorithms/base.py
from typing import List, Dict
import numpy as np

class StringArtAlgorithm:
    """
    Interface for any image→vector algorithm.
    """
    def generate(
        self,
        pixels: np.ndarray,
        n_anchors: int,
        n_strings: int,
        line_thickness: int,
        sample_pairs: int
    ) -> List[Dict[str, int]]:
        """
        Given a grayscale pixel-map, return a list of up-to- n_strings {"from": i, "to": j} pairs.
        """
        raise NotImplementedError("Must implement generate()")
