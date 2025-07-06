# stringart_app/image_to_vector_algorithms/base.py

from typing import List, Dict, Optional, Callable
import numpy as np
import logging

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
        sample_pairs: int,
        logger: Optional[logging.Logger] = None,
        *,
        vector_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, int]]:
        """
        Given a grayscale pixel-map, return a list of up-to-n_strings {"from": i, "to": j} pairs.

        :param pixels: Grayscale pixel map
        :param n_anchors: Number of radial anchor points
        :param n_strings: Number of string connections to make
        :param line_thickness: Thickness of each string
        :param sample_pairs: Number of candidate pairs to sample per iteration
        :param logger: Optional logger for debug/info messages
        :param vector_callback: Optional callback called as each vector is generated,
                                signature vector_callback(from_idx: int, to_idx: int)
        """
        raise NotImplementedError("Must implement generate()")
