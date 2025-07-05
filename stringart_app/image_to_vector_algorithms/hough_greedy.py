# stringart_app/image_to_vector_algorithms/hough_greedy.py

import numpy as np
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw

# you need scikit-image on your PYTHONPATH for these:
# pip install scikit-image
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line

from .base import StringArtAlgorithm
from ..renderer import generate_radial_anchors


class HoughGreedyAlgorithm(StringArtAlgorithm):
    """
    1) Use a probabilistic Hough transform to select a small candidate set of lines.
    2) Greedily pick the line that most reduces squared‐error at each step.
    """

    def generate(
        self,
        pixels: np.ndarray,
        n_anchors: int = 180,
        n_strings: int = 200,
        line_thickness: int = 1,
        sample_pairs: int = 1000  # unused here
    ) -> List[Dict[str, int]]:

        height, width = pixels.shape

        # 1) Edge‐detect
        edges: np.ndarray = canny(pixels / 255.0, sigma=2.0)

        # 2) Hough → list of ((x0,y0),(x1,y1)) segments
        lines: List[Tuple[Tuple[float,float], Tuple[float,float]]] = \
            probabilistic_hough_line(edges,
                                     threshold=10,
                                     line_length=30,
                                     line_gap=5)

        # 3) Anchor positions, as an (n_anchors × 2) array
        anchors_list = generate_radial_anchors(n_anchors, width, height)
        anchors_arr: np.ndarray = np.array(anchors_list, dtype=float)

        # 4) Map each Hough segment to the nearest anchor‐pair
        pairs: set[Tuple[int,int]] = set()
        for (pt0, pt1) in lines:
            # ensure these are 2‐vectors so types line up
            p0 = np.array(pt0, dtype=float)
            p1 = np.array(pt1, dtype=float)

            d0 = np.linalg.norm(anchors_arr - p0, axis=1)
            d1 = np.linalg.norm(anchors_arr - p1, axis=1)

            i = int(d0.argmin())
            j = int(d1.argmin())
            if i != j:
                pairs.add((min(i, j), max(i, j)))

        candidates: List[Tuple[int,int]] = list(pairs)
        if not candidates:
            return []

        # 5) Prepare a mutable “canvas” we’ll draw onto, and record vectors
        canvas = np.full_like(pixels, 255, dtype=np.int16)
        residual = (canvas - pixels).astype(np.int32)  # for error computations
        vectors: List[Dict[str,int]] = []

        for _ in range(n_strings):
            best_delta = 0
            best_pair: Tuple[int,int] | None = None

            # try each candidate
            for (i, j) in candidates:
                # draw that single line on a *copy* of canvas
                tmp_img = Image.fromarray(canvas.astype(np.uint8), mode='L')
                ImageDraw.Draw(tmp_img).line(
                    [tuple(anchors_list[i]), tuple(anchors_list[j])],
                    fill=0,
                    width=line_thickness
                )
                tmp_arr = np.array(tmp_img, dtype=np.int16)

                # how much does it reduce squared error?
                new_err = np.sum((tmp_arr - pixels.astype(np.int16))**2)
                old_err = np.sum((canvas - pixels.astype(np.int16))**2)
                delta = old_err - new_err

                if delta > best_delta:
                    best_delta = delta
                    best_pair = (i, j)

            if best_pair is None:
                # no candidate improves anything
                break

            # commit that best line to our true canvas
            i, j = best_pair
            base_img = Image.fromarray(canvas.astype(np.uint8), mode='L')
            ImageDraw.Draw(base_img).line(
                [tuple(anchors_list[i]), tuple(anchors_list[j])],
                fill=0,
                width=line_thickness
            )
            canvas = np.array(base_img, dtype=np.int16)

            vectors.append({"from": i, "to": j})

        return vectors
