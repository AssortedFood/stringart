import math
import random
from typing import List, Dict, Optional, Callable, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .base import StringArtAlgorithm
from ..renderer import generate_radial_anchors
import logging

class CrumGreedyAlgorithm(StringArtAlgorithm):
    """
    Multi-thread “fearless” greedy string-art algorithm with blur & color support.

    Key ideas:
    - Downscale input to simulate blur & speed up computation.
    - Keep an “original” and a “current” canvas; measure per-line error via only
      the pixels that line touches (Bresenham).
    - “Fearless” scoring: penalize darkening errors lightly, reward correct darkening fully.
    - Support multiple colored threads by picking (thread, next-nail) that gives
      best improvement over all.
    """

    def generate(
        self,
        pixels: np.ndarray,
        n_anchors: int = 180,
        n_strings: int = 200,
        line_thickness: int = 1,
        sample_pairs: int = 1000,
        logger: Optional[logging.Logger] = None,
        *,
        downscale: int = 4,
        thread_colors: List[Tuple[int,int,int]] = [(0,0,0)],
        vector_callback: Optional[Callable[[int,int],None]] = None,
    ) -> List[Dict[str,int]]:
        # 1) Build small “blurred” canvas
        h, w = pixels.shape
        small = (
            Image.fromarray(pixels)
            .resize((w//downscale, h//downscale), Image.Resampling.LANCZOS)
            .convert("RGB")
        )
        small_w, small_h = small.size
        orig = np.array(small, dtype=np.float32) / 255.0      # (H, W, 3)
        current = np.ones_like(orig)                         # start blank white

        # 2) Precompute anchors
        anchors = generate_radial_anchors(n_anchors, small_w, small_h)

        # 3) Enumerate all nail-pairs & precompute Bresenham masks
        all_pairs = [(i,j) for i in range(n_anchors) for j in range(i+1,n_anchors)]

        def bresenham(a, b):
            x0,y0 = map(int,a); x1,y1 = map(int,b)
            dx, dy = abs(x1-x0), abs(y1-y0)
            sx, sy = (1 if x0<x1 else -1), (1 if y0<y1 else -1)
            err = dx - dy
            pts = []
            while True:
                pts.append((x0,y0))
                if x0==x1 and y0==y1: break
                e2 = 2*err
                if e2 > -dy:
                    err -= dy; x0 += sx
                if e2 < dx:
                    err += dx; y0 += sy
            return pts

        line_masks = [bresenham(anchors[i], anchors[j]) for i,j in all_pairs]

        # 4) Thread state
        class ThreadState:
            def __init__(self, start_nail:int, color:Tuple[int,int,int]):
                self.current_nail = start_nail
                self.color = np.array(color, dtype=np.float32)/255.0
                self.prev = set()   # forbid immediate repeats
            def best_move(self):
                best = None
                best_score = math.inf
                best_line = None
                for idx,(i,j) in enumerate(all_pairs):
                    if i!=self.current_nail and j!=self.current_nail: continue
                    if (self.current_nail, idx) in self.prev: continue
                    mask = line_masks[idx]
                    # compute “fearless” diff: negative improvements count fully,
                    # positive errors are damped (here by 1/5)
                    tot = 0.0
                    for x,y in mask:
                        orig_px = orig[y,x]
                        cur_px  = current[y,x]
                        # blended = alpha*thread_color + (1-alpha)*cur_px
                        alpha = 1.0/downscale
                        new_px = alpha*self.color + (1-alpha)*cur_px
                        delta = np.linalg.norm(orig_px - new_px)**2 - np.linalg.norm(orig_px - cur_px)**2
                        if delta<0:
                            tot += delta
                        else:
                            tot += delta*0.2
                    if tot < best_score:
                        best_score,best=tot,idx
                return best, best_score

        threads = [ThreadState(0, c) for c in thread_colors]
        vectors: List[Dict[str,int]] = []

        # 5) Main loop
        for _ in range(n_strings):
            # 5a) pick best thread + move
            choice = None
            best_val = math.inf
            for t in threads:
                idx,score = t.best_move()
                if idx is not None and score<best_val:
                    choice = (t, idx, score)
                    best_val = score

            if choice is None:
                break

            t, idx, _ = choice
            i,j = all_pairs[idx]
            # record
            vectors.append({"from":i,"to":j})
            if vector_callback:
                vector_callback(i,j)

            # 5b) draw into current
            mask = line_masks[idx]
            alpha = 1.0/downscale
            for x,y in mask:
                current[y,x] = alpha*t.color + (1-alpha)*current[y,x]

            # 5c) forbid immediate repeat & update nail
            t.prev.add((t.current_nail, idx))
            t.current_nail = j if t.current_nail==i else i

        if logger:
            logger.debug(f"[crum-greedy] placed {len(vectors)} strings")
        return vectors
