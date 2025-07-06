# stringart_app/renderer.py

import math
import logging
from typing import Optional
from PIL import Image, ImageDraw


def generate_radial_anchors(
    n_anchors: int,
    width: int,
    height: int,
    margin: int = 10,
    logger: Optional[logging.Logger] = None
) -> list[tuple[float, float]]:
    """
    Compute evenly spaced points around a circle (the "anchors").
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.debug(
        f"generate_radial_anchors called with "
        f"n_anchors={n_anchors}, width={width}, height={height}, margin={margin}"
    )

    cx, cy = width / 2, height / 2
    radius = min(cx, cy) - margin
    anchors = [
        (
            cx + radius * math.cos(2 * math.pi * i / n_anchors),
            cy + radius * math.sin(2 * math.pi * i / n_anchors),
        )
        for i in range(n_anchors)
    ]

    logger.debug(f"Generated {len(anchors)} anchors")
    return anchors


def render_vector_list(
    vectors: list[dict[str, int]],
    size: tuple[int, int],
    n_anchors: int = 180,
    line_width: int = 1,
    logger: Optional[logging.Logger] = None
) -> Image.Image:
    """
    Render a blank-canvas preview of the string-art given the vector list.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.debug(
        f"render_vector_list called with "
        f"{len(vectors)} vectors, size={size}, "
        f"n_anchors={n_anchors}, line_width={line_width}"
    )

    img = Image.new('L', size, color=255)
    draw = ImageDraw.Draw(img)
    anchors = generate_radial_anchors(n_anchors, size[0], size[1], logger=logger)

    for idx, v in enumerate(vectors, start=1):
        draw.line(
            [anchors[v['from']], anchors[v['to']]],
            fill=0,
            width=line_width
        )
        if idx % 50 == 0:
            logger.debug(f"Drew {idx}/{len(vectors)} lines")

    logger.debug("Completed render_vector_list")
    return img


def render_overlay(
    vectors: list[dict[str, int]],
    base_image: Image.Image,
    n_anchors: int = 180,
    line_width: int = 1,
    line_colour: tuple[int, int, int] = (255, 0, 0),
    logger: Optional[logging.Logger] = None
) -> Image.Image:
    """
    Draws the string-art vectors overlaid on the `base_image` (an RGB PIL image),
    using a translucent line colour so you can see the underlying greyscale.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.debug(
        f"render_overlay called with "
        f"{len(vectors)} vectors, image size={base_image.size}, "
        f"n_anchors={n_anchors}, line_width={line_width}, "
        f"line_colour={line_colour}"
    )

    overlay = base_image.convert('RGBA')
    draw = ImageDraw.Draw(overlay, 'RGBA')
    w, h = base_image.size
    anchors = generate_radial_anchors(n_anchors, w, h, logger=logger)

    for idx, v in enumerate(vectors, start=1):
        draw.line(
            [anchors[v['from']], anchors[v['to']]],
            fill=line_colour + (128,),  # half-opacity
            width=line_width
        )
        if idx % 50 == 0:
            logger.debug(f"Overlayed {idx}/{len(vectors)} lines")

    logger.debug("Completed render_overlay")
    return overlay
