# stringart_app/renderer.py

import math
from PIL import Image, ImageDraw

DEBUG = True

def _log(msg: str):
    if DEBUG:
        print(msg)

def generate_radial_anchors(n_anchors, width, height, margin=10):
    _log(f"generate_radial_anchors called with n_anchors={n_anchors}, width={width}, height={height}, margin={margin}")
    cx, cy = width / 2, height / 2
    radius = min(cx, cy) - margin
    anchors = [
        (cx + radius * math.cos(2 * math.pi * i / n_anchors),
         cy + radius * math.sin(2 * math.pi * i / n_anchors))
        for i in range(n_anchors)
    ]
    _log(f"Generated {len(anchors)} anchors")
    return anchors

def render_vector_list(vectors, size, n_anchors=180, line_width=1):
    """As before: blank-canvas preview."""
    _log(f"render_vector_list called with {len(vectors)} vectors, size={size}, n_anchors={n_anchors}, line_width={line_width}")
    img = Image.new('L', size, color=255)
    draw = ImageDraw.Draw(img)
    anchors = generate_radial_anchors(n_anchors, *size)
    for idx, v in enumerate(vectors, start=1):
        draw.line([anchors[v['from']], anchors[v['to']]], fill=0, width=line_width)
        if idx % 50 == 0:
            _log(f"  Drew {idx}/{len(vectors)} lines")
    _log("Completed render_vector_list")
    return img

def render_overlay(vectors, base_image, n_anchors=180, line_width=1, line_colour=(255,0,0)):
    """
    Draws the string-art vectors overlaid on the `base_image` (a PIL RGB image),
    using a translucent line colour so you can see the underlying greyscale.
    """
    _log(f"render_overlay called with {len(vectors)} vectors, image size={base_image.size}, "
         f"n_anchors={n_anchors}, line_width={line_width}, line_colour={line_colour}")
    overlay = base_image.convert('RGBA')
    draw = ImageDraw.Draw(overlay, 'RGBA')
    w, h = base_image.size
    anchors = generate_radial_anchors(n_anchors, w, h)
    for idx, v in enumerate(vectors, start=1):
        draw.line(
            [anchors[v['from']], anchors[v['to']]],
            fill=line_colour + (128,),  # half-opacity
            width=line_width
        )
        if idx % 50 == 0:
            _log(f"  Overlayed {idx}/{len(vectors)} lines")
    _log("Completed render_overlay")
    return overlay
