# stringart_app/renderer.py

import math
from PIL import Image, ImageDraw

def generate_radial_anchors(n_anchors, width, height, margin=10):
    cx, cy = width / 2, height / 2
    radius = min(cx, cy) - margin
    return [
        (cx + radius * math.cos(2 * math.pi * i / n_anchors),
         cy + radius * math.sin(2 * math.pi * i / n_anchors))
        for i in range(n_anchors)
    ]

def render_vector_list(vectors, size, n_anchors=180, line_width=1):
    """As before: blank-canvas preview."""
    img = Image.new('L', size, color=255)
    draw = ImageDraw.Draw(img)
    anchors = generate_radial_anchors(n_anchors, *size)
    for v in vectors:
        draw.line([anchors[v['from']], anchors[v['to']]], fill=0, width=line_width)
    return img

def render_overlay(vectors, base_image, n_anchors=180, line_width=1, line_colour=(255,0,0)):
    """
    Draws the string-art vectors overlaid on the `base_image` (a PIL RGB image),
    using a translucent line colour so you can see the underlying greyscale.
    """
    overlay = base_image.convert('RGBA')
    draw = ImageDraw.Draw(overlay, 'RGBA')
    w, h = base_image.size
    anchors = generate_radial_anchors(n_anchors, w, h)
    for v in vectors:
        draw.line(
            [anchors[v['from']], anchors[v['to']]],
            fill=line_colour + (128,),  # half-opacity red
            width=line_width
        )
    return overlay
