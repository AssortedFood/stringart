# stringart_app/views.py

import base64
from io import BytesIO

from django.shortcuts import render
from PIL import Image
import numpy as np

from .renderer import render_vector_list, render_overlay
from .planner import generate_string_vectors

DEBUG = True

def _log(msg: str):
    if DEBUG:
        print(msg)

def home(request):
    context = {}

    _log(f"home() called with method={request.method}")

    if request.method == 'POST' and request.FILES.get('image'):
        img_file = request.FILES['image']
        _log(f"Received POST / with image upload: {img_file.name}")

        # Load & downscale
        img = Image.open(img_file)
        orig_w, orig_h = img.size
        _log(f"Loaded image into PIL; original mode={img.mode}, size={orig_w}×{orig_h}")
        img = img.convert('L')
        _log("Converted image to grayscale (L mode)")

        TARGET_SIZE = (200, 200)
        _log(f"Resizing image to target dimensions {TARGET_SIZE[0]}×{TARGET_SIZE[1]} using LANCZOS")
        img_small = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

        pixels = np.array(img_small)
        _log(f"Converted resized image to NumPy array with shape {pixels.shape} and dtype {pixels.dtype}")

        # Generate vectors
        _log("Starting generate_string_vectors (n_anchors=180, n_strings=200, sample_pairs=1000)")
        vectors = generate_string_vectors(
            pixels,
            n_anchors=180,
            n_strings=200,
            line_thickness=1,
            sample_pairs=1000
        )
        _log(f"generate_string_vectors returned {len(vectors)} vectors")
        context['vectors'] = vectors

        # Grayscale display
        buf = BytesIO()
        img_small.save(buf, format='PNG')
        processed_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        context['processed_image'] = processed_b64
        _log(f"Encoded processed grayscale image to base64 ({len(processed_b64)} chars)")

        # Line-only preview
        _log(f"Rendering vector-only preview with {len(vectors)} lines")
        preview_img = render_vector_list(vectors, img_small.size, n_anchors=180, line_width=1)
        buf2 = BytesIO()
        preview_img.save(buf2, format='PNG')
        vector_b64 = base64.b64encode(buf2.getvalue()).decode('ascii')
        context['vector_preview'] = vector_b64
        _log(f"Encoded vector preview to base64 ({len(vector_b64)} chars)")

        # Overlay preview (RGB)
        _log("Rendering overlay preview on RGB base image")
        base_rgb = img_small.convert('RGB')
        overlay_img = render_overlay(vectors, base_rgb, n_anchors=180, line_width=1)
        buf3 = BytesIO()
        overlay_img.save(buf3, format='PNG')
        overlay_b64 = base64.b64encode(buf3.getvalue()).decode('ascii')
        context['overlay_preview'] = overlay_b64
        _log(f"Encoded overlay preview to base64 ({len(overlay_b64)} chars)")

        _log(f"Final context keys: {list(context.keys())}")

    else:
        if request.method == 'POST':
            _log("POST received but no image file found in request.FILES; skipping processing")

    return render(request, 'core/home.html', context)
