# stringart_app/views.py

import base64
import json
from io import BytesIO

from django.shortcuts import render
from PIL import Image
import numpy as np

from .renderer import generate_radial_anchors
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

        # Load the original image and keep a copy for display
        img = Image.open(img_file)
        orig_w, orig_h = img.size
        _log(f"Loaded image into PIL; original mode={img.mode}, size={orig_w}×{orig_h}")

        # Encode original upload for template
        buf0 = BytesIO()
        img.save(buf0, format='PNG')
        context['original_image'] = base64.b64encode(buf0.getvalue()).decode('ascii')
        _log(f"Encoded original image to base64 ({len(context['original_image'])} chars)")

        # Convert to grayscale
        img = img.convert('L')
        _log("Converted image to grayscale (L mode)")

        # Resize
        TARGET_SIZE = (200, 200)
        _log(f"Resizing image to target dimensions {TARGET_SIZE[0]}×{TARGET_SIZE[1]} using LANCZOS")
        img_small = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

        # Prepare pixel data
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

        # Prepare JSON for physics renderer
        anchors = generate_radial_anchors(180, *img_small.size)
        context['anchors_json'] = json.dumps(anchors)
        context['vectors_json'] = json.dumps(vectors)

        # Encode grayscale display
        buf = BytesIO()
        img_small.save(buf, format='PNG')
        processed_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        context['processed_image'] = processed_b64
        _log(f"Encoded processed grayscale image to base64 ({len(processed_b64)} chars)")

        _log(f"Final context keys: {list(context.keys())}")

    else:
        if request.method == 'POST':
            _log("POST received but no image file found in request.FILES; skipping processing")

    return render(request, 'core/home.html', context)
