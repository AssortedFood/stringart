# core/views.py

import base64
from io import BytesIO

from django.shortcuts import render
from PIL import Image
import numpy as np

from .renderer import render_vector_list, render_overlay
from .planner import generate_string_vectors

def home(request):
    context = {}

    if request.method == 'POST' and request.FILES.get('image'):
        # Load & downscale
        img = Image.open(request.FILES['image']).convert('L')
        TARGET_SIZE = (200, 200)
        img_small = img.resize(TARGET_SIZE, Image.LANCZOS)
        pixels = np.array(img_small)

        # Generate vectors
        vectors = generate_string_vectors(
            pixels,
            n_anchors=180,
            n_strings=200,
            line_thickness=1,
            sample_pairs=1000
        )
        context['vectors'] = vectors

        # Grayscale display
        buf = BytesIO()
        img_small.save(buf, format='PNG')
        context['processed_image'] = base64.b64encode(buf.getvalue()).decode('ascii')

        # Line-only preview
        preview_img = render_vector_list(vectors, img_small.size, n_anchors=180, line_width=1)
        buf2 = BytesIO()
        preview_img.save(buf2, format='PNG')
        context['vector_preview'] = base64.b64encode(buf2.getvalue()).decode('ascii')

        # Overlay preview (RGB)
        base_rgb = img_small.convert('RGB')
        overlay_img = render_overlay(vectors, base_rgb, n_anchors=180, line_width=1)
        buf3 = BytesIO()
        overlay_img.save(buf3, format='PNG')
        context['overlay_preview'] = base64.b64encode(buf3.getvalue()).decode('ascii')

    return render(request, 'core/home.html', context)
