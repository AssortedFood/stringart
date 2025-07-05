# stringart_app/views.py

import base64
import json
from io import BytesIO
from pathlib import Path

from django.shortcuts import render
from PIL import Image
import numpy as np

from .renderer import generate_radial_anchors
from .planner import generate_string_vectors, ALGORITHMS

DEBUG = True

def _log(msg: str):
    if DEBUG:
        print(msg)

def home(request):
    context = {}

    # --- algorithm controls ---
    algo_key = request.POST.get('algorithm', request.GET.get('algorithm', 'greedy'))
    if algo_key not in ALGORITHMS:
        algo_key = 'greedy'
    context['algorithms'] = list(ALGORITHMS.keys())
    context['selected_algorithm'] = algo_key

    # --- discover test images ---
    TEST_DIR = Path(__file__).parent / "static" / "test_images"
    _log(f"Looking for test images in {TEST_DIR}")
    img_paths = sorted(TEST_DIR.glob("*.[jp][pn]g"))
    # just names for static/<test_images>/<name>
    context['test_images'] = [p.name for p in img_paths]

    # --- if POST, process them all ---
    if request.method == 'POST':
        _log(f"Batch-processing {len(img_paths)} images with '{algo_key}'")
        test_results = []

        for p in img_paths:
            name = p.stem
            _log(f"Processing {p.name}")

            # load + grayscale + resize
            img = Image.open(p).convert("L")
            TARGET_SIZE = (200, 200)
            img_small = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            pixels = np.array(img_small)

            # generate vectors
            vectors = generate_string_vectors(
                pixels,
                n_anchors=180,
                n_strings=200,
                line_thickness=1,
                sample_pairs=1000,
                algorithm=algo_key,
            )

            # anchors + JSON for physics
            anchors = generate_radial_anchors(180, *TARGET_SIZE)
            anchors_json = json.dumps(anchors)
            vectors_json = json.dumps(vectors)

            # encode grayscale PNG
            buf = BytesIO()
            img_small.save(buf, format="PNG")
            processed_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            test_results.append({
                "name": name,
                "processed_image": processed_b64,
                "anchors_json": anchors_json,
                "vectors_json": vectors_json,
            })

        context['test_results'] = test_results

    return render(request, 'core/home.html', context)
