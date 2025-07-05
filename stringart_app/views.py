# stringart_app/views.py

import time
import base64
import json
from io import BytesIO, StringIO
from pathlib import Path
import contextlib

from django.http import StreamingHttpResponse
from django.shortcuts import render
from PIL import Image
import numpy as np

from .renderer import generate_radial_anchors
from .planner import generate_string_vectors, ALGORITHMS

DEBUG = True

# In-memory queue of log lines for Server-Sent Events
LOG_QUEUE: list[str] = []

def make_logger(log_list):
    """
    Returns a _log(msg) that prints to console (if DEBUG),
    appends msg to the per-request log_list, and also
    pushes it into the global LOG_QUEUE for SSE.
    """
    def _log(msg: str):
        if DEBUG:
            # Print to your server console
            print(msg)
            # Strip trailing newlines and append to this request's logs
            line = msg.rstrip("\n")
            log_list.append(line)
            # Append to global queue (for live streaming)
            LOG_QUEUE.append(line)
    return _log

def home(request):
    # Prepare context and per-request log buffer
    context: dict = {}
    logs: list[str] = []
    _log = make_logger(logs)
    context['logs'] = logs

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
    context['test_images'] = [p.name for p in img_paths]

    # --- if POST, process them all ---
    if request.method == 'POST':
        # Clear out any previous logs so we start fresh
        LOG_QUEUE.clear()

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

            # --- capture prints during vector generation ---
            buf = StringIO()
            with contextlib.redirect_stdout(buf):
                vectors = generate_string_vectors(
                    pixels,
                    n_anchors=180,
                    n_strings=200,
                    line_thickness=1,
                    sample_pairs=1000,
                    algorithm=algo_key,
                )
            for line in buf.getvalue().splitlines():
                _log(line)

            # --- capture prints during anchor generation ---
            buf = StringIO()
            with contextlib.redirect_stdout(buf):
                anchors = generate_radial_anchors(180, *TARGET_SIZE)
            for line in buf.getvalue().splitlines():
                _log(line)

            # JSON for physics
            anchors_json = json.dumps(anchors)
            vectors_json = json.dumps(vectors)
            vectors_count = len(vectors)

            # encode grayscale PNG
            img_buf = BytesIO()
            img_small.save(img_buf, format="PNG")
            processed_b64 = base64.b64encode(img_buf.getvalue()).decode("ascii")

            test_results.append({
                "name": name,
                "processed_image": processed_b64,
                "anchors_json": anchors_json,
                "vectors_json": vectors_json,
                "vectors_count": vectors_count,
            })

        context['test_results'] = test_results

    return render(request, 'core/home.html', context)


def stream_logs(request):
    """
    Server-Sent Events endpoint that streams lines from LOG_QUEUE.
    """
    def event_stream():
        last_idx = 0
        while True:
            # Emit any new log lines
            while last_idx < len(LOG_QUEUE):
                msg = LOG_QUEUE[last_idx]
                last_idx += 1
                yield f"data: {msg}\n\n".encode('utf-8')
            time.sleep(0.2)

    return StreamingHttpResponse(event_stream(), content_type='text/event-stream')
