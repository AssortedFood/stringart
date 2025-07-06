# stringart_app/views.py

import time
import base64
import json
import threading
import uuid
from io import BytesIO, StringIO
from pathlib import Path
import contextlib

from django.http import StreamingHttpResponse, JsonResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST
from PIL import Image
import numpy as np

from .renderer import generate_radial_anchors
from .planner import generate_string_vectors, ALGORITHMS
from .preprocessing import load_image_to_pixels  # ← your loader

DEBUG = True

# Per‐job registries
JOB_CANCEL_EVENTS: dict[str, threading.Event] = {}
JOB_LOGS: dict[str, list[str]] = {}
JOB_RESULTS: dict[str, list[dict]] = {}


def make_logger(job_id: str):
    """
    Returns a _log(msg) that prints to console (if DEBUG),
    appends msg to the per‐job log list, and also pushes it
    into the global JOB_LOGS registry for SSE.
    """
    def _log(msg: str):
        if DEBUG:
            print(msg)
        line = msg.rstrip("\n")
        JOB_LOGS[job_id].append(line)
    return _log


def home(request):
    context: dict = {}
    # normal GET: show form
    if request.method == 'GET':
        context['algorithms'] = list(ALGORITHMS.keys())
        context['selected_algorithms'] = list(ALGORITHMS.keys())
        return render(request, 'core/home.html', context)

    # handle image upload preview
    if request.method == 'POST' and request.FILES.getlist('images') and not request.POST.get('run_algos'):
        selected_algorithms = request.POST.getlist('algorithms') or list(ALGORITHMS.keys())
        selected_algorithms = [a for a in selected_algorithms if a in ALGORITHMS] or list(ALGORITHMS.keys())
        context['algorithms'] = list(ALGORITHMS.keys())
        context['selected_algorithms'] = selected_algorithms

        uploaded = []
        for f in request.FILES.getlist('images'):
            name = f.name
            data = f.read()
            img = Image.open(BytesIO(data)).convert('RGB')
            img.thumbnail((200, 200), Image.Resampling.LANCZOS)
            buf = BytesIO()
            img.save(buf, format='PNG')
            b64 = base64.b64encode(buf.getvalue()).decode('ascii')
            uploaded.append({'name': name, 'data': b64})
        context['uploaded_images'] = uploaded
        return render(request, 'core/home.html', context)

    # handle "Run Algorithms" AJAX kickoff
    if request.method == 'POST' and request.POST.get('run_algos'):
        # create a new job
        job_id = str(uuid.uuid4())
        cancel_ev = threading.Event()
        JOB_CANCEL_EVENTS[job_id] = cancel_ev
        JOB_LOGS[job_id] = []
        JOB_RESULTS[job_id] = []

        # reconstruct image bytes
        names = request.POST.getlist('image_name')
        datas = request.POST.getlist('image_data')
        files = {name: base64.b64decode(data) for name, data in zip(names, datas)}

        TARGET_SIZE = (200, 200)
        levels = int(request.POST.get('levels', 8))
        selected_algorithms = request.POST.getlist('algorithms') or list(ALGORITHMS.keys())
        selected_algorithms = [a for a in selected_algorithms if a in ALGORITHMS] or list(ALGORITHMS.keys())

        def worker():
            _log = make_logger(job_id)

            # Phase 1: grayscale pass
            _log(f"=== Phase 1: grayscale-only for {len(files)} images ===")
            for name, data in files.items():
                if cancel_ev.is_set():
                    _log("Job cancelled during grayscale phase.")
                    return
                stem = Path(name).stem
                _log(f"[grayscale] Loading {name}")
                pixels = load_image_to_pixels(
                    path=BytesIO(data),
                    size=TARGET_SIZE,
                    levels=levels,
                    gamma=0.8,
                    autocontrast=True
                )
                img_small = Image.fromarray(pixels, mode='L')
                buf_img = BytesIO()
                img_small.save(buf_img, format="PNG")
                b64 = base64.b64encode(buf_img.getvalue()).decode("ascii")

                JOB_RESULTS[job_id].append({
                    "phase": "grayscale",
                    "algorithm": None,
                    "name": stem,
                    "processed_image": b64,
                })

            # Phase 2: run each selected algorithm
            for algo_key in selected_algorithms:
                _log(f"=== Phase 2: algorithm '{algo_key}' ===")
                for name, data in files.items():
                    if cancel_ev.is_set():
                        _log("Job cancelled during algorithm phase.")
                        return
                    stem = Path(name).stem
                    _log(f"[{algo_key}] Loading & processing {name}")
                    pixels = load_image_to_pixels(
                        path=BytesIO(data),
                        size=TARGET_SIZE,
                        levels=levels,
                        gamma=0.8,
                        autocontrast=True
                    )
                    buf = StringIO()
                    with contextlib.redirect_stdout(buf):
                        vectors = generate_string_vectors(
                            pixels,
                            n_anchors=180,
                            n_strings=300,
                            line_thickness=1,
                            sample_pairs=1000,
                            algorithm=algo_key,
                        )
                    for line in buf.getvalue().splitlines():
                        _log(line)

                    buf2 = StringIO()
                    with contextlib.redirect_stdout(buf2):
                        anchors = generate_radial_anchors(180, *TARGET_SIZE)
                    for line in buf2.getvalue().splitlines():
                        _log(line)

                    JOB_RESULTS[job_id].append({
                        "phase": "algorithm",
                        "algorithm": algo_key,
                        "name": stem,
                        "anchors_json": json.dumps(anchors),
                        "vectors_json": json.dumps(vectors),
                        "vectors_count": len(vectors),
                    })

            _log("Job complete.")

        # start background thread
        threading.Thread(target=worker, daemon=True).start()

        # respond immediately with job_id
        return JsonResponse({"job_id": job_id})

    # fallback
    return render(request, 'core/home.html', context)


@require_GET
def stream_logs(request):
    """
    SSE endpoint streaming new lines from JOB_LOGS[job_id].
    """
    job_id = request.GET.get('job_id')
    if job_id not in JOB_LOGS:
        return HttpResponse(status=404)

    def event_stream():
        last_idx = 0
        cancel_ev = JOB_CANCEL_EVENTS.get(job_id)
        while not (cancel_ev and cancel_ev.is_set()):
            logs = JOB_LOGS[job_id]
            while last_idx < len(logs):
                msg = logs[last_idx]
                last_idx += 1
                yield f"data: {msg}\n\n".encode('utf-8')
            time.sleep(0.2)
        # cleanup
        JOB_LOGS.pop(job_id, None)
        JOB_CANCEL_EVENTS.pop(job_id, None)
        JOB_RESULTS.pop(job_id, None)

    return StreamingHttpResponse(event_stream(), content_type='text/event-stream')


@require_GET
def stream_results(request):
    """
    SSE endpoint streaming items from JOB_RESULTS[job_id] as JSON.
    """
    job_id = request.GET.get('job_id')
    if job_id not in JOB_RESULTS:
        return HttpResponse(status=404)

    def event_stream():
        last_idx = 0
        cancel_ev = JOB_CANCEL_EVENTS.get(job_id)
        while not (cancel_ev and cancel_ev.is_set()):
            results = JOB_RESULTS[job_id]
            while last_idx < len(results):
                payload = results[last_idx]
                last_idx += 1
                yield f"data: {json.dumps(payload)}\n\n".encode('utf-8')
            time.sleep(0.2)
        # cleanup handled in stream_logs
    return StreamingHttpResponse(event_stream(), content_type='text/event-stream')


@require_POST
def stop_job(request, job_id):
    """
    Endpoint to cancel a running job.
    """
    ev = JOB_CANCEL_EVENTS.get(job_id)
    if ev:
        ev.set()
        return HttpResponse(status=204)
    return HttpResponse(status=404)
