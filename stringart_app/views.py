# stringart_app/views.py
#
# Views for the stringart web app:
# - Handles image uploads, previews, and job execution
# - Streams logs and results to the frontend using Server-Sent Events (SSE)
# - Streams each string-art vector as it's generated
# - Manages per-job state (cancellation, logs, results)
#

import time
import base64
import json
import threading
import uuid
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from django.http import StreamingHttpResponse, JsonResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST
from PIL import Image

import logging

from .planner import generate_string_vectors, ALGORITHMS
from .preprocessing import load_image_to_pixels
from .sse_logging import create_sse_logger

# === Per-job registries ===
JOB_CANCEL_EVENTS: dict[str, threading.Event] = {}
JOB_LOGS: dict[str, list[str]] = {}
JOB_RESULTS: dict[str, list[dict]] = {}


def home(request):
    if request.method == 'GET':
        return render(request, 'core/home.html', {
            'algorithms': list(ALGORITHMS.keys()),
            'selected_algorithms': list(ALGORITHMS.keys()),
        })

    # Preview upload
    if request.method == 'POST' and request.FILES.getlist('images') and not request.POST.get('run_algos'):
        selected = request.POST.getlist('algorithms') or list(ALGORITHMS.keys())
        selected = [a for a in selected if a in ALGORITHMS] or list(ALGORITHMS.keys())
        uploaded = []
        for f in request.FILES.getlist('images'):
            data = f.read()
            img = Image.open(BytesIO(data))

            # Composite transparency over white
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                img = img.convert("RGBA")
                bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(bg, img)

            img = img.convert("RGB")
            img.thumbnail((200, 200), Image.Resampling.LANCZOS)

            buf = BytesIO()
            img.save(buf, format='PNG')
            uploaded.append({
                'name': f.name,
                'data': base64.b64encode(buf.getvalue()).decode('ascii')
            })

        return render(request, 'core/home.html', {
            'algorithms': list(ALGORITHMS.keys()),
            'selected_algorithms': selected,
            'uploaded_images': uploaded
        })

    # Kickoff job
    if request.method == 'POST' and request.POST.get('run_algos'):
        job_id = str(uuid.uuid4())
        cancel_ev = threading.Event()
        JOB_CANCEL_EVENTS[job_id] = cancel_ev
        JOB_LOGS[job_id] = []
        JOB_RESULTS[job_id] = []

        logger = create_sse_logger(job_id, JOB_LOGS)

        names = request.POST.getlist('image_name')
        datas = request.POST.getlist('image_data')
        files = {n: base64.b64decode(d) for n, d in zip(names, datas)}

        TARGET_SIZE = (200, 200)
        levels = int(request.POST.get('levels', 8))
        algos = request.POST.getlist('algorithms') or list(ALGORITHMS.keys())
        algos = [a for a in algos if a in ALGORITHMS] or list(ALGORITHMS.keys())
        n_anchors = int(request.POST.get('n_anchors', 180))
        n_strings = int(request.POST.get('n_strings', 200))

        def worker():
            # Phase 1: grayscale-only
            logger.info(f"=== Phase 1: grayscale-only for {len(files)} images ===")
            for name, data in files.items():
                if cancel_ev.is_set():
                    logger.info("Job cancelled.")
                    return
                stem = Path(name).stem
                logger.info(f"[grayscale] {name}")
                stream: BinaryIO = BytesIO(data)
                pixels = load_image_to_pixels(
                    path=stream,
                    size=TARGET_SIZE,
                    levels=levels,
                    gamma=0.8,
                    autocontrast=True
                )
                buf = BytesIO()
                Image.fromarray(pixels, 'L').save(buf, 'PNG')
                JOB_RESULTS[job_id].append({
                    "phase": "grayscale",
                    "algorithm": None,
                    "name": stem,
                    "processed_image": base64.b64encode(buf.getvalue()).decode('ascii'),
                })

            # Phase 2: string-art algorithms, streaming each vector
            for algo in algos:
                logger.info(f"=== Phase 2: {algo} ===")
                for name, data in files.items():
                    if cancel_ev.is_set():
                        logger.info("Job cancelled.")
                        return
                    stem = Path(name).stem
                    logger.info(f"[{algo}] {name}")
                    stream: BinaryIO = BytesIO(data)
                    pixels = load_image_to_pixels(
                        path=stream,
                        size=TARGET_SIZE,
                        levels=levels,
                        gamma=0.8,
                        autocontrast=True
                    )

                    # Callback streams one vector at a time, including node count
                    def on_vector(frm: int, to: int):
                        JOB_RESULTS[job_id].append({
                            "phase": "algorithm",
                            "algorithm": algo,
                            "name": stem,
                            "node_count": n_anchors,
                            "vector": {"from": frm, "to": to},
                        })

                    generate_string_vectors(
                        pixels,
                        n_anchors=n_anchors,
                        n_strings=n_strings,
                        line_thickness=1,
                        sample_pairs=1000,
                        algorithm=algo,
                        logger=logger,
                        vector_callback=on_vector
                    )

            logger.info("Job complete.")

        threading.Thread(target=worker, daemon=True).start()
        return JsonResponse({"job_id": job_id})

    return render(request, 'core/home.html', {})


@require_GET
def stream_logs(request):
    job_id = request.GET.get('job_id')
    if job_id not in JOB_LOGS:
        return HttpResponse(status=404)

    def event_stream():
        idx = 0
        cancel_ev = JOB_CANCEL_EVENTS.get(job_id)
        while not (cancel_ev and cancel_ev.is_set()):
            logs = JOB_LOGS[job_id]
            while idx < len(logs):
                yield f"data: {logs[idx]}\n\n".encode()
                idx += 1
            time.sleep(0.2)
        JOB_LOGS.pop(job_id, None)
        JOB_CANCEL_EVENTS.pop(job_id, None)
        JOB_RESULTS.pop(job_id, None)

    return StreamingHttpResponse(event_stream(), content_type='text/event-stream')


@require_GET
def stream_results(request):
    job_id = request.GET.get('job_id')
    if job_id not in JOB_RESULTS:
        return HttpResponse(status=404)

    def event_stream():
        idx = 0
        cancel_ev = JOB_CANCEL_EVENTS.get(job_id)
        while not (cancel_ev and cancel_ev.is_set()):
            results = JOB_RESULTS[job_id]
            while idx < len(results):
                yield f"data: {json.dumps(results[idx])}\n\n".encode()
                idx += 1
            time.sleep(0.2)

    return StreamingHttpResponse(event_stream(), content_type='text/event-stream')


@require_POST
def stop_job(request, job_id):
    ev = JOB_CANCEL_EVENTS.get(job_id)
    if ev:
        ev.set()
        return HttpResponse(status=204)
    return HttpResponse(status=404)
