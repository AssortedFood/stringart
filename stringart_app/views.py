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

from .renderer import generate_radial_anchors
from .planner import generate_string_vectors, ALGORITHMS
from .preprocessing import load_image_to_pixels

DEBUG = True

# Per-job registries
JOB_CANCEL_EVENTS: dict[str, threading.Event] = {}
JOB_LOGS: dict[str, list[str]] = {}
JOB_RESULTS: dict[str, list[dict]] = {}


class SSELogWriter(StringIO):
    """
    A StringIO subclass that discards its own buffer and
    instead pushes every write immediately into JOB_LOGS[job_id].
    """

    def __init__(self, job_id: str):
        super().__init__()
        self.job_id = job_id

    def write(self, s: str) -> int:
        # Push each line into the per-job log list
        for line in s.rstrip("\n").split("\n"):
            JOB_LOGS[self.job_id].append(line)
        # Discard the internal buffer by not calling super().write
        return len(s)

    def getvalue(self):
        # Override to avoid returning anything
        return ""


def make_logger(job_id: str):
    def _log(msg: str):
        if DEBUG:
            print(msg)
        JOB_LOGS[job_id].append(msg.rstrip("\n"))
    return _log


def home(request):
    if request.method == 'GET':
        return render(request, 'core/home.html', {
            'algorithms': list(ALGORITHMS.keys()),
            'selected_algorithms': list(ALGORITHMS.keys()),
        })

    # preview upload
    if request.method == 'POST' and request.FILES.getlist('images') and not request.POST.get('run_algos'):
        selected = request.POST.getlist('algorithms') or list(ALGORITHMS.keys())
        selected = [a for a in selected if a in ALGORITHMS] or list(ALGORITHMS.keys())
        uploaded = []
        for f in request.FILES.getlist('images'):
            data = f.read()
            img = Image.open(BytesIO(data)).convert('RGB')
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

    # kickoff job
    if request.method == 'POST' and request.POST.get('run_algos'):
        job_id = str(uuid.uuid4())
        cancel_ev = threading.Event()
        JOB_CANCEL_EVENTS[job_id] = cancel_ev
        JOB_LOGS[job_id] = []
        JOB_RESULTS[job_id] = []

        names = request.POST.getlist('image_name')
        datas = request.POST.getlist('image_data')
        files = {n: base64.b64decode(d) for n, d in zip(names, datas)}

        TARGET_SIZE = (200, 200)
        levels = int(request.POST.get('levels', 8))
        algos = request.POST.getlist('algorithms') or list(ALGORITHMS.keys())
        algos = [a for a in algos if a in ALGORITHMS] or list(ALGORITHMS.keys())
        n_anchors = int(request.POST.get('n_anchors', 180))
        n_strings = int(request.POST.get('n_strings', 300))

        def worker():
            _log = make_logger(job_id)
            sse_writer = SSELogWriter(job_id)

            # Phase 1
            _log(f"=== Phase 1: grayscale-only for {len(files)} images ===")
            for name, data in files.items():
                if cancel_ev.is_set():
                    _log("Job cancelled.")
                    return
                stem = Path(name).stem
                _log(f"[grayscale] {name}")
                pixels = load_image_to_pixels(
                    path=BytesIO(data),
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

            # Phase 2
            for algo in algos:
                _log(f"=== Phase 2: {algo} ===")
                for name, data in files.items():
                    if cancel_ev.is_set():
                        _log("Job cancelled.")
                        return
                    stem = Path(name).stem
                    _log(f"[{algo}] {name}")
                    pixels = load_image_to_pixels(
                        path=BytesIO(data),
                        size=TARGET_SIZE,
                        levels=levels,
                        gamma=0.8,
                        autocontrast=True
                    )
                    # live‐stream prints from the algorithm
                    with contextlib.redirect_stdout(sse_writer), \
                         contextlib.redirect_stderr(sse_writer):
                        vectors = generate_string_vectors(
                            pixels,
                            n_anchors=n_anchors,
                            n_strings=n_strings,
                            line_thickness=1,
                            sample_pairs=1000,
                            algorithm=algo,
                        )
                    # capture anchor debug
                    with contextlib.redirect_stdout(sse_writer):
                        anchors = generate_radial_anchors(n_anchors, *TARGET_SIZE)

                    JOB_RESULTS[job_id].append({
                        "phase": "algorithm",
                        "algorithm": algo,
                        "name": stem,
                        "anchors_json": json.dumps(anchors),
                        "vectors_json": json.dumps(vectors),
                        "vectors_count": len(vectors),
                    })

            _log("Job complete.")

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
