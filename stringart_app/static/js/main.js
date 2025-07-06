// stringart_app/static/js/main.js

import { StringArtStreamer } from './physics_renderer.js';

let evtSourceLogs, evtSourceResults;
let currentJobId = null;

// Hold StringArtStreamer instances per job/algo/image
const canvasContexts = {};

/** Capitalize first letter */
function capFirst(str) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

/** Read Django CSRF token from cookie */
function getCSRFToken() {
  const match = document.cookie.match(/(^|;)\s*csrftoken=([^;]+)/);
  return match ? match.pop() : '';
}

/** Kick off the background job via AJAX; returns job_id */
async function kickOffJob(form) {
  const resp = await fetch('.', {
    method: 'POST',
    headers: { 'X-Requested-With': 'XMLHttpRequest' },
    body: new FormData(form)
  });
  const data = await resp.json();
  return data.job_id;
}

/**
 * Start streaming logs and results for a given job_id.
 * Creates and updates canvases via StringArtStreamer.
 */
function startStreams(jobId) {
  currentJobId = jobId;
  canvasContexts[jobId] = {};

  // Clear previous results/logs
  document.getElementById('log-content').textContent = '';
  const container = document.getElementById('results-container');
  container.innerHTML = '';

  document.getElementById('results-section').style.display = '';
  document.getElementById('debug-section').style.display = '';

  // Stream logs
  evtSourceLogs = new EventSource(`{% url 'stream_logs' %}?job_id=${jobId}`);
  evtSourceLogs.onmessage = e => {
    const logEl = document.getElementById('log-content');
    logEl.textContent += e.data + "\n";
    logEl.scrollTop = logEl.scrollHeight;
  };

  // Stream results
  evtSourceResults = new EventSource(`{% url 'stream_results' %}?job_id=${jobId}`);
  evtSourceResults.onmessage = e => {
    const t = JSON.parse(e.data);

    // New phase?
    if (t.phase !== window.currentPhase) {
      window.currentPhase = t.phase;
      window.currentAlgo = null;
      const h2 = document.createElement('h2');
      h2.textContent = capFirst(t.phase);
      container.appendChild(h2);
    }

    // Grayscale thumbnails
    if (t.phase === 'grayscale') {
      if (!window.currentRow) {
        window.currentRow = document.createElement('div');
        window.currentRow.className = 'preview-grid';
        container.appendChild(window.currentRow);
      }
      const img = document.createElement('img');
      img.src = `data:image/png;base64,${t.processed_image}`;
      img.alt = t.name;
      window.currentRow.appendChild(img);

    // Algorithm streaming
    } else if (t.phase === 'algorithm') {
      // New algorithm block?
      if (t.algorithm !== window.currentAlgo || !window.currentRow) {
        window.currentAlgo = t.algorithm;
        window.currentRow = document.createElement('div');
        window.currentRow.className = 'preview-grid';
        const h3 = document.createElement('h3');
        h3.textContent = capFirst(t.algorithm);
        container.appendChild(h3);
        container.appendChild(window.currentRow);
      }

      const key = `${t.algorithm}::${t.name}`;

      // First vector for this image? create canvas + streamer
      if (!canvasContexts[jobId][key]) {
        const cell = document.createElement('div');

        const details = document.createElement('details');
        const summary = document.createElement('summary');
        summary.textContent = `Vectors for ${t.name}`;
        details.appendChild(summary);
        cell.appendChild(details);

        const canvas = document.createElement('canvas');
        canvas.width = 200;
        canvas.height = 200;
        cell.appendChild(canvas);
        window.currentRow.appendChild(cell);

        // Use node_count from SSE payload
        const streamer = new StringArtStreamer(canvas, t.node_count);
        canvasContexts[jobId][key] = streamer;
      }

      // Draw the incoming vector
      if (t.vector) {
        const streamer = canvasContexts[jobId][key];
        streamer.addLine(t.vector.from, t.vector.to);
      }
    }
  };

  // Enable Stop button
  const stopBtn = document.getElementById('stop-btn');
  stopBtn.disabled = false;
  stopBtn.onclick = async () => {
    await fetch(`/stop-job/${jobId}/`, {
      method: 'POST',
      headers: { 'X-CSRFToken': getCSRFToken() }
    });
    stopBtn.disabled = true;
  };

  // Auto-stop if the page unloads
  window.addEventListener('beforeunload', () => {
    if (currentJobId) {
      navigator.sendBeacon(`/stop-job/${currentJobId}/`);
    }
  });
}

document.addEventListener('DOMContentLoaded', () => {
  // Attach form submit handler
  const form = document.getElementById('algo-form');
  if (form) {
    form.addEventListener('submit', async e => {
      e.preventDefault();
      const jobId = await kickOffJob(form);
      startStreams(jobId);
    });
  }
});
