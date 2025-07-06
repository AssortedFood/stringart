// stringart_app/static/js/main.js
import { createStringArtRenderer } from './physics_renderer.js';

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.physicsCanvas').forEach((canvas, idx) => {
    try {
      const anchors = JSON.parse(canvas.dataset.anchors);
      const vectors = JSON.parse(canvas.dataset.vectors);
      // derive node count from anchors array length
      const nodeCount = anchors.length;
      // initialize the new string‐art renderer
      createStringArtRenderer(canvas, nodeCount, vectors);
    } catch (err) {
      console.error(`Failed to init physicsCanvas #${idx}`, err);
    }
  });
});
