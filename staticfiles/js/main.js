// stringart_app/static/js/main.js
import { createPhysicsRenderer } from './physics_renderer.js';

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.physicsCanvas').forEach((canvas, idx) => {
    try {
      const anchors = JSON.parse(canvas.dataset.anchors);
      const vectors = JSON.parse(canvas.dataset.vectors);
      // create a brand-new sim instance for this canvas:
      createPhysicsRenderer(canvas, anchors, vectors);
    } catch (err) {
      console.error(`Failed to init physicsCanvas #${idx}`, err);
    }
  });
});
