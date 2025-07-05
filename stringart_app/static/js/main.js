// stringart_app/static/js/main.js
import { PhysicsRenderer } from './physics_renderer.js';

const DEBUG = false;

document.addEventListener('DOMContentLoaded', () => {
  const canvas = document.getElementById('physicsCanvas');
  if (!canvas) return;

  const anchorsRaw = canvas.getAttribute('data-anchors');
  const vectorsRaw = canvas.getAttribute('data-vectors');

  if (DEBUG) {
    console.log('anchorsRaw=', anchorsRaw);
    console.log('vectorsRaw=', vectorsRaw);
  }

  const anchors = JSON.parse(anchorsRaw);
  const vectors = JSON.parse(vectorsRaw);

  PhysicsRenderer.init({ anchors, vectors, canvas });
});
