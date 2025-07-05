// stringart_app/static/js/physics_renderer.js

/**
 * Creates and starts a physics‐based string‐art animation
 * on a single canvas. Each call is fully independent.
 */
export function createPhysicsRenderer(canvas, anchors, vectors) {
  const ctx = canvas.getContext('2d');
  ctx.strokeStyle = 'black';
  ctx.lineWidth   = 1;

  // build Verlet nodes
  const nodes = anchors.map(([x, y]) => ({ x, y, px: x, py: y }));
  // build springs
  const springs = vectors.map(({ from, to }) => {
    const a = nodes[from], b = nodes[to];
    const dx = b.x - a.x, dy = b.y - a.y;
    return { a, b, rest: Math.hypot(dx, dy), k: 0.1 };
  });

  const DAMPING = 0.98;

  function step() {
    // 1. apply spring forces
    for (const { a, b, rest, k } of springs) {
      let dx = b.x - a.x, dy = b.y - a.y;
      const dist = Math.hypot(dx, dy) || 1;
      const F = k * (dist - rest);
      dx /= dist; dy /= dist;
      a.x += dx * F; a.y += dy * F;
      b.x -= dx * F; b.y -= dy * F;
    }

    // 2. integrate (Verlet)
    for (const n of nodes) {
      const vx = (n.x - n.px) * DAMPING;
      const vy = (n.y - n.py) * DAMPING;
      n.px = n.x; n.py = n.y;
      n.x += vx;   n.y += vy;
    }

    // 3. draw strings
    const { width, height } = ctx.canvas;
    ctx.clearRect(0, 0, width, height);
    ctx.beginPath();
    for (const { a, b } of springs) {
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
    }
    ctx.stroke();

    requestAnimationFrame(step);
  }

  // kick things off
  requestAnimationFrame(step);
}
