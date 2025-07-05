// stringart_app/static/js/physics_renderer.js
export const PhysicsRenderer = (() => {
  let ctx, nodes, springs;
  const dt = 0.016, damping = 0.98;

  function init({ anchors, vectors, canvas }) {
    ctx = canvas.getContext('2d');
    nodes = anchors.map(([x,y]) => ({ x, y, px: x, py: y }));
    springs = vectors.map(v => {
      const a = nodes[v.from], b = nodes[v.to];
      const dx = b.x - a.x, dy = b.y - a.y;
      return { a, b, rest: Math.hypot(dx, dy), k: 0.1 };
    });
    requestAnimationFrame(step);
  }

  function step() {
    applyForces();
    integrate();
    draw();
    requestAnimationFrame(step);
  }

  function applyForces() {
    for (const s of springs) {
      const { a, b, rest, k } = s;
      let dx = b.x - a.x, dy = b.y - a.y;
      const dist = Math.hypot(dx, dy) || 1;
      const F = k * (dist - rest);
      dx /= dist; dy /= dist;
      // Verlet: move endpoints
      a.x +=  dx * F; a.y +=  dy * F;
      b.x -=  dx * F; b.y -=  dy * F;
    }
  }

  function integrate() {
    for (const n of nodes) {
      const vx = (n.x - n.px) * damping;
      const vy = (n.y - n.py) * damping;
      n.px = n.x; n.py = n.y;
      n.x += vx;   n.y += vy;
    }
  }

  function draw() {
    ctx.clearRect(0,0,ctx.canvas.width,ctx.canvas.height);
    ctx.beginPath();
    for (const s of springs) {
      ctx.moveTo(s.a.x, s.a.y);
      ctx.lineTo(s.b.x, s.b.y);
    }
    ctx.stroke();
  }

  return { init };
})();
