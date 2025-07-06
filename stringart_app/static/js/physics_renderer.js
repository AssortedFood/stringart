// stringart_app/static/js/physics_renderer.js

/**
 * Renders a “string art” build-up:
 *  - N nodes around a circle (0 at 12 o’clock, clockwise)
 *  - Precomputes per-pixel overlap counts → finds k_max
 *  - Uses alpha = 1/k_max so densest region is pitch-black
 *  - Draws each line one frame at a time
 */
export function createStringArtRenderer(canvas, nodeCount, vectors) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  const CX = W / 2;
  const CY = H / 2;
  const R  = Math.min(W, H) * 0.45;

  // 1) compute our N anchor coords
  const coords = Array.from({ length: nodeCount }, (_, i) => {
    const angle = 2 * Math.PI * i / nodeCount - Math.PI / 2;
    return {
      x: CX + R * Math.cos(angle),
      y: CY + R * Math.sin(angle)
    };
  });

  // 2) build a per-pixel coverage map
  //    we'll round coords to integers and rasterize each line
  const counts = new Uint32Array(W * H);
  function plotLine(x0, y0, x1, y1) {
    // Bresenham's line algorithm
    let dx =  Math.abs(x1 - x0),
        dy = -Math.abs(y1 - y0),
        sx = x0 < x1 ? 1 : -1,
        sy = y0 < y1 ? 1 : -1,
        err = dx + dy;
    while (true) {
      if (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H) {
        counts[y0 * W + x0]++;
      }
      if (x0 === x1 && y0 === y1) break;
      let e2 = 2 * err;
      if (e2 >= dy) { err += dy; x0 += sx; }
      if (e2 <= dx) { err += dx; y0 += sy; }
    }
  }

  // Rasterize every vector once to build counts
  for (const { from, to } of vectors) {
    const A = coords[from], B = coords[to];
    plotLine(
      Math.round(A.x), Math.round(A.y),
      Math.round(B.x), Math.round(B.y)
    );
  }

  // Find maximum overlaps
  const k_max = counts.reduce((m, c) => c > m ? c : m, 0) || 1;

  // 3) configure stroke so that k_max * alpha = 1 → full black
  ctx.lineWidth   = 1;
  ctx.strokeStyle = `rgba(0,0,0,${1 / k_max})`;

  // (Optional) draw tiny anchor dots
  ctx.fillStyle = 'rgba(0,0,0,0.2)';
  for (const { x, y } of coords) {
    ctx.beginPath();
    ctx.arc(x, y, 2, 0, 2 * Math.PI);
    ctx.fill();
  }

  // 4) draw one line per frame
  let idx = 0;
  function drawNext() {
    if (idx >= vectors.length) return;
    const { from, to } = vectors[idx++];
    const A = coords[from], B = coords[to];
    ctx.beginPath();
    ctx.moveTo(A.x, A.y);
    ctx.lineTo(B.x, B.y);
    ctx.stroke();
    requestAnimationFrame(drawNext);
  }

  // clear & start
  ctx.clearRect(0, 0, W, H);
  drawNext();
}
