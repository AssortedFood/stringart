// stringart_app/static/js/physics_renderer.js

/**
 * StringArtStreamer
 *
 * Incrementally renders “string art” build‐up on a <canvas> as individual
 * (from, to) line vectors are fed in. It:
 *  - Computes N radial anchor points around a circle.
 *  - Maintains a per-pixel coverage map (`counts`) to track how many
 *    times each pixel has been drawn over.
 *  - After each new line, recomputes the maximum overlap `k_max` and
 *    adjusts the global stroke alpha so the densest region stays fully black.
 *  - Draws each new line immediately at the appropriate opacity.
 */
export class StringArtStreamer {
  constructor(canvas, nodeCount) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.W = canvas.width;
    this.H = canvas.height;
    this.nodeCount = nodeCount;

    // Center & radius for the radial layout
    this.CX = this.W / 2;
    this.CY = this.H / 2;
    this.R  = Math.min(this.W, this.H) * 0.45;

    // Precompute anchor coordinates
    this.coords = Array.from({ length: nodeCount }, (_, i) => {
      const angle = 2 * Math.PI * i / nodeCount - Math.PI / 2;
      return {
        x: this.CX + this.R * Math.cos(angle),
        y: this.CY + this.R * Math.sin(angle)
      };
    });

    // Coverage counts per pixel
    this.counts = new Uint32Array(this.W * this.H);

    // Draw optional anchor dots
    this._drawAnchors();

    // Clear canvas before streaming lines
    this.ctx.clearRect(0, 0, this.W, this.H);
  }

  /**
   * Plot a single pixel‐wise line into our coverage counts using
   * Bresenham's algorithm.
   */
  _plotLineToCounts(x0, y0, x1, y1) {
    let dx =  Math.abs(x1 - x0),
        dy = -Math.abs(y1 - y0),
        sx = x0 < x1 ? 1 : -1,
        sy = y0 < y1 ? 1 : -1,
        err = dx + dy;

    while (true) {
      if (x0 >= 0 && x0 < this.W && y0 >= 0 && y0 < this.H) {
        this.counts[y0 * this.W + x0]++;
      }
      if (x0 === x1 && y0 === y1) break;
      const e2 = 2 * err;
      if (e2 >= dy) { err += dy; x0 += sx; }
      if (e2 <= dx) { err += dx; y0 += sy; }
    }
  }

  /**
   * Recomputes k_max, the maximum count across all pixels.
   */
  _computeKMax() {
    let max = 0;
    for (let i = 0; i < this.counts.length; i++) {
      if (this.counts[i] > max) {
        max = this.counts[i];
      }
    }
    return max || 1;
  }

  /**
   * Draw small dots at each anchor location for visual reference.
   */
  _drawAnchors() {
    this.ctx.fillStyle = 'rgba(0,0,0,0.2)';
    for (const { x, y } of this.coords) {
      this.ctx.beginPath();
      this.ctx.arc(x, y, 2, 0, 2 * Math.PI);
      this.ctx.fill();
    }
  }

  /**
   * Add and draw a single line from `from` to `to` anchor indices.
   * Updates the coverage map, recomputes alpha, and strokes the line.
   */
  addLine(from, to) {
    const A = this.coords[from];
    const B = this.coords[to];

    // 1) Update coverage counts
    this._plotLineToCounts(
      Math.round(A.x), Math.round(A.y),
      Math.round(B.x), Math.round(B.y)
    );

    // 2) Recompute stroke alpha so that max overlap → black
    const kMax = this._computeKMax();
    this.ctx.strokeStyle = `rgba(0,0,0,${1 / kMax})`;
    this.ctx.lineWidth = 1;

    // 3) Draw the new line
    this.ctx.beginPath();
    this.ctx.moveTo(A.x, A.y);
    this.ctx.lineTo(B.x, B.y);
    this.ctx.stroke();
  }
}
