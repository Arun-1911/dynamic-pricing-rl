import { useEffect, useRef } from "react";

/**
 * Animated canvas background: soft drifting aurora blobs + a subtle particle
 * field, dark-theme only. Pure canvas, no deps, GPU-cheap.
 */
export default function AuroraBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const maybeCtx = canvas.getContext("2d");
    if (!maybeCtx) return;
    const ctx: CanvasRenderingContext2D = maybeCtx;

    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);

    const onResize = () => {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    };
    window.addEventListener("resize", onResize);

    const blobs = [
      { x: 0.2, y: 0.25, r: 0.35, color: "34,211,238", speed: 0.00018, phase: 0 },
      { x: 0.8, y: 0.2, r: 0.3, color: "167,139,250", speed: 0.00022, phase: 2 },
      { x: 0.5, y: 0.8, r: 0.38, color: "244,114,182", speed: 0.00015, phase: 4 },
    ];

    const particles = Array.from({ length: 60 }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      r: Math.random() * 1.4 + 0.3,
      vx: (Math.random() - 0.5) * 0.15,
      vy: (Math.random() - 0.5) * 0.15,
      alpha: Math.random() * 0.5 + 0.15,
    }));

    let raf = 0;
    const start = performance.now();

    function draw(now: number) {
      const t = now - start;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#05060d";
      ctx.fillRect(0, 0, width, height);

      for (const b of blobs) {
        const cx = (b.x + Math.sin(t * b.speed + b.phase) * 0.06) * width;
        const cy = (b.y + Math.cos(t * b.speed * 0.8 + b.phase) * 0.06) * height;
        const radius = b.r * Math.max(width, height);
        const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
        grad.addColorStop(0, `rgba(${b.color},0.16)`);
        grad.addColorStop(1, `rgba(${b.color},0)`);
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, width, height);
      }

      for (const p of particles) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0) p.x = width;
        if (p.x > width) p.x = 0;
        if (p.y < 0) p.y = height;
        if (p.y > height) p.y = 0;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(230,232,245,${p.alpha})`;
        ctx.fill();
      }

      raf = requestAnimationFrame(draw);
    }
    raf = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 -z-10 h-full w-full"
      aria-hidden
    />
  );
}
