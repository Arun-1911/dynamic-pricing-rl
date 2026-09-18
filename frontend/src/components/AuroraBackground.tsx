/**
 * Ambient page backdrop: two soft, static burgundy/gold glows on an ivory
 * ground. Pure CSS, no canvas or per-frame work — restrained by design
 * rather than an animated "demo" effect.
 */
export default function AuroraBackground() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden" aria-hidden>
      <div className="absolute inset-0" style={{ background: "var(--bg)" }} />
      <div
        className="absolute -top-40 left-1/2 h-[560px] w-[900px] -translate-x-1/2 rounded-full opacity-[0.16] blur-[100px]"
        style={{ background: "radial-gradient(closest-side, var(--burgundy), transparent)" }}
      />
      <div
        className="absolute -right-32 top-1/3 h-[420px] w-[420px] rounded-full opacity-[0.12] blur-[100px]"
        style={{ background: "radial-gradient(closest-side, var(--gold), transparent)" }}
      />
    </div>
  );
}
