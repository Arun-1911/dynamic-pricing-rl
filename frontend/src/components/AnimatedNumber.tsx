import { useEffect, useState } from "react";
import { motion } from "framer-motion";

interface Props {
  value: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  duration?: number;
  className?: string;
}

/** Counts up from 0 to `value` on mount / whenever `value` changes. */
export default function AnimatedNumber({
  value,
  decimals = 0,
  prefix = "",
  suffix = "",
  duration = 1.2,
  className = "",
}: Props) {
  const [display, setDisplay] = useState(0);

  useEffect(() => {
    // requestAnimationFrame is throttled/paused while the tab is hidden —
    // skip straight to the final value rather than getting stuck at 0.
    if (document.hidden) {
      setDisplay(value);
      return;
    }
    let raf: number;
    const start = performance.now();
    function tick(now: number) {
      const elapsed = (now - start) / 1000;
      const progress = Math.min(1, elapsed / duration);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(value * eased);
      if (progress < 1 && !document.hidden) raf = requestAnimationFrame(tick);
      else setDisplay(value);
    }
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value, duration]);

  return (
    <motion.span
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className={className}
    >
      {prefix}
      {display.toFixed(decimals)}
      {suffix}
    </motion.span>
  );
}
