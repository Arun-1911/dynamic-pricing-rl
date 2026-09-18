import { NavLink } from "react-router-dom";
import { motion } from "framer-motion";
import { LayoutGrid, SlidersHorizontal, LineChart, Gem } from "lucide-react";

const links = [
  { to: "/", label: "Overview", icon: LayoutGrid },
  { to: "/explore", label: "Explorer", icon: SlidersHorizontal },
  { to: "/training", label: "Training", icon: LineChart },
];

export default function NavBar() {
  return (
    <motion.header
      initial={{ y: -40, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="sticky top-0 z-50 border-b border-[var(--border)] bg-[var(--bg)]/85 backdrop-blur-xl"
    >
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-[var(--burgundy)] to-[var(--burgundy-dark)] shadow-[0_2px_10px_rgba(122,27,49,0.35)]">
            <Gem size={15} className="text-[var(--gold-light)]" strokeWidth={2} />
          </div>
          <span className="font-display text-[15px] font-semibold tracking-tight text-[var(--text)]">
            Dynamic Pricing <span className="text-gradient">RL</span>
          </span>
        </div>

        <nav className="flex items-center gap-1 rounded-full border border-[var(--border)] bg-white p-1">
          {links.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                `relative flex items-center gap-1.5 rounded-full px-4 py-1.5 text-[13px] font-medium transition-colors ${
                  isActive ? "text-white" : "text-[var(--text-dim)] hover:text-[var(--burgundy)]"
                }`
              }
            >
              {({ isActive }) => (
                <>
                  {isActive && (
                    <motion.span
                      layoutId="nav-pill"
                      className="absolute inset-0 rounded-full bg-gradient-to-r from-[var(--burgundy)] to-[var(--burgundy-dark)]"
                      transition={{ type: "spring", stiffness: 400, damping: 32 }}
                    />
                  )}
                  <Icon size={13} className="relative" />
                  <span className="relative">{label}</span>
                </>
              )}
            </NavLink>
          ))}
        </nav>
      </div>
    </motion.header>
  );
}
