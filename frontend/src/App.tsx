import { Routes, Route, useLocation } from "react-router-dom";
import { AnimatePresence } from "framer-motion";
import AuroraBackground from "./components/AuroraBackground";
import NavBar from "./components/NavBar";
import Overview from "./pages/Overview";
import Explorer from "./pages/Explorer";
import Training from "./pages/Training";

export default function App() {
  const location = useLocation();

  return (
    <div className="relative min-h-screen text-[var(--text)]">
      <AuroraBackground />
      <div className="grid-bg fixed inset-0 -z-10 [mask-image:radial-gradient(ellipse_80%_60%_at_50%_0%,black,transparent)]" />
      <NavBar />
      <main className="mx-auto max-w-7xl px-6 pb-24">
        <AnimatePresence mode="wait">
          <Routes location={location} key={location.pathname}>
            <Route path="/" element={<Overview />} />
            <Route path="/explore" element={<Explorer />} />
            <Route path="/training" element={<Training />} />
          </Routes>
        </AnimatePresence>
      </main>
    </div>
  );
}
