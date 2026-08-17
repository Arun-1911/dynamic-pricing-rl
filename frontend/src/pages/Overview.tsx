import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { ArrowUpRight, TrendingUp, Package, FlaskConical } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  ResponsiveContainer,
  Tooltip,
  Cell,
} from "recharts";
import { api, type Summary } from "../lib/api";
import SpotlightCard from "../components/SpotlightCard";
import AnimatedNumber from "../components/AnimatedNumber";
import PageTransition from "../components/PageTransition";

const CATEGORY_COLORS = ["#22d3ee", "#a78bfa", "#f472b6", "#34d399", "#fbbf24", "#fb7185"];

export default function Overview() {
  const [summary, setSummary] = useState<Summary | null>(null);

  useEffect(() => {
    api.summary().then(setSummary).catch(console.error);
  }, []);

  const revenueUplift = summary?.uplift_report.static_base_price.revenue_uplift_vs_ppo_pct ?? 0;
  const profitUplift = summary?.uplift_report.static_base_price.profit_uplift_vs_ppo_pct ?? 0;

  const categoryData = summary
    ? Object.entries(summary.category_counts)
        .sort((a, b) => b[1] - a[1])
        .map(([name, count]) => ({ name, count }))
    : [];

  const histData = summary?.elasticity_histogram.map((b) => ({
    mid: (b.bin_start + b.bin_end) / 2,
    count: b.count,
  }));

  return (
    <PageTransition>
      {/* Hero */}
      <section className="relative py-24 text-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
        >
          <span className="mb-6 inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.03] px-3.5 py-1.5 text-[12px] font-medium text-[var(--text-dim)]">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 pulse-glow" />
            PPO policy live &middot; trained on real transaction data
          </span>
          <h1 className="font-display mx-auto max-w-4xl text-5xl font-semibold leading-[1.08] tracking-tight text-white sm:text-6xl">
            Dynamic pricing,
            <br />
            <span className="text-gradient">reinforced.</span>
          </h1>
          <p className="mx-auto mt-6 max-w-xl text-[15px] leading-relaxed text-[var(--text-dim)]">
            A PPO agent priced {summary ? summary.n_products.toLocaleString() : "…"} real
            e-commerce products from fitted price elasticity &mdash; evaluated on{" "}
            {summary?.n_test.toLocaleString()} it never trained on.
          </p>
          <div className="mt-9 flex items-center justify-center gap-3">
            <Link
              to="/explore"
              className="group flex items-center gap-1.5 rounded-full bg-gradient-to-r from-cyan-400 via-violet-400 to-pink-400 px-5 py-2.5 text-[13px] font-semibold text-black transition-transform hover:scale-[1.03]"
            >
              Explore products
              <ArrowUpRight size={14} className="transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
            </Link>
            <Link
              to="/training"
              className="rounded-full border border-white/10 bg-white/[0.03] px-5 py-2.5 text-[13px] font-medium text-white transition-colors hover:bg-white/[0.07]"
            >
              View training results
            </Link>
          </div>
        </motion.div>
      </section>

      {/* Stat cards */}
      <section className="grid grid-cols-2 gap-4 pb-16 sm:grid-cols-4">
        {[
          {
            icon: Package,
            label: "Real products",
            value: summary?.n_products ?? 0,
            suffix: "",
            color: "text-cyan-300",
          },
          {
            icon: FlaskConical,
            label: "Held-out test set",
            value: summary?.n_test ?? 0,
            suffix: "",
            color: "text-violet-300",
          },
          {
            icon: TrendingUp,
            label: "Revenue uplift",
            value: revenueUplift,
            suffix: "%",
            decimals: 1,
            color: "text-emerald-300",
          },
          {
            icon: TrendingUp,
            label: "Profit uplift",
            value: profitUplift,
            suffix: "%",
            decimals: 1,
            color: "text-pink-300",
          },
        ].map((stat, i) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: i * 0.06 }}
          >
            <SpotlightCard className="p-5">
              <stat.icon size={16} className={stat.color} />
              <div className="font-display mt-3 text-3xl font-semibold text-white">
                <AnimatedNumber value={stat.value} decimals={stat.decimals ?? 0} suffix={stat.suffix} prefix={stat.suffix === "%" ? "+" : ""} />
              </div>
              <div className="mt-1 text-[12.5px] text-[var(--text-dim)]">{stat.label}</div>
            </SpotlightCard>
          </motion.div>
        ))}
      </section>

      {/* Category + elasticity charts */}
      <section className="grid grid-cols-1 gap-5 pb-20 lg:grid-cols-2">
        <SpotlightCard className="p-6">
          <h3 className="font-display text-sm font-semibold text-white">Catalog by category</h3>
          <p className="mt-1 text-[12.5px] text-[var(--text-dim)]">
            Derived from product descriptions &mdash; no category field in the source data
          </p>
          <div className="mt-5 h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={categoryData} layout="vertical" margin={{ left: 8, right: 16 }}>
                <XAxis type="number" hide />
                <Tooltip
                  cursor={{ fill: "rgba(255,255,255,0.03)" }}
                  contentStyle={{ background: "#0a0d1a", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 10, fontSize: 12 }}
                  labelStyle={{ color: "#e7e9f5" }}
                />
                <Bar dataKey="count" radius={[0, 6, 6, 0]} barSize={14}>
                  {categoryData.map((_, i) => (
                    <Cell key={i} fill={CATEGORY_COLORS[i % CATEGORY_COLORS.length]} fillOpacity={0.85} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </SpotlightCard>

        <SpotlightCard className="p-6">
          <h3 className="font-display text-sm font-semibold text-white">Fitted price elasticity distribution</h3>
          <p className="mt-1 text-[12.5px] text-[var(--text-dim)]">
            Median {summary?.elasticity_median.toFixed(2)} &middot; mean {summary?.elasticity_mean.toFixed(2)} &mdash; from real log-log regression fits
          </p>
          <div className="mt-5 h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={histData}>
                <XAxis
                  dataKey="mid"
                  tickFormatter={(v) => v.toFixed(1)}
                  tick={{ fill: "#8b90ac", fontSize: 10 }}
                  axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
                  tickLine={false}
                />
                <Tooltip
                  cursor={{ fill: "rgba(255,255,255,0.03)" }}
                  contentStyle={{ background: "#0a0d1a", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 10, fontSize: 12 }}
                  labelFormatter={(v) => `elasticity ≈ ${Number(v).toFixed(2)}`}
                />
                <Bar dataKey="count" radius={[4, 4, 0, 0]} fill="#a78bfa" fillOpacity={0.8} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </SpotlightCard>
      </section>
    </PageTransition>
  );
}
