import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { ArrowUpRight, TrendingUp, Package, FlaskConical } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
  Cell,
} from "recharts";
import { api, type Summary } from "../lib/api";
import SpotlightCard from "../components/SpotlightCard";
import AnimatedNumber from "../components/AnimatedNumber";
import PageTransition from "../components/PageTransition";

const CATEGORY_COLORS = ["#7a1b31", "#a8445b", "#c97b8c", "#a9812e", "#6b5348", "#8c6e52"];

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
          <span className="mb-6 inline-flex items-center gap-1.5 rounded-full border border-[var(--border)] bg-white px-3.5 py-1.5 text-[12px] font-medium text-[var(--text-dim)]">
            <span className="h-1.5 w-1.5 rounded-full bg-[var(--green)] pulse-glow" />
            PPO policy live &middot; trained on real transaction data
          </span>
          <h1 className="font-display mx-auto max-w-4xl text-5xl font-semibold leading-[1.08] tracking-tight text-[var(--text)] sm:text-6xl">
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
              className="group flex items-center gap-1.5 rounded-full bg-gradient-to-r from-[var(--burgundy)] to-[var(--burgundy-dark)] px-5 py-2.5 text-[13px] font-semibold text-white shadow-[0_4px_16px_rgba(122,27,49,0.28)] transition-transform hover:scale-[1.03]"
            >
              Explore products
              <ArrowUpRight size={14} className="transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
            </Link>
            <Link
              to="/training"
              className="rounded-full border border-[var(--border)] bg-white px-5 py-2.5 text-[13px] font-medium text-[var(--text)] transition-colors hover:border-[var(--border-hover)] hover:bg-[var(--card-hover)]"
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
            color: "text-[var(--burgundy)]",
          },
          {
            icon: FlaskConical,
            label: "Held-out test set",
            value: summary?.n_test ?? 0,
            suffix: "",
            color: "text-[var(--gold)]",
          },
          {
            icon: TrendingUp,
            label: "Revenue vs. static price",
            value: revenueUplift,
            suffix: "%",
            decimals: 1,
            color: "text-[var(--green)]",
          },
          {
            icon: TrendingUp,
            label: "Profit vs. static price",
            value: profitUplift,
            suffix: "%",
            decimals: 1,
            color: "text-[var(--burgundy-light)]",
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
              <div className="font-display mt-3 text-3xl font-semibold text-[var(--text)]">
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
          <h3 className="font-display text-sm font-semibold text-[var(--text)]">Catalog by category</h3>
          <p className="mt-1 text-[12.5px] text-[var(--text-dim)]">
            Derived from product descriptions &mdash; no category field in the source data
          </p>
          <div className="mt-5 h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={categoryData} layout="vertical" margin={{ left: 8, right: 20 }} barCategoryGap="28%">
                <XAxis type="number" hide />
                <YAxis
                  type="category"
                  dataKey="name"
                  width={130}
                  tick={{ fill: "#7d5d61", fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  cursor={{ fill: "rgba(122,27,49,0.04)" }}
                  contentStyle={{ background: "#ffffff", border: "1px solid var(--border)", borderRadius: 10, fontSize: 12 }}
                  labelStyle={{ color: "var(--text)" }}
                />
                <Bar dataKey="count" radius={[0, 6, 6, 0]} barSize={12}>
                  {categoryData.map((_, i) => (
                    <Cell key={i} fill={CATEGORY_COLORS[i % CATEGORY_COLORS.length]} fillOpacity={0.9} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </SpotlightCard>

        <SpotlightCard className="p-6">
          <h3 className="font-display text-sm font-semibold text-[var(--text)]">Fitted price elasticity distribution</h3>
          <p className="mt-1 text-[12.5px] text-[var(--text-dim)]">
            Median {summary?.elasticity_median.toFixed(2)} &middot; mean {summary?.elasticity_mean.toFixed(2)} &mdash; from real log-log regression fits
          </p>
          <div className="mt-5 h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={histData}>
                <XAxis
                  dataKey="mid"
                  tickFormatter={(v) => v.toFixed(1)}
                  tick={{ fill: "#7d5d61", fontSize: 10 }}
                  axisLine={{ stroke: "var(--border)" }}
                  tickLine={false}
                />
                <Tooltip
                  cursor={{ fill: "rgba(122,27,49,0.04)" }}
                  contentStyle={{ background: "#ffffff", border: "1px solid var(--border)", borderRadius: 10, fontSize: 12 }}
                  labelFormatter={(v) => `elasticity ≈ ${Number(v).toFixed(2)}`}
                />
                <Bar dataKey="count" radius={[4, 4, 0, 0]} fill="var(--burgundy)" fillOpacity={0.85} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </SpotlightCard>
      </section>
    </PageTransition>
  );
}
