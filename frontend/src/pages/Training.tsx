import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
  CartesianGrid,
  Legend,
} from "recharts";
import { AlertTriangle } from "lucide-react";
import { api, type LearningPoint, type UpliftReport } from "../lib/api";
import SpotlightCard from "../components/SpotlightCard";
import PageTransition from "../components/PageTransition";

const BASELINE_LABELS: Record<string, string> = {
  static_base_price: "Static base price",
  undercut_competitor_5pct: "Undercut competitor 5%",
  elasticity_optimal_static: "Elasticity-optimal static",
};

export default function Training() {
  const [curve, setCurve] = useState<LearningPoint[]>([]);
  const [uplift, setUplift] = useState<UpliftReport | null>(null);

  useEffect(() => {
    api.learningCurve(200).then(setCurve).catch(console.error);
    api.uplift().then(setUplift).catch(console.error);
  }, []);

  const barData = uplift
    ? (["static_base_price", "undercut_competitor_5pct", "elasticity_optimal_static"] as const).map((key) => ({
        name: BASELINE_LABELS[key],
        revenue: uplift[key].revenue_uplift_vs_ppo_pct,
        profit: uplift[key].profit_uplift_vs_ppo_pct,
      }))
    : [];

  return (
    <PageTransition>
      <div className="space-y-6 py-10">
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <h1 className="font-display text-2xl font-semibold text-white">Training results</h1>
          <p className="mt-1 text-[13px] text-[var(--text-dim)]">
            PPO trained for 400,000 timesteps across {uplift?.episodes_per_product ?? "…"} × {uplift?.n_held_out_products ?? "…"} held-out
            evaluation episodes.
          </p>
        </motion.div>

        <SpotlightCard className="p-6">
          <h3 className="font-display text-sm font-semibold text-white">Episode reward during training</h3>
          <p className="mt-1 text-[12.5px] text-[var(--text-dim)]">
            Real learning curve &mdash; reward is profit normalized by each product's base revenue, minus stockout/excess-inventory penalties.
          </p>
          <div className="mt-5 h-72">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={curve}>
                <defs>
                  <linearGradient id="rewardGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#a78bfa" stopOpacity={0.5} />
                    <stop offset="100%" stopColor="#a78bfa" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis
                  dataKey="timestep"
                  tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`}
                  tick={{ fill: "#8b90ac", fontSize: 10 }}
                  axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
                  tickLine={false}
                />
                <YAxis tick={{ fill: "#8b90ac", fontSize: 10 }} axisLine={false} tickLine={false} />
                <Tooltip
                  contentStyle={{ background: "#0a0d1a", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 10, fontSize: 12 }}
                  labelFormatter={(v) => `timestep ${Number(v).toLocaleString()}`}
                />
                <Area type="monotone" dataKey="episode_reward" stroke="#a78bfa" strokeWidth={2} fill="url(#rewardGrad)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </SpotlightCard>

        <SpotlightCard className="p-6">
          <h3 className="font-display text-sm font-semibold text-white">PPO uplift vs. baselines (held-out products)</h3>
          <p className="mt-1 text-[12.5px] text-[var(--text-dim)]">
            Measured on {uplift?.n_held_out_products} products never seen during training, {uplift?.episodes_per_product} episodes each.
          </p>
          <div className="mt-5 h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="name" tick={{ fill: "#8b90ac", fontSize: 11 }} axisLine={{ stroke: "rgba(255,255,255,0.08)" }} tickLine={false} />
                <YAxis tickFormatter={(v) => `${v}%`} tick={{ fill: "#8b90ac", fontSize: 10 }} axisLine={false} tickLine={false} />
                <Tooltip
                  contentStyle={{ background: "#0a0d1a", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 10, fontSize: 12 }}
                  formatter={(v) => `${Number(v).toFixed(1)}%`}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="revenue" name="Revenue uplift" fill="#22d3ee" radius={[6, 6, 0, 0]} fillOpacity={0.85} />
                <Bar dataKey="profit" name="Profit uplift" fill="#f472b6" radius={[6, 6, 0, 0]} fillOpacity={0.85} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </SpotlightCard>

        <SpotlightCard className="p-6">
          <div className="flex items-start gap-3">
            <AlertTriangle size={16} className="mt-0.5 shrink-0 text-amber-300" />
            <div>
              <h3 className="font-display text-sm font-semibold text-white">
                Why "elasticity-optimal static" is the most interesting baseline
              </h3>
              <p className="mt-2 text-[13px] leading-relaxed text-[var(--text-dim)]">
                It's a static price computed once from the same fitted elasticity PPO can see, using the textbook
                constant-elasticity monopoly formula <code className="font-mono-tech text-cyan-300">price* = cost × e/(e+1)</code>.
                It should be the hardest baseline to beat &mdash; yet it underperforms even the do-nothing baseline.
                Tracing individual rollouts showed why: ~34% of held-out products get pushed to the price bounds because
                the formula optimizes only the isolated elasticity term and is blind to the competitor-price dynamic in
                the environment. PPO, which observes competitor price every step, avoids that trap. That gap is the real
                value dynamic, context-aware pricing adds over "just know your elasticity."
              </p>
            </div>
          </div>
        </SpotlightCard>
      </div>
    </PageTransition>
  );
}
