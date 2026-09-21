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
import { Info } from "lucide-react";
import { api, type LearningPoint, type UpliftReport } from "../lib/api";
import SpotlightCard from "../components/SpotlightCard";
import PageTransition from "../components/PageTransition";

const BASELINES = [
  { key: "static_base_price", label: "Static base price" },
  { key: "tuned_static_oracle", label: "Best constant price" },
  { key: "undercut_competitor_5pct", label: "Undercut competitor 5%" },
  { key: "elasticity_optimal_static", label: "Elasticity-optimal static" },
] as const;

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

export default function Training() {
  const [curve, setCurve] = useState<LearningPoint[]>([]);
  const [uplift, setUplift] = useState<UpliftReport | null>(null);

  useEffect(() => {
    api.learningCurve(200).then(setCurve).catch(console.error);
    api.uplift().then(setUplift).catch(console.error);
  }, []);

  const barData = uplift
    ? BASELINES.map(({ key, label }) => ({
        name: label,
        revenue: uplift[key].revenue_uplift_vs_ppo_pct,
        profit: uplift[key].profit_uplift_vs_ppo_pct,
      }))
    : [];

  return (
    <PageTransition>
      <div className="space-y-6 py-10">
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <h1 className="font-display text-2xl font-semibold text-[var(--text)]">Training results</h1>
          <p className="mt-1 text-[13px] text-[var(--text-dim)]">
            PPO trained for 400,000 timesteps and evaluated on {uplift?.n_held_out_products ?? "…"} held-out products,{" "}
            {uplift?.episodes_per_product ?? "…"} episodes each.
          </p>
        </motion.div>

        <SpotlightCard className="p-6">
          <h3 className="font-display text-sm font-semibold text-[var(--text)]">Episode reward during training</h3>
          <p className="mt-1 text-[12.5px] text-[var(--text-dim)]">
            Real learning curve &mdash; reward is weekly gross profit normalized by each product's base revenue.
          </p>
          <div className="mt-5 h-72">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={curve}>
                <defs>
                  <linearGradient id="rewardGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#7a1b31" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="#7a1b31" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="var(--border)" vertical={false} />
                <XAxis
                  dataKey="timestep"
                  tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`}
                  tick={{ fill: "#7d5d61", fontSize: 10 }}
                  axisLine={{ stroke: "var(--border)" }}
                  tickLine={false}
                />
                <YAxis tick={{ fill: "#7d5d61", fontSize: 10 }} axisLine={false} tickLine={false} />
                <Tooltip
                  contentStyle={{ background: "#ffffff", border: "1px solid var(--border)", borderRadius: 10, fontSize: 12 }}
                  labelFormatter={(v) => `timestep ${Number(v).toLocaleString()}`}
                />
                <Area type="monotone" dataKey="episode_reward" stroke="var(--burgundy)" strokeWidth={2} fill="url(#rewardGrad)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </SpotlightCard>

        <SpotlightCard className="p-6">
          <h3 className="font-display text-sm font-semibold text-[var(--text)]">PPO uplift vs. baselines (held-out products)</h3>
          <p className="mt-1 text-[12.5px] text-[var(--text-dim)]">
            Measured on {uplift?.n_held_out_products} products never seen during training, {uplift?.episodes_per_product} episodes each.
          </p>
          <div className="mt-5 h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData}>
                <CartesianGrid stroke="var(--border)" vertical={false} />
                <XAxis dataKey="name" tick={{ fill: "#7d5d61", fontSize: 11 }} axisLine={{ stroke: "var(--border)" }} tickLine={false} />
                <YAxis tickFormatter={(v) => `${v}%`} tick={{ fill: "#7d5d61", fontSize: 10 }} axisLine={false} tickLine={false} />
                <Tooltip
                  contentStyle={{ background: "#ffffff", border: "1px solid var(--border)", borderRadius: 10, fontSize: 12 }}
                  formatter={(v) => `${Number(v).toFixed(1)}%`}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="revenue" name="Revenue uplift" fill="var(--burgundy)" radius={[6, 6, 0, 0]} fillOpacity={0.9} />
                <Bar dataKey="profit" name="Profit uplift" fill="var(--gold)" radius={[6, 6, 0, 0]} fillOpacity={0.9} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </SpotlightCard>

        {uplift && (
          <SpotlightCard className="p-6">
            <div className="flex items-start gap-3">
              <Info size={16} className="mt-0.5 shrink-0 text-[var(--gold)]" />
              <div>
                <h3 className="font-display text-sm font-semibold text-[var(--text)]">How to read these numbers</h3>
                <p className="mt-2 text-[13px] leading-relaxed text-[var(--text-dim)]">
                  Against never changing the price, PPO gains {uplift.static_base_price.revenue_uplift_vs_ppo_pct.toFixed(1)}% revenue
                  and {uplift.static_base_price.profit_uplift_vs_ppo_pct.toFixed(1)}% profit. That gain is small because the catalog's
                  base prices are the retailer's own historical prices and are already close to the best constant price: even the best
                  constant price per product, found by grid search inside the simulator, improves profit only slightly over the base
                  price. PPO still edges out that oracle by {uplift.tuned_static_oracle.profit_uplift_vs_ppo_pct.toFixed(1)}% profit by
                  reacting to competitor price and stock levels.
                </p>
                <p className="mt-2 text-[13px] leading-relaxed text-[var(--text-dim)]">
                  The larger gaps against the last two baselines are not large gains from learning. Those strategies cut prices and
                  sell faster than the retailer restocks, so their sales were limited by stock in{" "}
                  {pct(uplift.undercut_competitor_5pct.supply_capped_share)} and {pct(uplift.elasticity_optimal_static.supply_capped_share)} of
                  weeks, against {pct(uplift.ppo.supply_capped_share)} for PPO. All results are simulated (competitor price and inventory
                  are modelled) and depend on the inventory assumption.
                </p>
              </div>
            </div>
          </SpotlightCard>
        )}
      </div>
    </PageTransition>
  );
}
