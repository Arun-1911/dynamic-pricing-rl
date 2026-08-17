import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Search, Lock, Sparkles } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import { api, type Product, type Recommendation, type Explanation, type CurvePoint } from "../lib/api";
import SpotlightCard from "../components/SpotlightCard";
import AnimatedNumber from "../components/AnimatedNumber";
import PageTransition from "../components/PageTransition";

function useDebounced<T>(value: T, delay = 250): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(t);
  }, [value, delay]);
  return debounced;
}

export default function Explorer() {
  const [categories, setCategories] = useState<{ category: string; count: number }[]>([]);
  const [category, setCategory] = useState("All");
  const [search, setSearch] = useState("");
  const debouncedSearch = useDebounced(search);
  const [products, setProducts] = useState<Product[]>([]);
  const [selected, setSelected] = useState<Product | null>(null);

  const [competitorRatio, setCompetitorRatio] = useState(1.0);
  const [inventory, setInventory] = useState(1.0);
  const [demandRatio, setDemandRatio] = useState(1.0);
  const [timeNorm, setTimeNorm] = useState(0.0);

  const [recommendation, setRecommendation] = useState<Recommendation | null>(null);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [curve, setCurve] = useState<CurvePoint[]>([]);
  const [loadingDecision, setLoadingDecision] = useState(false);

  useEffect(() => {
    api.categories().then(setCategories).catch(console.error);
  }, []);

  useEffect(() => {
    let ignore = false;
    api
      .products({ category, q: debouncedSearch || undefined, limit: 40 })
      .then((res) => {
        if (ignore) return;
        setProducts(res.items);
        setSelected((current) => {
          if (current && res.items.some((p) => p.stock_code === current.stock_code)) return current;
          return res.items[0] ?? null;
        });
      })
      .catch(console.error);
    return () => {
      ignore = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [category, debouncedSearch]);

  const state = useMemo(
    () =>
      selected
        ? {
            stock_code: selected.stock_code,
            competitor_ratio: competitorRatio,
            inventory,
            demand_ratio: demandRatio,
            time_norm: timeNorm,
          }
        : null,
    [selected, competitorRatio, inventory, demandRatio, timeNorm]
  );
  const debouncedState = useDebounced(state, 200);

  useEffect(() => {
    if (!debouncedState) return;
    let ignore = false;
    setLoadingDecision(true);
    Promise.all([
      api.recommend(debouncedState),
      api.explain(debouncedState),
      api.curve(debouncedState.stock_code, debouncedState.competitor_ratio, debouncedState.inventory),
    ])
      .then(([rec, exp, cv]) => {
        if (ignore) return;
        setRecommendation(rec);
        setExplanation(exp);
        setCurve(cv);
      })
      .catch(console.error)
      .finally(() => {
        if (!ignore) setLoadingDecision(false);
      });
    return () => {
      ignore = true;
    };
  }, [debouncedState]);

  return (
    <PageTransition>
      <div className="grid grid-cols-1 gap-5 py-10 lg:grid-cols-[300px_1fr]">
        {/* Sidebar: product list */}
        <SpotlightCard className="h-fit lg:sticky lg:top-24">
          <div className="border-b border-white/[0.07] p-4">
            <div className="relative">
              <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-dimmer)]" />
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search products..."
                className="w-full rounded-lg border border-white/10 bg-white/[0.03] py-2 pl-8 pr-3 text-[13px] text-white placeholder:text-[var(--text-dimmer)] outline-none focus:border-violet-400/50"
              />
            </div>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="mt-2.5 w-full rounded-lg border border-white/10 bg-white/[0.03] px-3 py-2 text-[13px] text-white outline-none focus:border-violet-400/50"
            >
              <option value="All">All categories</option>
              {categories.map((c) => (
                <option key={c.category} value={c.category}>
                  {c.category} ({c.count})
                </option>
              ))}
            </select>
          </div>
          <div className="max-h-[560px] overflow-y-auto p-2">
            {products.map((p) => (
              <button
                key={p.stock_code}
                onClick={() => setSelected(p)}
                className={`mb-1 flex w-full flex-col rounded-lg px-3 py-2 text-left transition-colors ${
                  selected?.stock_code === p.stock_code
                    ? "bg-gradient-to-r from-cyan-500/15 to-violet-500/15 ring-1 ring-violet-400/30"
                    : "hover:bg-white/[0.04]"
                }`}
              >
                <span className="truncate text-[12.5px] font-medium text-white">{p.description}</span>
                <span className="mt-0.5 flex items-center gap-1.5 text-[11px] text-[var(--text-dimmer)]">
                  £{p.base_price.toFixed(2)} &middot; e={p.elasticity.toFixed(2)}
                  {p.held_out_from_training && <Lock size={9} />}
                </span>
              </button>
            ))}
            {products.length === 0 && (
              <p className="p-4 text-center text-[12.5px] text-[var(--text-dimmer)]">No products match.</p>
            )}
          </div>
        </SpotlightCard>

        {/* Main panel */}
        <div className="space-y-5">
          <>
            {selected && (
              <motion.div
                key={selected.stock_code}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.25 }}
                className="space-y-5"
              >
                <SpotlightCard className="p-6">
                  <div className="flex flex-wrap items-start justify-between gap-4">
                    <div>
                      <h2 className="font-display text-xl font-semibold text-white">{selected.description}</h2>
                      <p className="mt-1 text-[12.5px] text-[var(--text-dim)]">
                        {selected.stock_code} &middot; {selected.category}{" "}
                        {selected.held_out_from_training && (
                          <span className="ml-1 inline-flex items-center gap-1 rounded-full border border-amber-400/30 bg-amber-400/10 px-2 py-0.5 text-[10px] text-amber-300">
                            <Lock size={9} /> held out from training
                          </span>
                        )}
                      </p>
                    </div>
                    <div className="grid grid-cols-3 gap-4 text-right">
                      <div>
                        <div className="text-[11px] text-[var(--text-dimmer)]">Base price</div>
                        <div className="font-mono-tech text-sm text-white">£{selected.base_price.toFixed(2)}</div>
                      </div>
                      <div>
                        <div className="text-[11px] text-[var(--text-dimmer)]">Elasticity</div>
                        <div className="font-mono-tech text-sm text-white">{selected.elasticity.toFixed(2)}</div>
                      </div>
                      <div>
                        <div className="text-[11px] text-[var(--text-dimmer)]">Transactions</div>
                        <div className="font-mono-tech text-sm text-white">{selected.n_transactions.toLocaleString()}</div>
                      </div>
                    </div>
                  </div>

                  {/* Recommendation */}
                  <div className="mt-6 flex flex-wrap items-end justify-between gap-6 rounded-xl border border-violet-400/20 bg-gradient-to-br from-violet-500/10 via-transparent to-cyan-500/10 p-5">
                    <div>
                      <div className="flex items-center gap-1.5 text-[11.5px] text-violet-300">
                        <Sparkles size={12} /> PPO recommendation
                      </div>
                      <div className="font-display mt-1 text-4xl font-semibold text-white">
                        {recommendation ? (
                          <AnimatedNumber value={recommendation.recommended_price} decimals={2} prefix="£" duration={0.5} />
                        ) : (
                          "£—"
                        )}
                      </div>
                    </div>
                    {recommendation && (
                      <div
                        className={`rounded-full px-3 py-1 text-[13px] font-semibold ${
                          recommendation.pct_vs_base >= 0
                            ? "bg-emerald-400/15 text-emerald-300"
                            : "bg-rose-400/15 text-rose-300"
                        }`}
                      >
                        {recommendation.pct_vs_base >= 0 ? "+" : ""}
                        {recommendation.pct_vs_base.toFixed(1)}% vs base
                      </div>
                    )}
                  </div>

                  {/* Sliders */}
                  <div className="mt-6 grid grid-cols-1 gap-5 sm:grid-cols-2">
                    <SliderControl label="Competitor price (× base)" value={competitorRatio} min={0.5} max={1.5} step={0.01} onChange={setCompetitorRatio} />
                    <SliderControl label="Inventory level" value={inventory} min={0} max={1} step={0.05} onChange={setInventory} />
                    <SliderControl label="Recent demand (× typical)" value={demandRatio} min={0} max={3} step={0.1} onChange={setDemandRatio} />
                    <SliderControl label="Point in pricing cycle" value={timeNorm} min={0} max={1} step={0.05} onChange={setTimeNorm} />
                  </div>
                </SpotlightCard>

                <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
                  <SpotlightCard className="p-6">
                    <h3 className="font-display text-sm font-semibold text-white">Expected profit vs. price</h3>
                    <div className="mt-4 h-56">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={curve}>
                          <XAxis
                            dataKey="price"
                            tickFormatter={(v) => `£${v.toFixed(0)}`}
                            tick={{ fill: "#8b90ac", fontSize: 10 }}
                            axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
                            tickLine={false}
                          />
                          <YAxis hide />
                          <Tooltip
                            contentStyle={{ background: "#0a0d1a", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 10, fontSize: 12 }}
                            formatter={(v) => [`£${Number(v).toFixed(2)}`, "profit"]}
                            labelFormatter={(v) => `price £${Number(v).toFixed(2)}`}
                          />
                          <Line type="monotone" dataKey="profit" stroke="#22d3ee" strokeWidth={2} dot={false} />
                          {recommendation && (
                            <ReferenceLine x={recommendation.recommended_price} stroke="#f472b6" strokeDasharray="4 4" />
                          )}
                          {selected && <ReferenceLine x={selected.base_price} stroke="#8b90ac" strokeDasharray="2 3" />}
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </SpotlightCard>

                  <SpotlightCard className="p-6">
                    <h3 className="font-display text-sm font-semibold text-white">Why this price? (SHAP)</h3>
                    <p className="mt-1 text-[11.5px] text-[var(--text-dimmer)]">
                      base {explanation?.base_value.toFixed(3)} &rarr; decision {explanation?.prediction.toFixed(3)}
                    </p>
                    <div className="mt-4 space-y-2.5">
                      {explanation?.features
                        .slice()
                        .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
                        .map((f) => {
                          const maxAbs = Math.max(...explanation.features.map((x) => Math.abs(x.shap_value)), 0.001);
                          const widthPct = (Math.abs(f.shap_value) / maxAbs) * 100;
                          const positive = f.shap_value >= 0;
                          return (
                            <div key={f.name}>
                              <div className="mb-1 flex items-center justify-between text-[11.5px]">
                                <span className="text-[var(--text-dim)]">{f.label}</span>
                                <span className={`font-mono-tech ${positive ? "text-cyan-300" : "text-pink-300"}`}>
                                  {positive ? "+" : ""}
                                  {f.shap_value.toFixed(4)}
                                </span>
                              </div>
                              <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/[0.05]">
                                <motion.div
                                  initial={{ width: 0 }}
                                  animate={{ width: `${widthPct}%` }}
                                  transition={{ duration: 0.5, ease: "easeOut" }}
                                  className={`h-full rounded-full ${
                                    positive ? "bg-gradient-to-r from-cyan-400 to-cyan-300" : "bg-gradient-to-r from-pink-400 to-pink-300"
                                  }`}
                                />
                              </div>
                            </div>
                          );
                        })}
                    </div>
                  </SpotlightCard>
                </div>
              </motion.div>
            )}
          </>
          {loadingDecision && (
            <div className="fixed bottom-6 right-6 flex items-center gap-2 rounded-full border border-white/10 bg-[#0a0d1a]/90 px-3.5 py-2 text-[11.5px] text-[var(--text-dim)] backdrop-blur">
              <span className="h-1.5 w-1.5 animate-ping rounded-full bg-violet-400" />
              recomputing policy decision…
            </div>
          )}
        </div>
      </div>
    </PageTransition>
  );
}

function SliderControl({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="mb-2 flex items-center justify-between text-[12.5px]">
        <span className="text-[var(--text-dim)]">{label}</span>
        <span className="font-mono-tech text-white">{value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full"
      />
    </div>
  );
}
