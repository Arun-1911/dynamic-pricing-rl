const BASE = "http://127.0.0.1:8000";

export interface Product {
  stock_code: string;
  description: string;
  category: string;
  base_price: number;
  elasticity: number;
  elasticity_source: string;
  n_transactions: number;
  held_out_from_training: boolean;
}

export interface Summary {
  n_products: number;
  n_train: number;
  n_test: number;
  category_counts: Record<string, number>;
  elasticity_source_counts: Record<string, number>;
  elasticity_mean: number;
  elasticity_median: number;
  elasticity_std: number;
  elasticity_histogram: { bin_start: number; bin_end: number; count: number }[];
  uplift_report: UpliftReport;
}

export interface UpliftReport {
  ppo: { total_revenue: number; total_profit: number; mean_revenue_per_product: number; supply_capped_share: number };
  static_base_price: BaselineResult;
  undercut_competitor_5pct: BaselineResult;
  elasticity_optimal_static: BaselineResult;
  tuned_static_oracle: BaselineResult;
  n_held_out_products: number;
  episodes_per_product: number;
}

interface BaselineResult {
  total_revenue: number;
  total_profit: number;
  mean_revenue_per_product: number;
  supply_capped_share: number;
  revenue_uplift_vs_ppo_pct: number;
  profit_uplift_vs_ppo_pct: number;
  revenue_uplift_ci95_pct: [number, number];
  profit_uplift_ci95_pct: [number, number];
}

export interface StateInput {
  stock_code: string;
  competitor_ratio: number;
  inventory: number;
  demand_ratio: number;
  time_norm: number;
}

export interface Recommendation {
  stock_code: string;
  recommended_price: number;
  price_ratio: number;
  base_price: number;
  pct_vs_base: number;
}

export interface ExplainFeature {
  name: string;
  label: string;
  value: number;
  shap_value: number;
}

export interface Explanation {
  stock_code: string;
  features: ExplainFeature[];
  base_value: number;
  prediction: number;
}

export interface CurvePoint {
  price: number;
  ratio: number;
  profit: number;
  revenue: number;
}

export interface LearningPoint {
  timestep: number;
  episode_reward: number;
  episode_length: number;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
  return res.json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`POST ${path} failed: ${res.status}`);
  return res.json();
}

export const api = {
  summary: () => get<Summary>("/api/summary"),
  categories: () => get<{ category: string; count: number }[]>("/api/categories"),
  products: (params: { category?: string; q?: string; limit?: number; offset?: number }) => {
    const qs = new URLSearchParams();
    if (params.category) qs.set("category", params.category);
    if (params.q) qs.set("q", params.q);
    qs.set("limit", String(params.limit ?? 50));
    qs.set("offset", String(params.offset ?? 0));
    return get<{ total: number; items: Product[] }>(`/api/products?${qs}`);
  },
  productDetail: (stockCode: string) => get<Product & Record<string, unknown>>(`/api/products/${stockCode}`),
  recommend: (state: StateInput) => post<Recommendation>("/api/recommend", state),
  explain: (state: StateInput) => post<Explanation>("/api/explain", state),
  curve: (stockCode: string, competitorRatio: number, inventory: number) =>
    get<CurvePoint[]>(
      `/api/products/${stockCode}/curve?competitor_ratio=${competitorRatio}&inventory=${inventory}`
    ),
  learningCurve: (maxPoints = 200) => get<LearningPoint[]>(`/api/training/learning_curve?max_points=${maxPoints}`),
  uplift: () => get<UpliftReport>("/api/training/uplift"),
};
