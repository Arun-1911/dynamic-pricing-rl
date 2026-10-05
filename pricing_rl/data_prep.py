"""
Data preparation for the dynamic pricing RL project.

Loads the raw UCI "Online Retail II" transaction log, cleans it, derives a
rough product category from free-text descriptions, aggregates transactions
to weekly (product, week) observations, and fits a per-product price
elasticity of demand via log-log regression on those real observations.

Output: data/products.csv, one row per real product, with:
    stock_code, description, category, base_price, elasticity,
    elasticity_source (fitted | category_fallback), base_demand,
    n_transactions, n_distinct_prices, n_weekly_obs, r2
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_XLSX = DATA_DIR / "online_retail_II.xlsx"
OUT_CSV = DATA_DIR / "products.csv"

# Ordered keyword -> category rules. First match wins. This is a heuristic
# label derived from free-text Description, not a ground-truth taxonomy —
# the dataset has no category column.
CATEGORY_RULES = [
    ("Christmas & Seasonal", r"CHRISTMAS|XMAS|ADVENT|EASTER|HALLOWEEN|SANTA|REINDEER"),
    ("Lighting & Candles", r"LIGHT|CANDLE|LANTERN|T-LIGHT|LAMP"),
    ("Bags", r"\bBAG\b|BACKPACK|HANDBAG|SHOPPER"),
    ("Jewellery & Accessories", r"NECKLACE|BRACELET|EARRING|JEWEL|RING\b|BROOCH"),
    ("Kitchen & Dining", r"MUG|CUP\b|PLATE|BOWL|KITCHEN|TEAPOT|CUTLERY|APRON|BAKING|CAKE\s?TIN"),
    ("Home Decor", r"FRAME|CUSHION|CLOCK|SIGN\b|MIRROR|VASE|ORNAMENT|DECORATION"),
    ("Cards & Stationery", r"CARD\b|NOTEBOOK|STICKER|PEN\b|GIFT\s?WRAP|RIBBON|ENVELOPE"),
    ("Garden & Outdoor", r"GARDEN|PLANT\s?POT|BIRD\s?HOUSE|WATERING"),
    ("Toys & Games", r"TOY\b|GAME\b|PUZZLE|DOLL\b"),
    ("Bath & Body", r"SOAP|BATH|CANDLE\s?SET|TOWEL"),
    ("Storage & Boxes", r"\bBOX\b|STORAGE|TIN\b|BASKET"),
    ("Bottles & Glassware", r"BOTTLE|GLASS\b|JAR\b"),
]


def derive_category(description: str) -> str:
    desc = description.upper()
    for category, pattern in CATEGORY_RULES:
        if re.search(pattern, desc):
            return category
    return "General Giftware"


def load_raw() -> pd.DataFrame:
    sheets = pd.read_excel(RAW_XLSX, sheet_name=None, engine="openpyxl")
    frames = []
    for name, df in sheets.items():
        df["source_sheet"] = name
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    for c in ["Invoice", "StockCode", "Description", "Country"]:
        full[c] = full[c].astype(str)
    return full


def clean(full: pd.DataFrame) -> pd.DataFrame:
    is_cancel = full["Invoice"].str.startswith("C")
    df = full[(~is_cancel) & (full["Quantity"] > 0) & (full["Price"] > 0)].copy()
    # Keep only real products: StockCode starting with 5 digits (excludes
    # POST, BANK CHARGES, ADJUST, TEST001, gift-card/manual codes, etc.)
    df = df[df["StockCode"].str.match(r"^\d{5}")].copy()

    # Per-product outlier clipping: a handful of one-off wholesale lines have
    # prices 10-100x a product's typical price and would distort elasticity
    # fits. Clip to each product's own [1st, 99th] percentile price.
    def clip_group(g):
        lo, hi = g["Price"].quantile([0.01, 0.99])
        return g[(g["Price"] >= lo) & (g["Price"] <= hi)]

    df = df.groupby("StockCode", group_keys=False)[df.columns.tolist()].apply(clip_group)
    return df


def fit_elasticity(weekly: pd.DataFrame) -> pd.DataFrame:
    """Fit log(qty) = a + b*log(price) per product. Returns per-product fit stats."""
    records = []
    for code, g in weekly.groupby("StockCode"):
        n_obs = len(g)
        n_distinct = g["price"].nunique()
        price_range_ratio = float(g["price"].max() / g["price"].min()) if g["price"].min() > 0 else 1.0

        # Require real price movement (>=10%) before trusting a fit — with
        # near-constant price, log-log slope estimates are numerically
        # unstable and can blow up to physically implausible magnitudes.
        can_fit = n_distinct >= 2 and n_obs >= 3 and price_range_ratio >= 1.10

        if can_fit:
            X = np.log(g["price"].values).reshape(-1, 1)
            y = np.log(g["qty"].values)
            reg = LinearRegression().fit(X, y)
            elasticity = float(reg.coef_[0])
            r2 = float(reg.score(X, y))
            base_price = float(g["price"].median())
            pred_log_demand = reg.predict([[np.log(base_price)]])[0]
            base_demand = float(np.exp(pred_log_demand))
            # Sanity-bound: retail price elasticities essentially never
            # exceed magnitude 10 in absolute value. Anything past that is a
            # numerical artifact of a poorly-conditioned fit, not a real
            # demand response — fall back to category median instead.
            if -10.0 <= elasticity <= -0.05:
                records.append((code, elasticity, r2, base_price, base_demand, n_obs, n_distinct, "fitted"))
                continue

        base_price = float(g["price"].median())
        base_demand = float(g["qty"].mean())
        source = "insufficient_price_variation" if not can_fit else "implausible_fit"
        records.append((code, np.nan, np.nan, base_price, base_demand, n_obs, n_distinct, source))
    return pd.DataFrame(records, columns=[
        "stock_code", "elasticity", "r2", "base_price", "base_demand",
        "n_weekly_obs", "n_distinct_prices", "elasticity_source"
    ])


def main():
    print("Loading raw transactions...")
    full = load_raw()
    print(f"Raw rows: {len(full):,}")

    print("Cleaning...")
    df = clean(full)
    print(f"Clean product rows: {len(df):,}, unique products: {df['StockCode'].nunique():,}")

    print("Deriving categories from Description...")
    desc_map = df.groupby("StockCode")["Description"].agg(lambda s: s.mode().iat[0])
    category = desc_map.apply(derive_category)

    print("Aggregating to weekly (product, week) observations...")
    df["week"] = df["InvoiceDate"].dt.to_period("W").dt.start_time
    weekly = df.groupby(["StockCode", "week"]).apply(
        lambda g: pd.Series({
            "price": np.average(g["Price"], weights=g["Quantity"]),
            "qty": g["Quantity"].sum(),
        }),
        include_groups=False,
    ).reset_index()

    print("Fitting per-product elasticity (log-log regression on real price/quantity)...")
    fits = fit_elasticity(weekly)

    tx_counts = df.groupby("StockCode").size().rename("n_transactions")
    products = fits.set_index("stock_code").join(desc_map.rename("description")).join(category.rename("category")).join(tx_counts)
    products = products.reset_index().rename(columns={"index": "stock_code"})

    # Category-level fallback elasticity for products whose own fit was
    # rejected (insufficient price variation or an implausible slope).
    fitted_mask = products["elasticity_source"] == "fitted"
    category_elasticity = products.loc[fitted_mask].groupby("category")["elasticity"].median()
    overall_median = products.loc[fitted_mask, "elasticity"].median()

    needs_fallback = ~fitted_mask
    products.loc[needs_fallback, "elasticity_source"] = "category_fallback_" + products.loc[needs_fallback, "elasticity_source"]
    products.loc[needs_fallback, "elasticity"] = products.loc[needs_fallback, "category"].map(category_elasticity).fillna(overall_median)

    products = products.sort_values("n_transactions", ascending=False).reset_index(drop=True)

    products.to_csv(OUT_CSV, index=False)

    print(f"\nSaved {len(products):,} products to {OUT_CSV}")
    print("\nelasticity_source breakdown:")
    print(products["elasticity_source"].value_counts())
    print("\nelasticity stats:")
    print(products["elasticity"].describe())
    print("\ncategory breakdown:")
    print(products["category"].value_counts())


if __name__ == "__main__":
    main()
