import numpy as np
import pandas as pd
import pytest

import data_prep


def test_product_table_is_complete(products):
    assert products.stock_code.is_unique
    core = ["stock_code", "description", "category", "base_price", "elasticity",
            "base_demand", "n_transactions", "elasticity_source"]
    assert not products[core].isna().any().any()
    assert (products.base_price > 0).all()
    assert (products.base_demand > 0).all()
    assert products.elasticity.between(-10, -0.05).all()


def test_elasticity_sources_are_consistent(products):
    allowed = {"fitted", "category_fallback_insufficient_price_variation",
               "category_fallback_implausible_fit"}
    assert set(products.elasticity_source) <= allowed
    fitted = products[products.elasticity_source == "fitted"]
    assert fitted.r2.notna().all()
    assert products[products.elasticity_source != "fitted"].r2.isna().all()


def test_train_test_split_is_clean(products, train_df, test_df):
    assert set(train_df.stock_code).isdisjoint(test_df.stock_code)
    assert set(train_df.stock_code) | set(test_df.stock_code) == set(products.stock_code)
    assert len(test_df) / len(products) == pytest.approx(0.15, abs=0.005)


def test_category_rules():
    assert data_prep.derive_category("WHITE HANGING HEART T-LIGHT HOLDER") == "Lighting & Candles"
    assert data_prep.derive_category("JUMBO BAG RED RETROSPOT") == "Bags"
    assert data_prep.derive_category("CHRISTMAS TREE DECORATION") == "Christmas & Seasonal"
    assert data_prep.derive_category("SOMETHING THAT MATCHES NO RULE") == "General Giftware"


def _tx(rows):
    return pd.DataFrame(rows, columns=["Invoice", "StockCode", "Description", "Quantity",
                                       "InvoiceDate", "Price", "Customer ID", "Country"])


def test_clean_drops_cancellations_services_and_bad_rows():
    date = pd.Timestamp("2011-01-03")
    good = [(f"5000{i}", "12345", "WIDGET", 2, date, 1.0 + i * 0.01, 1.0, "UK") for i in range(100)]
    bad = [
        ("C9001", "12345", "WIDGET", -2, date, 1.5, 1.0, "UK"),   # cancellation
        ("5100", "POST", "POSTAGE", 1, date, 15.0, 1.0, "UK"),    # service code
        ("5101", "12345", "WIDGET", 1, date, 0.0, 1.0, "UK"),     # zero price
        ("5102", "12345", "WIDGET", -1, date, 1.2, 1.0, "UK"),    # negative quantity
    ]
    out = data_prep.clean(_tx(good + bad))
    assert set(out.StockCode) == {"12345"}
    assert not out.Invoice.str.startswith("C").any()
    assert (out.Price > 0).all() and (out.Quantity > 0).all()


def test_fit_elasticity_recovers_a_known_curve():
    price = np.linspace(1.0, 2.0, 30)
    weekly = pd.DataFrame({"StockCode": "A", "price": price, "qty": 500 * price ** -2.0})
    row = data_prep.fit_elasticity(weekly).iloc[0]
    assert row.elasticity_source == "fitted"
    assert row.elasticity == pytest.approx(-2.0, abs=1e-6)
    assert row.r2 == pytest.approx(1.0, abs=1e-9)


def test_fit_elasticity_rejects_unusable_fits():
    flat = pd.DataFrame({"StockCode": "B", "price": [2.0] * 10, "qty": np.arange(10) + 5.0})
    upward = pd.DataFrame({"StockCode": "C", "price": np.linspace(1, 2, 20), "qty": np.linspace(10, 40, 20)})
    flat_row = data_prep.fit_elasticity(flat).iloc[0]
    up_row = data_prep.fit_elasticity(upward).iloc[0]
    assert flat_row.elasticity_source == "insufficient_price_variation" and np.isnan(flat_row.elasticity)
    assert up_row.elasticity_source == "implausible_fit" and np.isnan(up_row.elasticity)
