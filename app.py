"""
Self-contained Streamlit app for Streamlit Community Cloud.
Fetches live data directly — no local database or backend process needed.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Correlation Breakdown Detector",
    page_icon="📊",
    layout="wide",
)

# ── Asset Universe ────────────────────────────────────────────────────────────

ASSETS: Dict[str, List[str]] = {
    "equities":    ["SPY", "QQQ", "IWM", "EEM", "XLF", "XLE", "XLK"],
    "bonds":       ["TLT", "IEF", "HYG", "LQD", "TIP"],
    "commodities": ["GLD", "SLV", "USO", "DBA"],
    "currencies":  ["UUP", "FXE", "FXY"],
    "crypto":      ["BTC-USD", "ETH-USD", "SOL-USD"],
}

CLASS_MAP: Dict[str, str] = {
    sym: cls for cls, syms in ASSETS.items() for sym in syms
}

ALL_SYMBOLS = [s for syms in ASSETS.values() for s in syms]

HEDGE_RULES: Dict[Tuple, List[Dict]] = {
    ("bonds", "equities", "positive_spike"): [
        {"instrument": "VIXY", "direction": "long", "weight": 0.08,
         "rationale": "VIX futures ETF profits when stocks and bonds fall together."},
        {"instrument": "GLD",  "direction": "long", "weight": 0.10,
         "rationale": "Gold acts as safe haven when bonds lose their equity hedge."},
        {"instrument": "SH",   "direction": "long", "weight": 0.08,
         "rationale": "Inverse S&P 500 ETF provides direct equity downside protection."},
    ],
    ("bonds", "equities", "negative_spike"): [
        {"instrument": "TLT", "direction": "long", "weight": 0.12,
         "rationale": "Add duration — Treasuries rallying hard vs equities signals flight to quality."},
        {"instrument": "UUP", "direction": "long", "weight": 0.05,
         "rationale": "USD typically strengthens in risk-off environments."},
    ],
    ("crypto", "equities", "positive_spike"): [
        {"instrument": "BITI", "direction": "long", "weight": 0.04,
         "rationale": "Short Bitcoin ETF hedges crypto downside when crypto follows equities lower."},
    ],
    ("commodities", "equities", "negative_spike"): [
        {"instrument": "PDBC", "direction": "long", "weight": 0.10,
         "rationale": "Broad commodity ETF captures inflation-driven rally vs equity weakness."},
        {"instrument": "GLD",  "direction": "long", "weight": 0.08,
         "rationale": "Gold benefits from inflation regime driving commodity-equity divergence."},
    ],
    ("commodities", "equities", "positive_spike"): [
        {"instrument": "SH", "direction": "long", "weight": 0.07,
         "rationale": "Inverse S&P when deflation fears pull both equities and commodities down."},
    ],
    ("bonds", "commodities", "positive_spike"): [
        {"instrument": "TIP", "direction": "long", "weight": 0.10,
         "rationale": "TIPS benefit when commodities and bonds move together on inflation."},
    ],
    ("bonds", "crypto", "positive_spike"): [
        {"instrument": "SHY",  "direction": "long", "weight": 0.08,
         "rationale": "Short-duration Treasuries as defence; crypto+bond sell-off signals deleveraging."},
        {"instrument": "BITI", "direction": "long", "weight": 0.04,
         "rationale": "Short Bitcoin ETF to hedge crypto-bond correlation spike."},
    ],
}

SEVERITY = [(4.0, "extreme"), (3.0, "severe"), (2.0, "moderate"), (0.0, "mild")]
SEV_COLOR = {"extreme": "#9C27B0", "severe": "#F44336", "moderate": "#FF9800", "mild": "#FFC107"}


# ── Data fetching ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_prices(symbols: tuple, days: int) -> pd.DataFrame:
    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    df = yf.download(list(symbols), start=start, auto_adjust=True, progress=False, threads=True)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    else:
        df = df[["Close"]].rename(columns={"Close": symbols[0]})
    return df.dropna(how="all")


# ── Correlation helpers ───────────────────────────────────────────────────────

def rolling_corr_matrix(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    returns = prices.pct_change().dropna(how="all")
    tail = returns.tail(window)
    return tail.corr().clip(-1.0, 1.0)


def pair_map(corr: pd.DataFrame) -> Dict[str, float]:
    cols = corr.columns.tolist()
    return {
        f"{cols[i]}|{cols[j]}": float(corr.iloc[i, j])
        for i in range(len(cols)) for j in range(i + 1, len(cols))
        if not np.isnan(corr.iloc[i, j])
    }


def baseline_stats(prices: pd.DataFrame, short_w: int, long_w: int) -> Dict[str, Dict]:
    returns = prices.pct_change().dropna(how="all")
    stats: Dict[str, Dict] = {}
    cols = returns.columns.tolist()
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j <= i:
                continue
            key = f"{a}|{b}"
            series = returns[a].rolling(short_w).corr(returns[b]).dropna()
            if len(series) < 10:
                continue
            stats[key] = {"mean": float(series.mean()), "std": float(series.std())}
    return stats


def detect_breakdowns(cur: Dict[str, float], stats: Dict[str, Dict], threshold: float) -> List[Dict]:
    bds = []
    for key, corr in cur.items():
        if key not in stats:
            continue
        mean, std = stats[key]["mean"], stats[key]["std"]
        if std < 1e-8:
            continue
        z = (corr - mean) / std
        if abs(z) < threshold:
            continue
        a1, a2 = key.split("|")
        sev = next(s for thresh, s in SEVERITY if abs(z) >= thresh)
        bds.append({
            "pair": f"{a1} / {a2}", "pair_key": key,
            "asset1": a1, "asset2": a2,
            "class1": CLASS_MAP.get(a1, "other"), "class2": CLASS_MAP.get(a2, "other"),
            "corr": round(corr, 3), "mean": round(mean, 3), "std": round(std, 3),
            "z": round(z, 2), "direction": "positive_spike" if z > 0 else "negative_spike",
            "severity": sev,
        })
    return sorted(bds, key=lambda x: abs(x["z"]), reverse=True)


def get_hedges(bd: Dict) -> List[Dict]:
    c1, c2 = sorted([bd["class1"], bd["class2"]])
    key = (c1, c2, bd["direction"])
    rules = HEDGE_RULES.get(key, [])
    if not rules:
        return [{"instrument": "GLD", "direction": "long", "weight": 0.05,
                 "rationale": f"Gold as generic diversifier for {bd['pair']} breakdown (z={bd['z']:.2f})."}]
    scale = {"mild": 0.5, "moderate": 0.8, "severe": 1.0, "extreme": 1.3}[bd["severity"]]
    return [{**r, "weight": round(min(r["weight"] * scale, 0.25), 3)} for r in rules]


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 Correlation Detector")
    st.caption("Live data · No database · Updates every 5 min")
    st.divider()

    history_days = st.slider("History (days)", 60, 365, 180)
    short_window = st.slider("Rolling window (days)", 10, 60, 30)
    long_window = st.slider("Baseline window (days)", 60, 252, 120)
    z_threshold = st.slider("Z-score threshold", 1.0, 4.0, 2.0, 0.25)

    selected_class = st.selectbox(
        "Filter asset class", ["All"] + list(ASSETS.keys())
    )
    st.divider()
    st.caption(f"Data cached for 5 min · Last load: {datetime.now().strftime('%H:%M:%S')}")
    if st.button("🔄 Force Refresh", width="stretch"):
        st.cache_data.clear()
        st.rerun()

# ── Load data ─────────────────────────────────────────────────────────────────

with st.spinner("Fetching live prices..."):
    prices = fetch_prices(tuple(ALL_SYMBOLS), history_days + long_window)

if prices.empty:
    st.error("Could not fetch price data. Check your internet connection.")
    st.stop()

# Filter by class
if selected_class != "All":
    keep = [s for s in prices.columns if CLASS_MAP.get(s) == selected_class]
    display_prices = prices[keep] if keep else prices
else:
    display_prices = prices

available = display_prices.dropna(axis=1, thresh=short_window).columns.tolist()
if len(available) < 2:
    st.error("Not enough assets with data. Try widening the history window.")
    st.stop()

with st.spinner("Computing correlations..."):
    corr = rolling_corr_matrix(display_prices[available], short_window)
    cur_pairs = pair_map(corr)
    stats = baseline_stats(display_prices[available], short_window, long_window)
    breakdowns = detect_breakdowns(cur_pairs, stats, z_threshold)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📡 Live Heatmap", "⚠️ Breakdowns", "🛡️ Hedge Recommendations"])


# ══ TAB 1 — HEATMAP ══════════════════════════════════════════════════════════

with tab1:
    st.header(f"{short_window}-Day Rolling Correlation Matrix")
    st.caption(f"{len(available)} assets · data through {prices.index[-1].date()}")

    col_heat, col_stats = st.columns([3, 1])

    with col_heat:
        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=[[0, "#d73027"], [0.5, "#ffffbf"], [1, "#1a9850"]],
            zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
        ))
        fig.update_layout(
            height=560, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="#fafafa", size=10),
        )
        st.plotly_chart(fig, width="stretch")

    with col_stats:
        upper_vals = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().values
        st.metric("Assets", len(available))
        st.metric("Avg Correlation", f"{upper_vals.mean():.3f}")
        st.metric("Max Correlation", f"{upper_vals.max():.3f}")
        st.metric("Min Correlation", f"{upper_vals.min():.3f}")
        st.metric("Active Breakdowns", len(breakdowns))

        st.divider()
        st.subheader("Extreme Pairs")
        upper_copy = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).copy()
        upper_copy.index.name = "A"
        upper_copy.columns.name = "B"
        ep = upper_copy.stack().reset_index()
        ep.columns = ["Asset 1", "Asset 2", "Corr"]
        ep = ep.reindex(ep["Corr"].abs().sort_values(ascending=False).index).head(6)
        st.dataframe(ep.style.format({"Corr": "{:.3f}"}), hide_index=True, width="stretch")

    # Cross-class heatmap
    st.subheader("Cross-Asset Class Average Correlation")
    class_list = list(ASSETS.keys())
    returns_all = display_prices[available].pct_change().tail(short_window)
    rows = []
    for c1 in class_list:
        row = []
        for c2 in class_list:
            s1 = [s for s in available if CLASS_MAP.get(s) == c1]
            s2 = [s for s in available if CLASS_MAP.get(s) == c2]
            if not s1 or not s2 or c1 == c2:
                row.append(1.0 if c1 == c2 else float("nan"))
            else:
                vals = [returns_all[a].corr(returns_all[b]) for a in s1 for b in s2 if a in returns_all and b in returns_all]
                row.append(float(np.nanmean(vals)) if vals else float("nan"))
        rows.append(row)
    class_df = pd.DataFrame(rows, index=class_list, columns=class_list)
    fig2 = px.imshow(class_df, color_continuous_scale=["#d73027", "#ffffbf", "#1a9850"],
                     zmin=-1, zmax=1, text_auto=".2f", height=320)
    fig2.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font=dict(color="#fafafa"))
    st.plotly_chart(fig2, width="stretch")


# ══ TAB 2 — BREAKDOWNS ═══════════════════════════════════════════════════════

with tab2:
    st.header("Active Correlation Breakdowns")

    c1, c2, c3, c4 = st.columns(4)
    for col, sev in zip([c1, c2, c3, c4], ["extreme", "severe", "moderate", "mild"]):
        col.metric(sev.capitalize(), sum(1 for b in breakdowns if b["severity"] == sev))

    if not breakdowns:
        st.success(f"No breakdowns detected above |z| > {z_threshold}.")
    else:
        bd_df = pd.DataFrame([{
            "Pair": b["pair"], "Class 1": b["class1"], "Class 2": b["class2"],
            "Current Corr": b["corr"], "Baseline": b["mean"],
            "Z-Score": b["z"], "Direction": b["direction"], "Severity": b["severity"],
        } for b in breakdowns])

        def sev_color(val):
            return f"background-color: {SEV_COLOR.get(val, '')}; color: white"

        st.dataframe(
            bd_df.style.map(sev_color, subset=["Severity"])
                .format({"Current Corr": "{:.3f}", "Baseline": "{:.3f}", "Z-Score": "{:+.2f}"}),
            hide_index=True, width="stretch",
        )

        fig_bar = px.bar(
            bd_df.head(20), x="Pair", y="Z-Score", color="Severity",
            color_discrete_map=SEV_COLOR,
            title="Breakdown Z-Scores (ranked by severity)",
            height=380,
        )
        fig_bar.add_hline(y=z_threshold, line_dash="dash", line_color="white",
                          annotation_text=f"Threshold ±{z_threshold}")
        fig_bar.add_hline(y=-z_threshold, line_dash="dash", line_color="white")
        fig_bar.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                              font=dict(color="#fafafa"))
        st.plotly_chart(fig_bar, width="stretch")


# ══ TAB 3 — HEDGES ═══════════════════════════════════════════════════════════

with tab3:
    st.header("Hedge Recommendations")
    st.caption("Positions to neutralize current breakdown risk, scaled by severity.")

    if not breakdowns:
        st.success("No active breakdowns — no hedges needed.")
    else:
        # Aggregate hedges across all breakdowns
        agg: Dict[str, Dict] = {}
        for bd in breakdowns[:10]:
            for h in get_hedges(bd):
                inst = h["instrument"]
                if inst not in agg:
                    agg[inst] = {"Instrument": inst, "Direction": h["direction"],
                                 "Weight (%)": 0.0, "Triggered By": [], "Rationale": h["rationale"]}
                agg[inst]["Weight (%)"] += h["weight"] * 100
                agg[inst]["Triggered By"].append(bd["pair"])

        hedge_df = pd.DataFrame(agg.values())
        hedge_df["Weight (%)"] = hedge_df["Weight (%)"].clip(upper=25).round(2)
        hedge_df["Triggered By"] = hedge_df["Triggered By"].apply(lambda x: ", ".join(set(x)))
        hedge_df = hedge_df.sort_values("Weight (%)", ascending=False)

        m1, m2, m3 = st.columns(3)
        m1.metric("Hedge Instruments", len(hedge_df))
        m2.metric("Avg Weight", f"{hedge_df['Weight (%)'].mean():.1f}%")
        m3.metric("Total Breakdowns Hedged", len(breakdowns))

        st.dataframe(
            hedge_df[["Instrument", "Direction", "Weight (%)", "Triggered By"]],
            hide_index=True, width="stretch",
        )

        fig_h = px.bar(
            hedge_df, x="Instrument", y="Weight (%)", color="Direction",
            color_discrete_map={"long": "#2196F3", "short": "#F44336"},
            title="Suggested Hedge Weights by Instrument",
            height=350,
        )
        fig_h.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                            font=dict(color="#fafafa"))
        st.plotly_chart(fig_h, width="stretch")

        with st.expander("Rationale for each hedge"):
            for _, row in hedge_df.iterrows():
                st.markdown(f"**{row['Instrument']}** ({row['Direction'].upper()} "
                            f"{row['Weight (%)']:.1f}%): {row['Rationale']}")
