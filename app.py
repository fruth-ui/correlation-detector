"""
Real-Time Correlation Breakdown Detector — Streamlit Cloud edition.
Self-contained: fetches live data from Yahoo Finance, no database needed.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

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
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Asset Universe ─────────────────────────────────────────────────────────────

ASSETS: Dict[str, List[str]] = {
    "equities":    ["SPY", "QQQ", "IWM", "EEM", "VEA", "XLF", "XLE", "XLK", "XLV", "XLU"],
    "bonds":       ["TLT", "IEF", "SHY", "HYG", "LQD", "TIP"],
    "commodities": ["GLD", "SLV", "USO", "DBA"],
    "currencies":  ["UUP", "FXE", "FXY"],
    "crypto":      ["BTC-USD", "ETH-USD", "SOL-USD"],
}

# Hedge instruments fetched alongside main universe for backtesting
HEDGE_SYMBOLS = ["VIXY", "SH", "BITI", "PDBC", "UUP", "TIP", "SHY", "TLT", "GLD"]

CLASS_MAP: Dict[str, str] = {s: c for c, syms in ASSETS.items() for s in syms}
ALL_SYMBOLS = [s for syms in ASSETS.values() for s in syms]

SEVERITY_LEVELS = [(4.0, "extreme"), (3.0, "severe"), (2.0, "moderate"), (0.0, "mild")]
SEV_COLOR = {"extreme": "#9C27B0", "severe": "#F44336", "moderate": "#FF9800", "mild": "#FFC107"}

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
        {"instrument": "SPY_PUT", "direction": "long", "weight": 0.03,
         "rationale": "SPY puts provide tail-risk protection; crypto risk-off signals equity weakness."},
    ],
    ("crypto", "equities", "negative_spike"): [
        {"instrument": "BITX", "direction": "long", "weight": 0.03,
         "rationale": "2x BTC ETF captures crypto alpha when it decouples positively from equities."},
    ],
    ("commodities", "equities", "negative_spike"): [
        {"instrument": "PDBC", "direction": "long", "weight": 0.10,
         "rationale": "Broad commodity ETF captures inflation-driven rally vs equity weakness."},
        {"instrument": "XLE",  "direction": "long", "weight": 0.07,
         "rationale": "Energy sector outperforms when oil rallies while equities lag."},
        {"instrument": "GLD",  "direction": "long", "weight": 0.08,
         "rationale": "Gold benefits from inflation regime driving commodity-equity divergence."},
    ],
    ("commodities", "equities", "positive_spike"): [
        {"instrument": "SH", "direction": "long", "weight": 0.07,
         "rationale": "Inverse S&P when deflation fears pull both equities and commodities down."},
    ],
    ("currencies", "equities", "positive_spike"): [
        {"instrument": "UUP", "direction": "long", "weight": 0.08,
         "rationale": "USD strengthens in risk-off; hedges FX volatility correlated with equity sell-off."},
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

# ── Data helpers ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_prices(symbols: tuple, days: int) -> pd.DataFrame:
    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    df = yf.download(list(symbols), start=start, auto_adjust=True, progress=False, threads=True)
    if df.empty:
        return pd.DataFrame()
    close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]].rename(columns={"Close": symbols[0]})
    return close.dropna(how="all")


def rolling_corr_matrix(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    returns = prices.pct_change().dropna(how="all")
    return returns.tail(window).corr().clip(-1.0, 1.0)


def get_pair_map(corr: pd.DataFrame) -> Dict[str, float]:
    cols = corr.columns.tolist()
    return {
        f"{cols[i]}|{cols[j]}": float(corr.iloc[i, j])
        for i in range(len(cols)) for j in range(i + 1, len(cols))
        if not np.isnan(corr.iloc[i, j])
    }


def compute_baseline(prices: pd.DataFrame, short_w: int, long_w: int) -> Dict[str, Dict]:
    returns = prices.pct_change().dropna(how="all")
    stats: Dict[str, Dict] = {}
    cols = returns.columns.tolist()
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j <= i:
                continue
            s = returns[a].rolling(short_w).corr(returns[b]).dropna()
            if len(s) >= 10:
                stats[f"{a}|{b}"] = {"mean": float(s.mean()), "std": float(s.std())}
    return stats


def detect_breakdowns(pairs: Dict[str, float], stats: Dict[str, Dict], threshold: float) -> List[Dict]:
    bds = []
    for key, corr in pairs.items():
        if key not in stats:
            continue
        mean, std = stats[key]["mean"], stats[key]["std"]
        if std < 1e-8:
            continue
        z = (corr - mean) / std
        if abs(z) < threshold:
            continue
        a1, a2 = key.split("|")
        sev = next(s for t, s in SEVERITY_LEVELS if abs(z) >= t)
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
    rules = HEDGE_RULES.get((c1, c2, bd["direction"]), [])
    if not rules:
        return [{"instrument": "GLD", "direction": "long", "weight": 0.05,
                 "rationale": f"Gold as generic diversifier for {bd['pair']} breakdown (z={bd['z']:.2f})."}]
    scale = {"mild": 0.5, "moderate": 0.8, "severe": 1.0, "extreme": 1.3}[bd["severity"]]
    return [{**r, "weight": round(min(r["weight"] * scale, 0.25), 3)} for r in rules]


def safe_ret(series: pd.Series, key: str) -> float:
    """Get a return value, returning 0.0 for missing or NaN."""
    val = series.get(key, 0.0)
    return 0.0 if (val is None or (isinstance(val, float) and np.isnan(val))) else float(val)


def run_backtest(
    prices: pd.DataFrame, hedge_prices: pd.DataFrame, benchmark: str,
    short_w: int, long_w: int, z_thresh: float,
    hold_days: int = 10, rebal_days: int = 5,
) -> Dict:
    # Merge main prices with hedge instrument prices for return lookup
    all_prices = pd.concat([prices, hedge_prices], axis=1)
    all_prices = all_prices.loc[:, ~all_prices.columns.duplicated()]

    rets = all_prices.pct_change()
    # Only use rows where benchmark has data
    if benchmark not in rets.columns:
        benchmark = prices.columns[0]
    rets = rets[rets[benchmark].notna()]

    dates = rets.index.tolist()
    if len(dates) < long_w + short_w + 5:
        raise ValueError(f"Not enough data rows ({len(dates)}) for long_w={long_w}.")

    hedged, unhedged = [1.0], [1.0]
    breakdown_log, active_hedge, expire_i = [], None, 0

    for i in range(long_w, len(dates)):
        bm_ret = safe_ret(rets.iloc[i], benchmark)

        if (i - long_w) % rebal_days == 0:
            price_hist = all_prices.iloc[:i]
            stats = compute_baseline(price_hist[prices.columns], short_w, long_w)
            corr = rolling_corr_matrix(price_hist[prices.columns], short_w)
            pairs = get_pair_map(corr)
            bds = detect_breakdowns(pairs, stats, z_thresh)
            if bds:
                breakdown_log.append({"date": str(dates[i].date()), "count": len(bds)})
                expire_i = i + hold_days
                active_hedge = {}
                for bd in bds[:5]:
                    for h in get_hedges(bd):
                        inst = h["instrument"]
                        if inst in rets.columns and not inst.endswith("_PUT"):
                            sign = 1.0 if h["direction"] == "long" else -1.0
                            active_hedge[inst] = active_hedge.get(inst, 0.0) + sign * h["weight"]
                if not active_hedge:
                    active_hedge = None
            else:
                if i > expire_i:
                    active_hedge = None

        hedge_ret = 0.0
        if active_hedge:
            for inst, w in active_hedge.items():
                hedge_ret += w * safe_ret(rets.iloc[i], inst)

        unhedged.append(unhedged[-1] * (1 + bm_ret))
        hedged.append(hedged[-1] * (1 + bm_ret + hedge_ret))

    dates_out = [str(d.date()) for d in dates[long_w:]]
    h_arr = np.nan_to_num(np.array(hedged[1:]), nan=1.0)
    u_arr = np.nan_to_num(np.array(unhedged[1:]), nan=1.0)

    def sharpe(arr):
        r = np.diff(arr) / np.where(arr[:-1] == 0, 1, arr[:-1])
        r = r[np.isfinite(r)]
        return float(np.mean(r) / np.std(r) * np.sqrt(252)) if len(r) > 1 and np.std(r) > 0 else 0.0

    def mdd(arr):
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return 0.0
        peak = np.maximum.accumulate(arr)
        return float(((arr - peak) / np.where(peak == 0, 1, peak)).min())

    eq = {d: {"hedged": round(float(h), 6), "unhedged": round(float(u), 6)}
          for d, h, u in zip(dates_out, h_arr, u_arr)}

    final_h = float(h_arr[-1]) if len(h_arr) > 0 and np.isfinite(h_arr[-1]) else 1.0
    final_u = float(u_arr[-1]) if len(u_arr) > 0 and np.isfinite(u_arr[-1]) else 1.0

    return {
        "total_return_hedged":   round(final_h - 1, 4),
        "total_return_unhedged": round(final_u - 1, 4),
        "sharpe_hedged":         round(sharpe(h_arr), 3),
        "sharpe_unhedged":       round(sharpe(u_arr), 3),
        "max_drawdown_hedged":   round(mdd(h_arr), 4),
        "max_drawdown_unhedged": round(mdd(u_arr), 4),
        "num_breakdowns":        len(breakdown_log),
        "equity_curve":          eq,
        "benchmark":             benchmark,
    }


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 Correlation Detector")
    st.caption("Real-time breakdown detection & hedge optimizer")
    st.divider()

    if st.button("🔄  Refresh Live Data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    st.subheader("Filters")
    asset_classes = ["All"] + list(ASSETS.keys())
    selected_class = st.selectbox("Asset Class", asset_classes)
    z_threshold = st.slider("Min |Z-Score|", 0.5, 5.0, 2.0, 0.25)
    history_days = st.slider("History Window (days)", 60, 365, 180)
    short_window = st.slider("Rolling Window (days)", 10, 90, 30)
    long_window  = st.slider("Baseline Window (days)", 60, 252, 120)

    st.divider()
    st.subheader("Backtest Settings")
    bt_benchmark = st.selectbox("Benchmark", ["SPY", "QQQ", "TLT", "GLD"])
    bt_z = st.slider("Backtest Z-Threshold", 1.0, 4.0, 2.0, 0.25)
    run_bt = st.button("▶  Run Backtest", width="stretch")

    st.divider()
    st.caption(f"Data cached 5 min · {datetime.now().strftime('%H:%M:%S')}")

# ── Load data ──────────────────────────────────────────────────────────────────

with st.spinner("Fetching live prices from Yahoo Finance…"):
    prices_raw  = fetch_prices(tuple(ALL_SYMBOLS), history_days + long_window + 30)
    hedge_prices = fetch_prices(tuple(HEDGE_SYMBOLS), history_days + long_window + 30)

if prices_raw.empty:
    st.error("Could not fetch price data. Check your internet connection and try refreshing.")
    st.stop()

# Apply asset class filter
if selected_class != "All":
    keep = [s for s in prices_raw.columns if CLASS_MAP.get(s) == selected_class]
    prices = prices_raw[keep] if keep else prices_raw
else:
    prices = prices_raw

available = prices.dropna(axis=1, thresh=short_window).columns.tolist()
if len(available) < 2:
    st.error("Not enough assets with sufficient data. Try widening the history window.")
    st.stop()

with st.spinner("Computing correlations…"):
    corr_matrix = rolling_corr_matrix(prices[available], short_window)
    cur_pairs   = get_pair_map(corr_matrix)
    stats       = compute_baseline(prices[available], short_window, long_window)
    breakdowns  = detect_breakdowns(cur_pairs, stats, z_threshold)

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(
    ["📡 Live Monitor", "⚠️ Breakdowns", "🛡️ Hedge Recommendations", "📈 Backtesting"]
)

# ══ TAB 1 — LIVE MONITOR ══════════════════════════════════════════════════════

with tab1:
    st.header("Real-Time Correlation Heatmap")

    col_heat, col_stats = st.columns([3, 1])

    with col_heat:
        st.subheader(f"{short_window}-Day Rolling Correlation Matrix")
        fig_heat = go.Figure(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale=[[0.0, "#d73027"], [0.5, "#ffffbf"], [1.0, "#1a9850"]],
            zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
        ))
        fig_heat.update_layout(
            height=580, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="#fafafa", size=10),
        )
        st.plotly_chart(fig_heat, width="stretch")

    with col_stats:
        upper_mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        upper_vals = corr_matrix.values[upper_mask]  # numpy direct — no stack() NaN issues

        st.subheader("Summary Stats")
        st.metric("Avg Correlation", f"{upper_vals.mean():.3f}")
        st.metric("Max Correlation", f"{upper_vals.max():.3f}")
        st.metric("Min Correlation", f"{upper_vals.min():.3f}")
        st.metric("Std Deviation",   f"{upper_vals.std():.3f}")
        st.metric("Active Breakdowns", len(breakdowns))

        st.divider()
        st.subheader("Most Extreme Pairs")
        cols = corr_matrix.columns.tolist()
        ep_rows = [
            {"Asset 1": cols[i], "Asset 2": cols[j], "Correlation": float(corr_matrix.values[i, j])}
            for i in range(len(cols)) for j in range(i + 1, len(cols))
            if not np.isnan(corr_matrix.values[i, j])
        ]
        ep_df = pd.DataFrame(ep_rows).sort_values("Correlation", key=abs, ascending=False).head(8)
        st.dataframe(
            ep_df,
            hide_index=True,
            width="stretch",
            column_config={
                "Correlation": st.column_config.NumberColumn(format="%.3f"),
            },
        )

    # Cross-asset class heatmap
    st.subheader("Cross-Asset Class Correlation Overview")
    class_list = list(ASSETS.keys())
    rets_tail  = prices[available].pct_change().tail(short_window)
    rows = []
    for c1 in class_list:
        row = []
        for c2 in class_list:
            s1 = [s for s in available if CLASS_MAP.get(s) == c1]
            s2 = [s for s in available if CLASS_MAP.get(s) == c2]
            if not s1 or not s2 or c1 == c2:
                row.append(1.0 if c1 == c2 else float("nan"))
            else:
                vals = [rets_tail[a].corr(rets_tail[b]) for a in s1 for b in s2
                        if a in rets_tail and b in rets_tail]
                row.append(float(np.nanmean(vals)) if vals else float("nan"))
        rows.append(row)
    class_df = pd.DataFrame(rows, index=class_list, columns=class_list)
    fig_class = px.imshow(
        class_df, color_continuous_scale=["#d73027", "#ffffbf", "#1a9850"],
        zmin=-1, zmax=1, text_auto=".2f",
        title="Average Cross-Asset Class Correlation", height=350,
    )
    fig_class.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font=dict(color="#fafafa"))
    st.plotly_chart(fig_class, width="stretch")


# ══ TAB 2 — BREAKDOWNS ════════════════════════════════════════════════════════

with tab2:
    st.header("Active Correlation Breakdowns")

    c1, c2, c3, c4 = st.columns(4)
    for col, sev in zip([c1, c2, c3, c4], ["extreme", "severe", "moderate", "mild"]):
        col.metric(sev.capitalize(), sum(1 for b in breakdowns if b["severity"] == sev))

    st.divider()

    if not breakdowns:
        st.success(f"No breakdowns detected above |z| > {z_threshold}.")
    else:
        bd_df = pd.DataFrame([{
            "Pair":         b["pair"],
            "Class 1":      b["class1"],
            "Class 2":      b["class2"],
            "Current Corr": b["corr"],
            "Baseline":     b["mean"],
            "Z-Score":      b["z"],
            "Direction":    b["direction"],
            "Severity":     b["severity"],
        } for b in breakdowns])

        st.dataframe(
            bd_df,
            hide_index=True,
            width="stretch",
            column_config={
                "Current Corr": st.column_config.NumberColumn(format="%.3f"),
                "Baseline":     st.column_config.NumberColumn(format="%.3f"),
                "Z-Score":      st.column_config.NumberColumn(format="%+.2f"),
                "Severity":     st.column_config.TextColumn(),
            },
        )

        fig_bar = px.bar(
            bd_df.head(20), x="Pair", y="Z-Score", color="Severity",
            color_discrete_map=SEV_COLOR,
            title="Top 20 Breakdown Z-Scores",
            height=420,
        )
        fig_bar.add_hline(y=z_threshold,  line_dash="dash", line_color="white",
                          annotation_text=f"Threshold ({z_threshold})")
        fig_bar.add_hline(y=-z_threshold, line_dash="dash", line_color="white")
        fig_bar.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                              font=dict(color="#fafafa"))
        st.plotly_chart(fig_bar, width="stretch")


# ══ TAB 3 — HEDGE RECOMMENDATIONS ════════════════════════════════════════════

with tab3:
    st.header("Hedge Recommendations")
    st.caption("Positions suggested to neutralize active correlation breakdown risk.")

    if not breakdowns:
        st.success("No active breakdowns — no hedges needed.")
    else:
        agg: Dict[str, Dict] = {}
        for bd in breakdowns[:10]:
            for h in get_hedges(bd):
                inst = h["instrument"]
                if inst not in agg:
                    agg[inst] = {"Instrument": inst, "Type": "etf", "Direction": h["direction"],
                                 "Weight (%)": 0.0, "Triggered By": [], "Rationale": h["rationale"]}
                agg[inst]["Weight (%)"] += h["weight"] * 100
                agg[inst]["Triggered By"].append(bd["pair"])

        hedge_df = pd.DataFrame(agg.values())
        hedge_df["Weight (%)"]    = hedge_df["Weight (%)"].clip(upper=25).round(2)
        hedge_df["Triggered By"]  = hedge_df["Triggered By"].apply(lambda x: ", ".join(set(x)))
        hedge_df = hedge_df.sort_values("Weight (%)", ascending=False)

        m1, m2, m3 = st.columns(3)
        m1.metric("Unique Hedge Instruments",    len(hedge_df))
        m2.metric("Avg Suggested Weight",        f"{hedge_df['Weight (%)'].mean():.1f}%")
        m3.metric("Max Expected Risk Reduction",
                  f"{hedge_df.get('Expected Risk Reduction (%)', pd.Series([0])).max():.0f}%"
                  if "Expected Risk Reduction (%)" in hedge_df else "—")

        st.dataframe(
            hedge_df[["Instrument", "Direction", "Weight (%)", "Triggered By"]],
            hide_index=True, width="stretch",
        )

        fig_hedge = px.bar(
            hedge_df, x="Instrument", y="Weight (%)", color="Direction",
            color_discrete_map={"long": "#2196F3", "short": "#F44336"},
            title="Suggested Hedge Weights", height=380,
        )
        fig_hedge.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                font=dict(color="#fafafa"))
        st.plotly_chart(fig_hedge, width="stretch")

        with st.expander("Hedge Rationale Details"):
            for _, row in hedge_df.iterrows():
                st.markdown(
                    f"**{row['Instrument']}** ({row['Direction'].upper()} {row['Weight (%)']:.1f}%): "
                    f"{row['Rationale']}"
                )


# ══ TAB 4 — BACKTESTING ══════════════════════════════════════════════════════

with tab4:
    st.header("Hedge Strategy Backtesting")
    st.caption(
        "Walk-forward simulation: detect breakdowns on available history, "
        "apply hedge positions for 10 days, compare vs unhedged benchmark."
    )

    if "bt_result" not in st.session_state:
        st.session_state["bt_result"] = None

    if run_bt:
        bt_long_w = min(long_window, max(60, len(prices_raw) // 3))
        if len(prices_raw) < bt_long_w + short_window + 10:
            st.error(f"Need at least {bt_long_w + short_window + 10} days of data. "
                     f"Increase History Window in sidebar.")
        else:
            with st.spinner("Running walk-forward backtest… (30–60 seconds)"):
                try:
                    result = run_backtest(
                        prices_raw, hedge_prices, bt_benchmark,
                        short_window, bt_long_w, bt_z,
                    )
                    st.session_state["bt_result"] = result
                    st.success("Backtest complete!")
                except Exception as e:
                    st.error(f"Backtest error: {e}")

    result = st.session_state.get("bt_result")

    if result is None:
        st.info("Configure settings in the sidebar and click **▶ Run Backtest** to start.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Hedged Return",
            f"{result['total_return_hedged']:.2%}",
            delta=f"{result['total_return_hedged'] - result['total_return_unhedged']:.2%} vs unhedged",
        )
        m2.metric(
            "Sharpe (Hedged)",
            f"{result['sharpe_hedged']:.2f}",
            delta=f"{result['sharpe_hedged'] - result['sharpe_unhedged']:.2f}",
        )
        m3.metric(
            "Max Drawdown (Hedged)",
            f"{result['max_drawdown_hedged']:.2%}",
            delta=f"{result['max_drawdown_hedged'] - result['max_drawdown_unhedged']:.2%}",
            delta_color="inverse",
        )
        m4.metric("Breakdowns Detected", result["num_breakdowns"])

        ec = result.get("equity_curve", {})
        if ec:
            ec_df = pd.DataFrame(ec).T.reset_index()
            ec_df.columns = ["date", "hedged", "unhedged"]
            ec_df["date"] = pd.to_datetime(ec_df["date"])
            ec_df = ec_df.sort_values("date")

            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=ec_df["date"], y=ec_df["hedged"],
                name="Hedged", line=dict(color="#2196F3", width=2),
            ))
            fig_eq.add_trace(go.Scatter(
                x=ec_df["date"], y=ec_df["unhedged"],
                name=f"Unhedged ({result['benchmark']})",
                line=dict(color="#FF9800", width=2, dash="dot"),
            ))
            fig_eq.update_layout(
                title="Equity Curve: Hedged vs Unhedged",
                yaxis_title="Portfolio Value (normalized to 1.0)",
                xaxis_title="Date",
                height=460,
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                font=dict(color="#fafafa"),
                legend=dict(orientation="h", y=1.02),
            )
            st.plotly_chart(fig_eq, width="stretch")

        with st.expander("Backtest Parameters"):
            st.json({
                "benchmark":       result["benchmark"],
                "short_window":    short_window,
                "baseline_window": long_window,
                "z_threshold":     bt_z,
                "hedge_hold_days": 10,
                "rebalance_days":  5,
            })
