from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from portfolio_model import personalize, load_cache, MC_PATHS

st.set_page_config(
    page_title="AI NIFTY 50 Personalized Portfolio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# GLOBAL STYLES — desktop-first, with @media overrides for phones/tablets.
# Breakpoints: <= 768px (tablet/phone), <= 480px (small phone).
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=IBM+Plex+Mono:wght@500;600&display=swap');

.stApp { background:#14161B; color:#E7E5DE; }
.stApp, .stApp p, .stApp span, .stApp label, .stApp div { font-family: -apple-system, "Segoe UI", sans-serif; }
[data-testid="stSidebar"] { background:#191B21; border-right:1px solid #2A2D35; }

/* Tighten Streamlit's own outer padding on small screens so content isn't cramped */
.block-container { padding-left: 3rem; padding-right: 3rem; }

.ledger-head { border-bottom:1px solid #3A3D46; padding-bottom:22px; margin-bottom:6px; }
.ledger-head .kicker { color:#B8872E; font-size:13px; letter-spacing:.2px; margin-bottom:10px; }
.ledger-head h1 {
    font-family:"Source Serif 4", Georgia, serif; font-weight:600;
    font-size:clamp(26px, 5vw, 42px);
    line-height:1.15; margin:0 0 14px 0; color:#F2F0E9;
}
.ledger-head p { color:#A9A69C; font-size:clamp(14px, 2.6vw, 16px); max-width:640px; line-height:1.5; margin:0; }

.pipeline { margin-top:26px; }
.pipeline-row { display:flex; gap:12px; border-top:1px solid #2A2D35; padding:16px 0; align-items:flex-start; }
.pipeline-row:last-child { border-bottom:1px solid #2A2D35; }
.pipeline-num { font-family:"IBM Plex Mono", monospace; color:#1F7A53; font-size:14px; width:28px; flex-shrink:0; padding-top:2px; }
.pipeline-body { min-width:0; }
.pipeline-body b { color:#F2F0E9; font-weight:600; font-size:clamp(14px, 2.6vw, 16px); }
.pipeline-body p { color:#A9A69C; font-size:clamp(12.5px, 2.4vw, 14px); margin:4px 0 0 0; line-height:1.5; word-wrap:break-word; }

.stat { border-left:2px solid #1F7A53; padding:2px 0 2px 14px; margin-bottom:14px; }
.stat .label { color:#8B8880; font-size:12px; }
.stat .value {
    font-family:"IBM Plex Mono", monospace; color:#F2F0E9;
    font-size:clamp(17px, 3.6vw, 24px);
    font-weight:600; margin-top:4px; font-variant-numeric: tabular-nums;
    overflow-wrap:break-word;
}

.section-title {
    font-family:"Source Serif 4", Georgia, serif;
    font-size:clamp(18px, 3.6vw, 22px);
    font-weight:600; color:#F2F0E9; border-bottom:1px solid #2A2D35;
    padding-bottom:10px; margin:32px 0 16px 0;
}
.note-block { color:#A9A69C; font-size:13px; line-height:1.6; }

/* Make sure any wide element (tables etc.) scrolls horizontally instead of
   overflowing the viewport and forcing the whole page to scroll sideways. */
[data-testid="stDataFrame"], [data-testid="stTable"] { overflow-x: auto; }

/* --- Tablet / large phone --- */
@media (max-width: 768px) {
    .block-container { padding-left: 1.1rem; padding-right: 1.1rem; padding-top: 1.5rem; }
    .ledger-head { padding-bottom:16px; }
    .stat { padding-left:10px; margin-bottom:10px; }
    .section-title { margin:24px 0 12px 0; }
}

/* --- Small phone --- */
@media (max-width: 480px) {
    .block-container { padding-left: 0.85rem; padding-right: 0.85rem; }
    .ledger-head .kicker { font-size:11.5px; }
    .pipeline-row { padding:12px 0; gap:8px; }
    .pipeline-num { width:22px; font-size:12.5px; }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="ledger-head">
<div class="kicker">Personalized equity allocation &middot; NSE / NIFTY 50</div>
<h1>Personalized NIFTY 50<br>Portfolio Optimizer</h1>
<p>Ranks NIFTY 50 stocks with a Random Forest model, tests candidate allocations against
thousands of simulated market paths, and solves for the weights that fit your risk profile.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Investor Profile")
    age = st.number_input("Age", 18, 100, 22)
    risk_profile = st.selectbox("Risk profile", ["Low", "Medium", "High"], index=2)
    horizon = st.number_input("Investment horizon (years)", 1, 50, 7)
    amount = st.number_input("Investment amount (₹)", 10000, 100000000, 500000, step=10000)
    run = st.button("Generate Portfolio", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption(f"Trained on 5 years of NIFTY 50 price history, {MC_PATHS:,} Monte Carlo paths per candidate.")

cache_path = Path(__file__).resolve().parent / "data" / "model_cache.pkl"
if not cache_path.exists():
    st.warning("Model cache is not prepared yet.")
    st.code("python download_data.py\npython prepare_model.py\nstreamlit run app.py")
    st.stop()

if not run:
    steps = [
        ("Rank", "Random Forest estimates the probability each stock beats NIFTY over the next 20 trading sessions."),
        ("Simulate", f"Every candidate allocation is tested against {MC_PATHS:,} joint simulated market paths."),
        ("Personalize", "Risk profile sets the variance penalty and per-stock weight cap; age and horizon adjust it further."),
        ("Solve", "A constrained optimizer finds the weights that fit your profile, rather than filling stock limits."),
    ]
    rows = "".join(
        f'<div class="pipeline-row"><div class="pipeline-num">{i:02d}</div>'
        f'<div class="pipeline-body"><b>{title}</b><p>{body}</p></div></div>'
        for i, (title, body) in enumerate(steps, 1)
    )
    st.markdown(f'<div class="pipeline">{rows}</div>', unsafe_allow_html=True)
    st.write("")
    st.info("Enter your investor profile on the left and click **Generate Portfolio**.")
    st.stop()

try:
    result = personalize(age, risk_profile, horizon, amount)
except Exception as e:
    st.error(f"Model execution failed: {e}")
    st.exception(e)
    st.stop()

portfolio = result["portfolio"]
stats = result["candidate_stats"].copy()

# On phones, 4 stat columns squeeze into unreadably narrow strips. Streamlit
# columns wrap to 2-per-row below its own internal breakpoint, but we help it
# along by using 2 columns x 2 rows instead of 4 columns x 1 row — this keeps
# each stat block wide enough to read on any screen without extra JS.
r1c1, r1c2 = st.columns(2)
r2c1, r2c2 = st.columns(2)
for col, label, value in [
    (r1c1, "Risk profile", risk_profile.upper()),
    (r1c2, "Risk aversion λ", f'{result["lambda"]:.2f}'),
    (r2c1, "Expected return", f'{result["expected_return"]:.2%}'),
    (r2c2, "Portfolio volatility", f'{result["volatility"]:.2%}'),
]:
    with col:
        st.markdown(f'<div class="stat"><div class="label">{label}</div><div class="value">{value}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section"><h3>Why this investor received these weights</h3></div>', unsafe_allow_html=True)
st.write(
    f"A **{risk_profile.lower()}-risk** profile produces λ = **{result['lambda']:.2f}** for an investor aged **{age}** "
    f"with a **{horizon}-year** horizon. The optimizer maximizes expected return while penalizing portfolio variance "
    "and excessive concentration. A higher λ places more emphasis on reducing variance."
)

left, right = st.columns([1.35, 1])
shown = portfolio[portfolio["weight"] > 0.001].copy()
with left:
    fig = px.bar(
        shown.sort_values("weight"),
        x="weight", y="ticker", orientation="h", text="weight",
        labels={"weight": "Portfolio weight", "ticker": "Stock"},
    )
    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig.update_layout(
        height=max(420, min(650, 28 * len(shown))),
        margin=dict(l=10, r=50, t=15, b=15),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E5E7EB", xaxis_tickformat=".0%",
        font_size=12,
    )
    st.plotly_chart(fig, use_container_width=True, config={"responsive": True})

with right:
    fig2 = px.pie(shown, values="weight", names="ticker", hole=.58)
    fig2.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=15, b=15),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#E5E7EB",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        font_size=12,
    )
    st.plotly_chart(fig2, use_container_width=True, config={"responsive": True})

st.markdown('<div class="section-title">Portfolio construction</div>', unsafe_allow_html=True)
cc1, cc2, cc3 = st.columns(3)
for col, (title, body) in zip(
    [cc1, cc2, cc3],
    [
        ("ML signal", "Random Forest probability contributes a modest return tilt to each stock."),
        ("Risk penalty", "Portfolio variance is penalized according to the investor's risk profile."),
        ("Diversification", f"No stock can exceed {result['max_weight']:.0%} for this risk profile, and the optimizer also penalizes concentration."),
    ],
):
    with col:
        st.markdown(f'<div class="stat"><div class="label">{title}</div><p class="note-block" style="margin-top:6px;">{body}</p></div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Stock-level analysis</div>', unsafe_allow_html=True)
display = portfolio[[
    "ticker", "ml_probability", "mc_expected_return",
    "mc_volatility", "mc_var_95", "mc_es_95",
    "weight", "allocation_inr"
]].copy()
display.columns = [
    "Stock", "ML probability", "MC expected return",
    "MC volatility", "MC 5% outcome", "MC expected shortfall",
    "Weight", "Allocation"
]
# use_container_width + the CSS overflow-x:auto rule above lets this table
# scroll horizontally on narrow screens instead of forcing the page to.
st.dataframe(display.style.format({
    "ML probability": "{:.1%}",
    "MC expected return": "{:.2%}",
    "MC volatility": "{:.2%}",
    "MC 5% outcome": "{:.2%}",
    "MC expected shortfall": "{:.2%}",
    "Weight": "{:.2%}",
    "Allocation": "₹{:,.0f}",
}), use_container_width=True, hide_index=True)

st.markdown('<div class="section-title">Risk–return map of candidate portfolios</div>', unsafe_allow_html=True)
plot_stats = stats.copy()
plot_stats["Selected candidate"] = plot_stats["is_selected"].map({True: "Nearest candidate", False: "Candidate"})
fig3 = px.scatter(
    plot_stats,
    x="volatility",
    y="expected_return",
    color="score",
    symbol="Selected candidate",
    hover_data=["candidate_id", "score", "var_95", "es_95"],
    labels={"volatility": "Simulated volatility", "expected_return": "Simulated expected return"},
)
fig3.update_layout(
    height=420,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#E5E7EB",
    font_size=12,
    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
)
st.plotly_chart(fig3, use_container_width=True, config={"responsive": True})

st.markdown('<div class="section-title">How this portfolio was built</div>', unsafe_allow_html=True)
st.markdown(f"""
1. **Random Forest:** estimates the probability that each stock beats NIFTY 50 over the next {20} trading sessions.
2. **Stock expected return:** combines the stock's Monte Carlo expected return with a modest ML probability tilt.
3. **Candidate portfolios:** feasible portfolios are generated for the risk–return comparison surface.
4. **Monte Carlo:** candidate portfolios are evaluated with **{MC_PATHS:,} joint paths**, preserving cross-stock dependence.
5. **Continuous optimization:** the final portfolio is solved directly using:

   **Score = Expected Return − λ × Portfolio Variance − diversification penalty**

6. **Personalization:** the single **Risk Profile** determines the base λ; age and horizon make secondary adjustments.
7. **Constraints:** long-only portfolio, maximum **{result['max_weight']:.0%}** per stock for this risk profile, weights sum to 100%.
""")

st.info(
    "Educational demonstration only. Risk-profile mappings, λ values and diversification penalties are model-design assumptions, "
    "not individualized financial advice or empirically calibrated suitability thresholds."
)
