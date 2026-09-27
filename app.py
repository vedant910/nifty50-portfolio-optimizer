from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
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
# DEVICE DETECTION
#
# Streamlit renders server-side, so there is no native way to read the
# browser's viewport width in Python. The standard workaround: inject a tiny
# script that measures window.innerWidth once, writes it into the URL as a
# query param, and reloads. After that one reload, `?view=...` is already in
# the URL, so no further redirect happens on subsequent interactions.
#
# A manual override in the sidebar is provided as a reliable fallback in case
# auto-detection is ever wrong (e.g. a tablet in landscape, or a host that
# blocks the redirect script).
# ---------------------------------------------------------------------------
MOBILE_BREAKPOINT_PX = 768

query_params = st.query_params
if "view" not in query_params:
    components.html(
        f"""
        <script>
        const width = window.innerWidth;
        const view = width < {MOBILE_BREAKPOINT_PX} ? "mobile" : "desktop";
        const url = new URL(window.parent.location);
        url.searchParams.set("view", view);
        window.parent.location.replace(url.toString());
        </script>
        """,
        height=0,
    )
    st.stop()

detected_view = query_params.get("view", "desktop")

with st.sidebar:
    st.markdown("## Investor Profile")
    layout_choice = st.radio(
        "Layout",
        ["Auto", "Desktop", "Mobile"],
        horizontal=True,
        help="Auto uses your detected screen size. Override it here to preview either layout.",
    )
    st.markdown("---")

if layout_choice == "Desktop":
    IS_MOBILE = False
elif layout_choice == "Mobile":
    IS_MOBILE = True
else:
    IS_MOBILE = detected_view == "mobile"

# ---------------------------------------------------------------------------
# STYLES — two separate blocks, not one scaled-down version of the other.
# ---------------------------------------------------------------------------
FONT_IMPORT = """
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=IBM+Plex+Mono:wght@500;600&display=swap');
"""

DESKTOP_CSS = f"""
<style>
{FONT_IMPORT}
.stApp {{ background:#14161B; color:#E7E5DE; }}
.stApp, .stApp p, .stApp span, .stApp label, .stApp div {{ font-family: -apple-system, "Segoe UI", sans-serif; }}
[data-testid="stSidebar"] {{ background:#191B21; border-right:1px solid #2A2D35; }}
.block-container {{ padding-left: 3rem; padding-right: 3rem; }}

.ledger-head {{ border-bottom:1px solid #3A3D46; padding-bottom:22px; margin-bottom:6px; }}
.ledger-head .kicker {{ color:#B8872E; font-size:13px; letter-spacing:.2px; margin-bottom:10px; }}
.ledger-head h1 {{
    font-family:"Source Serif 4", Georgia, serif; font-weight:600; font-size:42px;
    line-height:1.12; margin:0 0 14px 0; color:#F2F0E9;
}}
.ledger-head p {{ color:#A9A69C; font-size:16px; max-width:640px; line-height:1.5; margin:0; }}

.pipeline {{ margin-top:26px; }}
.pipeline-row {{ display:flex; gap:0; border-top:1px solid #2A2D35; padding:16px 0; }}
.pipeline-row:last-child {{ border-bottom:1px solid #2A2D35; }}
.pipeline-num {{ font-family:"IBM Plex Mono", monospace; color:#1F7A53; font-size:14px; width:34px; flex-shrink:0; padding-top:2px; }}
.pipeline-body b {{ color:#F2F0E9; font-weight:600; }}
.pipeline-body p {{ color:#A9A69C; font-size:14px; margin:4px 0 0 0; line-height:1.5; }}

.stat {{ border-left:2px solid #1F7A53; padding:2px 0 2px 14px; }}
.stat .label {{ color:#8B8880; font-size:12px; }}
.stat .value {{
    font-family:"IBM Plex Mono", monospace; color:#F2F0E9; font-size:24px;
    font-weight:600; margin-top:4px; font-variant-numeric: tabular-nums;
}}

.section-title {{
    font-family:"Source Serif 4", Georgia, serif; font-size:22px; font-weight:600;
    color:#F2F0E9; border-bottom:1px solid #2A2D35; padding-bottom:10px; margin:32px 0 16px 0;
}}
.note-block {{ color:#A9A69C; font-size:13px; line-height:1.6; }}
</style>
"""

MOBILE_CSS = f"""
<style>
{FONT_IMPORT}
.stApp {{ background:#14161B; color:#E7E5DE; }}
.stApp, .stApp p, .stApp span, .stApp label, .stApp div {{ font-family: -apple-system, "Segoe UI", sans-serif; }}
[data-testid="stSidebar"] {{ background:#191B21; border-right:1px solid #2A2D35; }}
.block-container {{ padding-left: 0.9rem; padding-right: 0.9rem; padding-top: 1.25rem; }}

.ledger-head {{ border-bottom:1px solid #3A3D46; padding-bottom:14px; margin-bottom:4px; }}
.ledger-head .kicker {{ color:#B8872E; font-size:11.5px; letter-spacing:.2px; margin-bottom:8px; }}
.ledger-head h1 {{
    font-family:"Source Serif 4", Georgia, serif; font-weight:600; font-size:26px;
    line-height:1.2; margin:0 0 10px 0; color:#F2F0E9;
}}
.ledger-head p {{ color:#A9A69C; font-size:14px; line-height:1.5; margin:0; }}

.pipeline {{ margin-top:16px; }}
.pipeline-row {{ display:flex; gap:8px; border-top:1px solid #2A2D35; padding:12px 0; align-items:flex-start; }}
.pipeline-row:last-child {{ border-bottom:1px solid #2A2D35; }}
.pipeline-num {{ font-family:"IBM Plex Mono", monospace; color:#1F7A53; font-size:12.5px; width:22px; flex-shrink:0; padding-top:2px; }}
.pipeline-body {{ min-width:0; }}
.pipeline-body b {{ color:#F2F0E9; font-weight:600; font-size:14px; }}
.pipeline-body p {{ color:#A9A69C; font-size:12.5px; margin:4px 0 0 0; line-height:1.5; word-wrap:break-word; }}

.stat {{ border-left:2px solid #1F7A53; padding:2px 0 2px 10px; margin-bottom:10px; }}
.stat .label {{ color:#8B8880; font-size:11px; }}
.stat .value {{
    font-family:"IBM Plex Mono", monospace; color:#F2F0E9; font-size:18px;
    font-weight:600; margin-top:3px; font-variant-numeric: tabular-nums;
    overflow-wrap:break-word;
}}

.section-title {{
    font-family:"Source Serif 4", Georgia, serif; font-size:17px; font-weight:600;
    color:#F2F0E9; border-bottom:1px solid #2A2D35; padding-bottom:8px; margin:22px 0 12px 0;
}}
.note-block {{ color:#A9A69C; font-size:12px; line-height:1.55; }}

[data-testid="stDataFrame"], [data-testid="stTable"] {{ overflow-x: auto; }}
</style>
"""

st.markdown(MOBILE_CSS if IS_MOBILE else DESKTOP_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown("""
<div class="ledger-head">
<div class="kicker">Personalized equity allocation &middot; NSE / NIFTY 50</div>
<h1>Personalized NIFTY 50<br>Portfolio Optimizer</h1>
<p>Ranks NIFTY 50 stocks with a Random Forest model, tests candidate allocations against
thousands of simulated market paths, and solves for the weights that fit your risk profile.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SIDEBAR INPUTS (shared — the same controls work fine on both layouts)
# ---------------------------------------------------------------------------
with st.sidebar:
    age = st.number_input("Age", 18, 100, 22)
    risk_profile = st.selectbox("Risk profile", ["Low", "Medium", "High"], index=2)
    horizon = st.number_input("Investment horizon (years)", 1, 50, 7)
    amount = st.number_input("Investment amount (₹)", 10000, 100000000, 500000, step=10000)
    run = st.button("Generate Portfolio", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption(f"Trained on 5 years of NIFTY 50 price history, {MC_PATHS:,} Monte Carlo paths per candidate.")
    st.caption(f"Layout: **{'Mobile' if IS_MOBILE else 'Desktop'}** ({layout_choice.lower()})")

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
shown = portfolio[portfolio["weight"] > 0.001].copy()

# ---------------------------------------------------------------------------
# STAT ROW — one row of 4 on desktop, 2x2 on mobile
# ---------------------------------------------------------------------------
stat_items = [
    ("Risk profile", risk_profile.upper()),
    ("Risk aversion λ", f'{result["lambda"]:.2f}'),
    ("Expected return", f'{result["expected_return"]:.2%}'),
    ("Portfolio volatility", f'{result["volatility"]:.2%}'),
]

if IS_MOBILE:
    row1 = st.columns(2)
    row2 = st.columns(2)
    cols = row1 + row2
else:
    cols = st.columns(4)

for col, (label, value) in zip(cols, stat_items):
    with col:
        st.markdown(f'<div class="stat"><div class="label">{label}</div><div class="value">{value}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section"><h3>Why this investor received these weights</h3></div>', unsafe_allow_html=True)
st.write(
    f"A **{risk_profile.lower()}-risk** profile produces λ = **{result['lambda']:.2f}** for an investor aged **{age}** "
    f"with a **{horizon}-year** horizon. The optimizer maximizes expected return while penalizing portfolio variance "
    "and excessive concentration. A higher λ places more emphasis on reducing variance."
)

# ---------------------------------------------------------------------------
# ALLOCATION CHARTS — side-by-side on desktop, stacked on mobile
# ---------------------------------------------------------------------------
def bar_chart():
    fig = px.bar(
        shown.sort_values("weight"),
        x="weight", y="ticker", orientation="h", text="weight",
        labels={"weight": "Portfolio weight", "ticker": "Stock"},
    )
    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig.update_layout(
        height=650 if not IS_MOBILE else max(360, min(560, 26 * len(shown))),
        margin=dict(l=10, r=50, t=15, b=15),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E5E7EB", xaxis_tickformat=".0%",
        font_size=12 if IS_MOBILE else 13,
    )
    return fig

def pie_chart():
    fig2 = px.pie(shown, values="weight", names="ticker", hole=.58)
    fig2.update_layout(
        height=650 if not IS_MOBILE else 440,
        margin=dict(l=10, r=10, t=15, b=15),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#E5E7EB",
        font_size=12 if IS_MOBILE else 13,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5) if IS_MOBILE else {},
    )
    return fig2

if IS_MOBILE:
    st.plotly_chart(bar_chart(), use_container_width=True, config={"responsive": True})
    st.plotly_chart(pie_chart(), use_container_width=True, config={"responsive": True})
else:
    left, right = st.columns([1.35, 1])
    with left:
        st.plotly_chart(bar_chart(), use_container_width=True)
    with right:
        st.plotly_chart(pie_chart(), use_container_width=True)

# ---------------------------------------------------------------------------
# PORTFOLIO CONSTRUCTION NOTES — 3 columns on desktop, stacked on mobile
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Portfolio construction</div>', unsafe_allow_html=True)
construction_items = [
    ("ML signal", "Random Forest probability contributes a modest return tilt to each stock."),
    ("Risk penalty", "Portfolio variance is penalized according to the investor's risk profile."),
    ("Diversification", f"No stock can exceed {result['max_weight']:.0%} for this risk profile, and the optimizer also penalizes concentration."),
]
if IS_MOBILE:
    for title, body in construction_items:
        st.markdown(f'<div class="stat"><div class="label">{title}</div><p class="note-block" style="margin-top:6px;">{body}</p></div>', unsafe_allow_html=True)
else:
    cc1, cc2, cc3 = st.columns(3)
    for col, (title, body) in zip([cc1, cc2, cc3], construction_items):
        with col:
            st.markdown(f'<div class="stat"><div class="label">{title}</div><p class="note-block" style="margin-top:6px;">{body}</p></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# STOCK-LEVEL TABLE — full table on desktop; trimmed table + expander on mobile
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Stock-level analysis</div>', unsafe_allow_html=True)

full_cols = [
    "ticker", "ml_probability", "mc_expected_return",
    "mc_volatility", "mc_var_95", "mc_es_95",
    "weight", "allocation_inr"
]
full_names = [
    "Stock", "ML probability", "MC expected return",
    "MC volatility", "MC 5% outcome", "MC expected shortfall",
    "Weight", "Allocation"
]
fmt_map = {
    "ML probability": "{:.1%}", "MC expected return": "{:.2%}",
    "MC volatility": "{:.2%}", "MC 5% outcome": "{:.2%}",
    "MC expected shortfall": "{:.2%}", "Weight": "{:.2%}",
    "Allocation": "₹{:,.0f}",
}

if IS_MOBILE:
    slim_cols = ["ticker", "mc_expected_return", "weight", "allocation_inr"]
    slim_names = ["Stock", "MC expected return", "Weight", "Allocation"]
    slim = portfolio[slim_cols].copy()
    slim.columns = slim_names
    st.dataframe(
        slim.style.format({k: v for k, v in fmt_map.items() if k in slim_names}),
        use_container_width=True, hide_index=True,
    )
    with st.expander("Show full stock-level detail"):
        display = portfolio[full_cols].copy()
        display.columns = full_names
        st.dataframe(display.style.format(fmt_map), use_container_width=True, hide_index=True)
else:
    display = portfolio[full_cols].copy()
    display.columns = full_names
    st.dataframe(display.style.format(fmt_map), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# RISK-RETURN SCATTER
# ---------------------------------------------------------------------------
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
    height=420 if IS_MOBILE else 500,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#E5E7EB",
    font_size=12 if IS_MOBILE else 13,
    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5) if IS_MOBILE else {},
)
st.plotly_chart(fig3, use_container_width=True, config={"responsive": True})

# ---------------------------------------------------------------------------
# METHODOLOGY
# ---------------------------------------------------------------------------
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
