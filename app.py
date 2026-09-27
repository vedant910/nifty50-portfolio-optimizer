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

st.markdown("""
<style>
.stApp { background:#070B14; color:#F4F7FB; }
[data-testid="stSidebar"] { background:#0B1120; border-right:1px solid #1E293B; }
.hero { padding:34px 38px; border:1px solid #1E293B; border-radius:22px;
background:linear-gradient(135deg,#0E172A 0%,#101B32 55%,#0B1222 100%); margin-bottom:24px; }
.eyebrow { color:#60A5FA; font-size:13px; font-weight:800; letter-spacing:1.8px; text-transform:uppercase; }
.hero h1 { font-size:44px; margin:8px 0; line-height:1.04; }
.hero p { color:#A9B4C7; font-size:16px; max-width:900px; }
.card { background:#0E1627; border:1px solid #1E293B; border-radius:16px; padding:18px; }
.label { color:#94A3B8; font-size:11px; text-transform:uppercase; letter-spacing:1px; }
.value { color:#F8FAFC; font-size:27px; font-weight:800; margin-top:5px; }
.small { color:#94A3B8; font-size:13px; }
.badge { display:inline-block; padding:6px 12px; border-radius:999px; background:#172554;
color:#93C5FD; font-size:12px; font-weight:700; letter-spacing:.4px; }
.section { margin-top:28px; margin-bottom:12px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<div class="eyebrow">AI × Monte Carlo × Personalization</div>
<h1>Personalized NIFTY 50<br>Portfolio Optimizer</h1>
<p>Random Forest stock ranking + joint Monte Carlo simulation + investor risk profiling + continuous portfolio optimization.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Investor Profile")
    age = st.number_input("Age", 18, 100, 22)
    risk_profile = st.selectbox("Risk profile", ["Low", "Medium", "High"], index=2)
    horizon = st.number_input("Investment horizon (years)", 1, 50, 7)
    amount = st.number_input("Investment amount (₹)", 10000, 100000000, 500000, step=10000)
    run = st.button("🚀 Generate AI Portfolio", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption(f"5 years • NIFTY 50 • Random Forest • {MC_PATHS:,} Monte Carlo paths")

cache_path = Path(__file__).resolve().parent / "data" / "model_cache.pkl"
if not cache_path.exists():
    st.warning("Model cache is not prepared yet.")
    st.code("python download_data.py\npython prepare_model.py\nstreamlit run app.py")
    st.stop()

if not run:
    cols = st.columns(4)
    cards = [
        ("01 · ML", "Random Forest estimates the probability of beating NIFTY over the next 20 trading sessions."),
        ("02 · Monte Carlo", f"Each candidate portfolio is evaluated with {MC_PATHS:,} joint simulated paths."),
        ("03 · Personalize", "Risk profile is converted into a variance penalty; age and horizon make secondary adjustments."),
        ("04 · Optimize", "A continuous constrained optimizer selects the weights instead of simply filling stock limits."),
    ]
    for c, (h, b) in zip(cols, cards):
        with c:
            st.markdown(f'<div class="card"><b>{h}</b><p class="small">{b}</p></div>', unsafe_allow_html=True)
    st.info("Enter the investor profile and click **Generate AI Portfolio**.")
    st.stop()

try:
    result = personalize(age, risk_profile, horizon, amount)
except Exception as e:
    st.error(f"Model execution failed: {e}")
    st.exception(e)
    st.stop()

portfolio = result["portfolio"]
stats = result["candidate_stats"].copy()

c1, c2, c3, c4 = st.columns(4)
for col, label, value in [
    (c1, "Risk profile", risk_profile.upper()),
    (c2, "Risk aversion λ", f'{result["lambda"]:.2f}'),
    (c3, "Expected return", f'{result["expected_return"]:.2%}'),
    (c4, "Portfolio volatility", f'{result["volatility"]:.2%}'),
]:
    with col:
        st.markdown(f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div></div>', unsafe_allow_html=True)

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
        height=650, margin=dict(l=10, r=50, t=15, b=15),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E5E7EB", xaxis_tickformat=".0%",
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig2 = px.pie(shown, values="weight", names="ticker", hole=.58)
    fig2.update_layout(
        height=650, margin=dict(l=10, r=10, t=15, b=15),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#E5E7EB",
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("### Portfolio construction")
cc1, cc2, cc3 = st.columns(3)
for col, title, body in [
    (cc1, "ML signal", "Random Forest probability contributes a modest return tilt to each stock."),
    (cc2, "Risk penalty", "Portfolio variance is penalized according to the investor's risk profile."),
    (cc3, "Diversification", f"No stock can exceed {result['max_weight']:.0%} for this risk profile, and the optimizer also penalizes concentration."),
]:
    with col:
        st.markdown(f'<div class="card"><b>{title}</b><p class="small">{body}</p></div>', unsafe_allow_html=True)

st.markdown("### Stock-level AI analysis")
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
st.dataframe(display.style.format({
    "ML probability": "{:.1%}",
    "MC expected return": "{:.2%}",
    "MC volatility": "{:.2%}",
    "MC 5% outcome": "{:.2%}",
    "MC expected shortfall": "{:.2%}",
    "Weight": "{:.2%}",
    "Allocation": "₹{:,.0f}",
}), use_container_width=True, hide_index=True)

st.markdown("### Risk–return map of candidate portfolios")
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
    height=500,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#E5E7EB",
)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("### Monte Carlo & personalization logic")
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
