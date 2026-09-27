# AI NIFTY 50 Personalized Portfolio Optimizer

## Model architecture

Yahoo Finance 5-year data
→ feature engineering
→ Random Forest stock ranking
→ candidate portfolio generation (per risk profile)
→ joint Monte Carlo simulation (1,000 paths per candidate)
→ portfolio return/risk/downside statistics
→ investor-specific risk-aversion parameter λ
→ continuous constrained optimization
→ personalized portfolio

### Expected returns

Each stock's modeled annual expected return blends its own historical
geometric return with the broad market (NIFTY) estimate, plus a small
Random Forest tilt:

```
expected_return = 0.80 * stock_geometric_return + 0.20 * NIFTY_geometric_return + ml_tilt
```

`ml_tilt` is capped at ±2 percentage points and is derived from the Random
Forest's estimated probability that the stock beats NIFTY over the next 20
trading sessions.

### Score (final optimizer objective)

The final portfolio is solved directly by a continuous constrained optimizer
(SLSQP, multi-start) over raw (non-standardized) annual return/volatility
figures:

```
Score = Expected Return
        − λ × Portfolio Volatility
        − 0.30 × λ × 1.645 × Portfolio Volatility      (downside proxy)
        − 0.15 × λ × Weighted Avg. Individual Stock Volatility
        − 0.12 × Concentration (Σ weight²)
```

The individual-stock-volatility term means the investor's risk preference
influences *which stocks* get weight, not only the portfolio-level
covariance. The concentration term discourages the optimizer from piling
weight into a small number of names even when the cap technically allows it.

The risk–return scatter plot in the app shows a simplified version of this
score (return, volatility and concentration only — it omits the
individual-stock-volatility term) for the pre-generated candidate surface,
since that surface is cached and computed before the final optimizer runs.
The actual selected portfolio is always the output of the full optimizer
above, not the nearest cached candidate; the "nearest candidate" marker
is shown for context on the chart only.

### Risk aversion (λ)

Base λ by risk profile:

| Risk profile | Base λ |
|---|---|
| Low    | 0.70 |
| Medium | 0.38 |
| High   | 0.10 |

Age applies a stronger, nonlinear multiplier (interpolated):

| Age | Multiplier |
|---|---|
| 18  | 0.60× |
| 25  | 0.75× |
| 35  | 0.95× |
| 45  | 1.15× |
| 55  | 1.40× |
| 65  | 1.70× |
| 75+ | 2.00× |

Horizon applies a smaller multiplier, from 1.10× at a 1-year horizon down to
0.90× at a 40+ year horizon.

```
λ = base(risk_profile) × age_multiplier(age) × horizon_multiplier(horizon)
```

### Per-stock weight caps

Risk profile also sets the maximum weight any single stock may receive —
lower-risk investors get a tighter concentration ceiling:

| Risk profile | Max weight per stock |
|---|---|
| Low    | 6%  |
| Medium | 8%  |
| High   | 10% |

Candidate portfolios used for the risk–return chart are generated separately
per risk profile, under that same profile's cap, so the displayed candidates
never violate the constraint the final optimizer is actually solving under.

## Installation

```powershell
pip install -r requirements.txt
```

## Download data

```powershell
python download_data.py
```

This downloads approximately five years of NIFTY 50 stock prices and the
NIFTY 50 index from Yahoo Finance into `data/`.

## Prepare the model cache

```powershell
python prepare_model.py
```

This is the expensive one-time step. It trains the Random Forest models,
generates the candidate portfolio surface for each risk profile, and runs
1,000-path joint Monte Carlo simulations for each candidate. Result is
cached to `data/model_cache.pkl`.

The cache is automatically rebuilt the next time it's loaded if any tunable
model constant (λ table, weight caps, MC paths, penalty weights, return
blending, etc.) has changed since it was built — you don't need to manually
delete `data/model_cache.pkl` after editing `portfolio_model.py`, just rerun
`prepare_model.py` or launch the app.

## Run website

```powershell
streamlit run app.py
```

After the cache exists and its parameters haven't changed, changing investor
inputs (age, risk profile, horizon, amount) does not re-download Yahoo
Finance data or retrain the Random Forest — it only re-runs the fast
optimizer and a final Monte Carlo pass on the selected weights.

## Important modeling note

Monte Carlo does not itself choose the investor's weights. Candidate
portfolios (generated per risk profile) have different weights; Monte Carlo
evaluates their future distributions purely for the diagnostic risk–return
chart. The actual portfolio shown to the investor comes from the continuous
optimizer described above, solved under that investor's λ and weight cap.
The investor profile changes λ and the weight cap, which changes both the
optimizer's solution and the score assigned to each candidate on the chart.

## Data alignment note

The feature-engineering pipeline explicitly aligns every stock with the
NIFTY trading-date index before calculating the 20-session forward target.
This prevents pandas index-mismatch errors when an individual stock has
missing Yahoo Finance observations.

## Disclaimer

Educational demonstration only. Risk-profile mappings, λ values and
diversification penalties are model-design assumptions, not individualized
financial advice or empirically calibrated suitability thresholds.
