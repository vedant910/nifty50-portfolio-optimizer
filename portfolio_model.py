from pathlib import Path
import pickle
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# -----------------------------
# Model configuration
# -----------------------------
HORIZON = 20
TRADING_DAYS = 252
MC_PATHS = 1000
MAX_WEIGHT = 0.10
MIN_WEIGHT = 0.0
N_CANDIDATES = 250
RISK_FREE_RATE = 0.065

# The objective uses annual volatility directly, not variance.
# These are calibrated for decimal annual-volatility units.
RISK_LAMBDA = {
    # Larger lambda = stronger penalty on portfolio risk.
    # These values are calibrated for annual returns/volatility expressed
    # in decimal units (e.g. 0.15 = 15%).
    "Low": 0.70,
    "Medium": 0.38,
    "High": 0.10,
}

# Downside penalty is proportional to risk aversion.
DOWNSIDE_PENALTY_RATIO = 0.30
CONCENTRATION_PENALTY = 0.12

# Additional penalty on the weighted average volatility of the individual
# stocks. This makes risk preference affect stock selection, not only the
# final portfolio covariance.
INDIVIDUAL_VOL_PENALTY = 0.15

# Risk profiles also impose different concentration ceilings.
# Lower-risk investors receive a tighter per-stock cap.
PROFILE_MAX_WEIGHT = {
    "Low": 0.06,
    "Medium": 0.08,
    "High": 0.10,
}

# Return-estimation controls. These shrink noisy individual-stock estimates
# toward the NIFTY market estimate instead of annualizing a raw arithmetic mean.
STOCK_RETURN_WEIGHT = 0.80
MARKET_RETURN_WEIGHT = 1.0 - STOCK_RETURN_WEIGHT
ML_TILT_MAX = 0.02  # maximum +/- 2 percentage-point annual ML adjustment

FEATURES = [
    "return_5", "return_20", "return_60", "rsi_14",
    "volatility_20", "volatility_60", "ma20_gap", "ma60_gap",
    "beta_60", "market_return_20"
]


def rsi(s, period=14):
    d = s.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    al = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # Textbook edge cases: no losses + gains -> RSI 100; completely flat -> 50.
    rs = ag / al.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out = out.where(al.ne(0), np.where(ag.gt(0), 100.0, 50.0))
    return out


def make_features(stock, nifty):
    """Build ML features on a common trading-date index."""
    common_index = nifty.index.intersection(stock.index).sort_values()
    stock = stock.reindex(common_index).astype(float)
    nifty = nifty.reindex(common_index).astype(float)

    sr = stock.pct_change(fill_method=None)
    nr = nifty.pct_change(fill_method=None)

    x = pd.DataFrame(index=common_index)
    x["return_5"] = stock.pct_change(5, fill_method=None)
    x["return_20"] = stock.pct_change(20, fill_method=None)
    x["return_60"] = stock.pct_change(60, fill_method=None)
    x["rsi_14"] = rsi(stock)
    x["volatility_20"] = sr.rolling(20, min_periods=20).std() * np.sqrt(252)
    x["volatility_60"] = sr.rolling(60, min_periods=60).std() * np.sqrt(252)
    x["ma20_gap"] = stock / stock.rolling(20, min_periods=20).mean() - 1
    x["ma60_gap"] = stock / stock.rolling(60, min_periods=60).mean() - 1

    cov = sr.rolling(60, min_periods=60).cov(nr)
    var = nr.rolling(60, min_periods=60).var()
    x["beta_60"] = cov / var.replace(0, np.nan)
    x["market_return_20"] = nifty.pct_change(20, fill_method=None)

    future_s = stock.shift(-HORIZON) / stock - 1
    future_n = nifty.shift(-HORIZON) / nifty - 1
    valid_target = future_s.notna() & future_n.notna()
    x["target"] = np.nan
    x.loc[valid_target, "target"] = (
        future_s.loc[valid_target] > future_n.loc[valid_target]
    ).astype(float)
    return x


def seed_for(ticker):
    return int(hashlib.sha256(ticker.encode()).hexdigest()[:8], 16)


def _config_signature():
    """Hash of every tunable modeling constant.

    Any change to these values changes what a cached model_cache.pkl actually
    represents, so this hash (not just a manually-bumped cache_version int)
    is what load_cache() checks before deciding to reuse a cache.
    """
    cfg = {
        "HORIZON": HORIZON,
        "TRADING_DAYS": TRADING_DAYS,
        "MC_PATHS": MC_PATHS,
        "MAX_WEIGHT": MAX_WEIGHT,
        "MIN_WEIGHT": MIN_WEIGHT,
        "N_CANDIDATES": N_CANDIDATES,
        "RISK_LAMBDA": RISK_LAMBDA,
        "DOWNSIDE_PENALTY_RATIO": DOWNSIDE_PENALTY_RATIO,
        "CONCENTRATION_PENALTY": CONCENTRATION_PENALTY,
        "INDIVIDUAL_VOL_PENALTY": INDIVIDUAL_VOL_PENALTY,
        "PROFILE_MAX_WEIGHT": PROFILE_MAX_WEIGHT,
        "STOCK_RETURN_WEIGHT": STOCK_RETURN_WEIGHT,
        "MARKET_RETURN_WEIGHT": MARKET_RETURN_WEIGHT,
        "ML_TILT_MAX": ML_TILT_MAX,
        "FEATURES": FEATURES,
    }
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def annualized_geometric_return(returns):
    """Historical annualized geometric return, with basic sanity bounds."""
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < 60:
        return np.nan
    wealth = np.prod(1.0 + r.clip(lower=-0.999))
    years = len(r) / TRADING_DAYS
    if years <= 0 or wealth <= 0:
        return np.nan
    return float(wealth ** (1.0 / years) - 1.0)


def stock_mc(returns, seed, paths=MC_PATHS, days=252, annual_expected_return=None):
    """Univariate MC for stock diagnostics.

    Uses a drift based on the modelled annual expected return rather than the
    raw arithmetic daily mean, which avoids exaggerated return forecasts.
    """
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < 60:
        return None

    log_r = np.log1p(r.clip(lower=-0.999))
    sigma_daily = float(log_r.std(ddof=1))
    if annual_expected_return is None:
        annual_expected_return = annualized_geometric_return(r)

    annual_expected_return = float(np.clip(annual_expected_return, -0.50, 0.50))
    mu_log_annual = np.log1p(annual_expected_return) - 0.5 * sigma_daily**2 * TRADING_DAYS
    mu_daily = mu_log_annual / TRADING_DAYS

    rng = np.random.default_rng(seed)
    shocks = rng.normal(mu_daily, sigma_daily, size=(paths, days))
    terminal = np.exp(shocks.sum(axis=1)) - 1
    q05 = float(np.quantile(terminal, 0.05))
    tail = terminal[terminal <= q05]
    return {
        "expected_return": float(terminal.mean()),
        "volatility": float(terminal.std(ddof=1)),
        "var_95": q05,
        "es_95": float(tail.mean()) if len(tail) else q05,
    }


def nearest_psd(a):
    a = np.asarray(a, dtype=float)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    a = (a + a.T) / 2
    vals, vecs = np.linalg.eigh(a)
    vals = np.maximum(vals, 1e-10)
    out = vecs @ np.diag(vals) @ vecs.T
    return (out + out.T) / 2


def fit_stock_models(prices):
    nifty = prices["^NSEI"].dropna()
    tickers = [c for c in prices.columns if c != "^NSEI"]
    records = []
    models = {}

    nifty_returns = nifty.pct_change(fill_method=None).dropna()
    market_return = annualized_geometric_return(nifty_returns)

    for ticker in tickers:
        stock = prices[ticker].reindex(prices.index)
        feat = make_features(stock, nifty)
        train = feat.loc[feat["target"].notna()].copy()
        train = train.dropna(subset=FEATURES + ["target"])
        if len(train) < 250 or train["target"].nunique() < 2:
            continue

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=seed_for(ticker),
                n_jobs=-1,
            )),
        ])
        model.fit(train[FEATURES], train["target"].astype(int))

        latest = feat[FEATURES].dropna().tail(1)
        if latest.empty:
            continue
        prob = float(model.predict_proba(latest)[0, 1])

        stock_returns = stock.pct_change(fill_method=None).dropna()
        stock_hist_return = annualized_geometric_return(stock_returns)
        if not np.isfinite(stock_hist_return):
            continue

        # Blend the stock-specific geometric return with the broad market estimate.
        base_return = (
            STOCK_RETURN_WEIGHT * stock_hist_return
            + MARKET_RETURN_WEIGHT * market_return
        )
        ml_tilt = float(np.clip((prob - 0.50) * 2 * ML_TILT_MAX, -ML_TILT_MAX, ML_TILT_MAX))
        model_return = float(np.clip(base_return + ml_tilt, -0.30, 0.35))

        mc = stock_mc(
            stock_returns,
            seed_for(ticker),
            annual_expected_return=model_return,
        )
        if mc is None:
            continue

        records.append({
            "ticker": ticker,
            "ml_probability": prob,
            "historical_geometric_return": stock_hist_return,
            "mc_expected_return": mc["expected_return"],
            "mc_volatility": mc["volatility"],
            "mc_var_95": mc["var_95"],
            "mc_es_95": mc["es_95"],
            "expected_return": model_return,
        })
        models[ticker] = model

    analytics = pd.DataFrame(records)
    if analytics.empty:
        raise RuntimeError("No stocks had sufficient data for the model.")

    usable = analytics["ticker"].tolist()
    ret = prices[usable].pct_change(fill_method=None).dropna(how="all")
    ret = ret.dropna(how="any")
    covariance = nearest_psd(ret.cov().to_numpy() * TRADING_DAYS)

    return analytics, usable, covariance, models


def generate_candidate_weights(n, rng, max_weight=MAX_WEIGHT):
    # Reject portfolios violating the cap. This produces varied feasible candidates.
    if n * max_weight < 1.0 - 1e-12:
        raise ValueError(
            f"Infeasible weight cap: {n} assets with max_weight={max_weight} "
            "cannot sum to 1."
        )
    for _ in range(1000):
        w = rng.dirichlet(np.ones(n) * 2.2)
        if np.max(w) <= max_weight + 1e-12:
            return w

    # Guaranteed feasible fallback: start from the cap and redistribute the
    # remaining mass randomly among assets that are below the cap.
    w = np.full(n, min(max_weight, 1.0 / n), dtype=float)
    deficit = 1.0 - w.sum()
    while deficit > 1e-12:
        eligible = np.flatnonzero(w < max_weight - 1e-12)
        if len(eligible) == 0:
            break
        proposal = rng.dirichlet(np.ones(len(eligible))) * deficit
        capacity = max_weight - w[eligible]
        add = np.minimum(proposal, capacity)
        w[eligible] += add
        deficit = 1.0 - w.sum()
    return w / w.sum()


def candidate_portfolios(expected, cov, n_candidates, max_weight=MAX_WEIGHT, seed=12345):
    """Build a feasible candidate surface under the investor's actual cap."""
    expected = np.asarray(expected, dtype=float)
    cov = nearest_psd(cov)
    n = len(expected)
    rng = np.random.default_rng(seed)

    if n * max_weight < 1.0 - 1e-12:
        raise ValueError(
            f"Infeasible max_weight={max_weight} for {n} assets; "
            "the cap must satisfy n * max_weight >= 1."
        )

    equal = np.repeat(1.0 / n, n)
    candidates = [equal]

    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(MIN_WEIGHT, max_weight)] * n

    def solve(start, objective):
        res = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 1500, "ftol": 1e-10, "disp": False},
        )
        if res.success and np.isfinite(res.fun):
            return project_to_capped_simplex(res.x, max_weight)
        return None

    # Return-tilted and minimum-volatility feasible starts are explicitly
    # included, matching the intended multi-start design.
    ret_start = solve(equal, lambda w: -(w @ expected))
    if ret_start is not None:
        candidates.append(ret_start)

    vol_start = solve(equal, lambda w: float(np.sqrt(max(w @ cov @ w, 0.0))))
    if vol_start is not None:
        candidates.append(vol_start)

    while len(candidates) < n_candidates:
        candidates.append(generate_candidate_weights(n, rng, max_weight))

    return np.array(candidates[:n_candidates])


def project_to_capped_simplex(x, cap):
    """Project a vector onto {w: sum(w)=1, 0<=w<=cap}."""
    x = np.asarray(x, dtype=float)
    if len(x) * cap < 1.0 - 1e-12:
        raise ValueError("Weight cap is infeasible for the number of assets.")

    # Find theta such that sum(clip(x - theta, 0, cap)) == 1.
    lo = float(np.min(x) - cap)
    hi = float(np.max(x))
    for _ in range(100):
        theta = 0.5 * (lo + hi)
        w = np.clip(x - theta, 0.0, cap)
        if w.sum() > 1.0:
            lo = theta
        else:
            hi = theta
    w = np.clip(x - 0.5 * (lo + hi), 0.0, cap)
    # Tiny floating-point correction while preserving the cap.
    deficit = 1.0 - w.sum()
    if abs(deficit) > 1e-12:
        if deficit > 0:
            room = cap - w
            for i in np.argsort(-room):
                add = min(deficit, room[i])
                w[i] += add
                deficit -= add
                if deficit <= 1e-12:
                    break
        else:
            excess = -deficit
            for i in np.argsort(-w):
                sub = min(excess, w[i])
                w[i] -= sub
                excess -= sub
                if excess <= 1e-12:
                    break
    return w / w.sum()


def portfolio_mc(weights, daily_returns, expected_returns, paths=MC_PATHS, days=252, seed=777):
    """Joint Monte Carlo preserving cross-stock covariance.

    The drift is based on the modelled annual expected return vector, while
    volatility/correlation comes from the historical log-return covariance.
    """
    weights = np.asarray(weights, dtype=float)
    expected_returns = np.asarray(expected_returns, dtype=float)

    log_returns = np.log1p(daily_returns.clip(lower=-0.999))
    cov_daily = nearest_psd(log_returns.cov().to_numpy())
    chol = np.linalg.cholesky(cov_daily + np.eye(len(expected_returns)) * 1e-12)

    annual_mu_log = np.log1p(np.clip(expected_returns, -0.95, 2.0))
    sigma2_daily = np.diag(cov_daily)
    # Convert annual expected simple return into a log-return drift.
    mu_daily = (annual_mu_log - 0.5 * sigma2_daily * TRADING_DAYS) / TRADING_DAYS

    rng = np.random.default_rng(seed)
    z = rng.normal(size=(paths, days, len(expected_returns)))
    shocks = z @ chol.T + mu_daily
    stock_growth = np.exp(shocks)
    stock_daily_returns = stock_growth - 1.0
    port_daily = np.einsum("pdk,k->pd", stock_daily_returns, weights)
    terminal = np.prod(1.0 + port_daily, axis=1) - 1.0

    q05 = float(np.quantile(terminal, 0.05))
    tail = terminal[terminal <= q05]
    return {
        "expected_return": float(terminal.mean()),
        "volatility": float(terminal.std(ddof=1)),
        "var_95": q05,
        "es_95": float(tail.mean()) if len(tail) else q05,
    }


def investor_lambda(age, risk_profile, horizon):
    """
    Convert investor characteristics into risk aversion.

    Risk profile remains the primary driver, but age has a deliberately
    stronger and nonlinear effect so that changing age produces a visible
    change in the optimizer's risk preference.

    The age multiplier is anchored at:
      18 -> 0.60x
      25 -> 0.75x
      35 -> 0.95x
      45 -> 1.15x
      55 -> 1.40x
      65 -> 1.70x
      75+ -> 2.00x

    Horizon is a secondary adjustment:
      1 year -> 1.10x
      20 years -> 1.00x
      40+ years -> 0.90x
    """
    base = RISK_LAMBDA[risk_profile]

    age = float(np.clip(age, 18, 100))
    horizon = float(np.clip(horizon, 1, 50))

    age_points = np.array([18, 25, 35, 45, 55, 65, 75, 100], dtype=float)
    age_multipliers = np.array([0.60, 0.75, 0.95, 1.15, 1.40, 1.70, 2.00, 2.00], dtype=float)
    age_multiplier = float(np.interp(age, age_points, age_multipliers))

    horizon_multiplier = 1.10 - 0.20 * ((horizon - 1.0) / 39.0)
    horizon_multiplier = float(np.clip(horizon_multiplier, 0.90, 1.10))

    return float(base * age_multiplier * horizon_multiplier)

def portfolio_objective(w, expected, cov, lam):
    """
    Risk-adjusted portfolio score.

    Return is rewarded, while three forms of risk are penalized:
    1. portfolio volatility from the full covariance matrix;
    2. a normal 5% downside proxy (1.645 * volatility);
    3. weighted average individual-stock volatility.

    The third term makes the investor's risk profile directly affect which
    stocks receive weight, rather than relying only on portfolio covariance.
    """
    portfolio_return = float(w @ expected)
    portfolio_vol = float(np.sqrt(max(w @ cov @ w, 0.0)))

    individual_vol = np.sqrt(np.maximum(np.diag(cov), 0.0))
    weighted_stock_vol = float(w @ individual_vol)

    downside_proxy = 1.645 * portfolio_vol
    downside_penalty = DOWNSIDE_PENALTY_RATIO * lam * downside_proxy
    concentration = float(np.sum(w ** 2))

    score = (
        portfolio_return
        - lam * portfolio_vol
        - downside_penalty
        - INDIVIDUAL_VOL_PENALTY * lam * weighted_stock_vol
        - CONCENTRATION_PENALTY * concentration
    )
    return score


def optimize_personalized_portfolio(expected, cov, lam, max_weight=MAX_WEIGHT):
    expected = np.asarray(expected, dtype=float)
    cov = nearest_psd(cov)
    n = len(expected)

    if n * max_weight < 1.0 - 1e-12:
        raise ValueError(
            f"Infeasible max_weight={max_weight} for {n} assets; "
            "the cap must satisfy n * max_weight >= 1."
        )

    objective = lambda w: -portfolio_objective(w, expected, cov, lam)
    bounds = [(MIN_WEIGHT, max_weight)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    equal = np.repeat(1.0 / n, n)
    starts = [equal]
    rng = np.random.default_rng(2026)

    # Explicit multi-start search: equal-weight, return-tilted,
    # minimum-volatility, plus one feasible random start.
    for start_obj in [
        lambda w: -(w @ expected),
        lambda w: float(np.sqrt(max(w @ cov @ w, 0.0))),
    ]:
        res = minimize(
            start_obj,
            equal,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1500, "ftol": 1e-10, "disp": False},
        )
        if res.success:
            starts.append(project_to_capped_simplex(res.x, max_weight))

    starts.append(generate_candidate_weights(n, rng, max_weight))

    best = None
    for start in starts:
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 2500, "ftol": 1e-12, "disp": False},
        )
        if result.success and np.isfinite(result.fun):
            if best is None or result.fun < best.fun:
                best = result

    if best is None:
        raise RuntimeError("Portfolio optimization failed to converge.")

    w = project_to_capped_simplex(best.x, max_weight)
    return w, -float(best.fun)


def build_score_surface(stats, lam):
    out = stats.copy()

    # Candidate-level MC statistics do not contain the individual stock
    # volatility term, so this surface uses the portfolio-level components
    # available for each candidate. The final optimizer still uses the full
    # individual-stock risk penalty.
    out["score"] = (
        out["expected_return"]
        - lam * out["volatility"]
        - DOWNSIDE_PENALTY_RATIO * lam * 1.645 * out["volatility"]
        - CONCENTRATION_PENALTY * out["concentration"]
    )
    return out


def build_cache():
    prices = pd.read_parquet(DATA_DIR / "nifty50_5y_prices.parquet")
    prices.index = pd.to_datetime(prices.index)
    if getattr(prices.index, "tz", None) is not None:
        prices.index = prices.index.tz_localize(None)
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()

    if "^NSEI" not in prices.columns:
        raise RuntimeError("The downloaded price file does not contain ^NSEI.")

    analytics, tickers, cov, models = fit_stock_models(prices)
    expected = analytics["expected_return"].to_numpy(float)

    # Compute returns first, then remove rows containing missing values.
    # Dropping price rows before pct_change would turn a one-day return into
    # a multi-day return whenever any stock had a gap on that date.
    daily = prices[tickers].pct_change(fill_method=None).dropna(how="any")
    candidates = candidate_portfolios(expected, cov, N_CANDIDATES, MAX_WEIGHT, seed=12345)

    candidate_stats = []
    for i, w in enumerate(candidates):
        mc = portfolio_mc(
            w,
            daily,
            expected,
            paths=MC_PATHS,
            days=TRADING_DAYS,
            seed=10000 + i,
        )
        candidate_stats.append({
            "candidate_id": i,
            "expected_return": mc["expected_return"],
            "volatility": mc["volatility"],
            "var_95": mc["var_95"],
            "es_95": mc["es_95"],
            "concentration": float(np.sum(w ** 2)),
        })

    candidate_stats_df = pd.DataFrame(candidate_stats)

    # Cache profile-specific candidate surfaces so the candidate shown to the
    # user is generated under the same concentration cap as the final portfolio.
    profile_candidates = {}
    profile_stats = {}
    for profile, cap in PROFILE_MAX_WEIGHT.items():
        pw = candidate_portfolios(expected, cov, N_CANDIDATES, cap, seed=seed_for(profile))
        ps = []
        for i, w in enumerate(pw):
            mc = portfolio_mc(
                w, daily, expected,
                paths=MC_PATHS, days=TRADING_DAYS,
                seed=20000 + seed_for(profile) % 10000 + i,
            )
            ps.append({
                "candidate_id": i,
                "expected_return": mc["expected_return"],
                "volatility": mc["volatility"],
                "var_95": mc["var_95"],
                "es_95": mc["es_95"],
                "concentration": float(np.sum(w ** 2)),
            })
        profile_candidates[profile] = pw
        profile_stats[profile] = pd.DataFrame(ps)

    cache = {
        "analytics": analytics,
        "tickers": tickers,
        "covariance": cov,
        # Legacy/default surface retained for compatibility.
        "candidate_weights": candidates,
        "candidate_stats": candidate_stats_df,
        # Correct profile-specific surfaces used by personalize().
        "candidate_weights_by_profile": profile_candidates,
        "candidate_stats_by_profile": profile_stats,
        "daily_returns": daily,
        "latest_date": prices.index.max(),
        "cache_version": 4,
        "stock_return_weight": STOCK_RETURN_WEIGHT,
        "market_return_weight": MARKET_RETURN_WEIGHT,
        "mc_paths": MC_PATHS,
        "config_hash": _config_signature(),
    }

    with open(DATA_DIR / "model_cache.pkl", "wb") as f:
        pickle.dump(cache, f)

    return cache


def load_cache():
    path = DATA_DIR / "model_cache.pkl"
    required_version = 4
    if not path.exists():
        return build_cache()

    with open(path, "rb") as f:
        cache = pickle.load(f)

    # Rebuild automatically whenever the cache was produced by an older model
    # shape, or ANY tunable constant (lambda table, weight caps, penalties,
    # MC paths, return blending, etc.) has changed since the cache was built.
    # Relying on config_hash (rather than only a manually-bumped version
    # number) prevents a stale cache from silently surviving a constants edit.
    if (
        cache.get("cache_version") != required_version
        or cache.get("config_hash") != _config_signature()
        or "candidate_weights_by_profile" not in cache
        or "candidate_stats_by_profile" not in cache
    ):
        return build_cache()

    return cache


def personalize(age, risk_profile, horizon, amount):
    cache = load_cache()
    analytics = cache["analytics"].copy()

    if risk_profile not in PROFILE_MAX_WEIGHT:
        raise ValueError(
            f"Unknown risk profile '{risk_profile}'. "
            f"Expected one of: {list(PROFILE_MAX_WEIGHT)}"
        )

    lam = investor_lambda(age, risk_profile, horizon)
    profile_max_weight = PROFILE_MAX_WEIGHT[risk_profile]

    expected = analytics["expected_return"].to_numpy(dtype=float)
    cov = cache["covariance"]

    # Risk profile changes both the risk-aversion coefficient and the
    # maximum concentration permitted in any individual stock.
    best_w, optimization_score = optimize_personalized_portfolio(
        expected,
        cov,
        lam,
        max_weight=profile_max_weight,
    )

    out = analytics.copy()
    out["weight"] = best_w
    out["allocation_inr"] = best_w * float(amount)

    # Use candidates generated with the same profile-specific cap as the
    # actual optimizer. This avoids matching a Low/Medium portfolio against
    # candidates that violate its concentration limit.
    candidate_weights_by_profile = cache.get("candidate_weights_by_profile")
    candidate_stats_by_profile = cache.get("candidate_stats_by_profile")
    if candidate_weights_by_profile and candidate_stats_by_profile:
        candidate_weights = candidate_weights_by_profile[risk_profile]
        base_stats = candidate_stats_by_profile[risk_profile]
    else:
        # Backward-compatible fallback for an older cache.
        candidate_weights = cache["candidate_weights"]
        base_stats = cache["candidate_stats"]

    stats = build_score_surface(base_stats, lam)
    distances = np.linalg.norm(candidate_weights - best_w, axis=1)
    nearest_idx = int(np.argmin(distances))
    stats["is_selected"] = False
    stats.loc[stats.index[nearest_idx], "is_selected"] = True

    # Final Monte Carlo is run on the actual optimized weights shown to the
    # user, not on the nearest cached candidate.
    selected_mc = portfolio_mc(
        best_w,
        cache["daily_returns"],
        expected,
        paths=MC_PATHS,
        days=TRADING_DAYS,
        seed=987654,
    )

    return {
        "portfolio": out.sort_values("weight", ascending=False).reset_index(drop=True),
        "selected_candidate": int(stats.iloc[nearest_idx]["candidate_id"]),
        "score": float(optimization_score),
        "lambda": float(lam),
        "age": float(age),
        "risk_profile": str(risk_profile),
        "horizon": float(horizon),
        "max_weight": float(profile_max_weight),
        "expected_return": float(selected_mc["expected_return"]),
        "volatility": float(selected_mc["volatility"]),
        "var_95": float(selected_mc["var_95"]),
        "es_95": float(selected_mc["es_95"]),
        "candidate_stats": stats,
        "latest_date": cache["latest_date"],
    }
