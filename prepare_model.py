from portfolio_model import build_cache, MC_PATHS, N_CANDIDATES

if __name__ == "__main__":
    print("Preparing ML + Monte Carlo cache...")
    cache = build_cache()

    print("\nMODEL CACHE READY")
    print(f"Stocks: {len(cache['tickers'])}")
    print(f"Candidate portfolios: {N_CANDIDATES}")
    print(f"Monte Carlo paths per candidate: {MC_PATHS:,}")
    print("Expected returns: 80% stock geometric return + 20% NIFTY + small ML tilt")
    print("Risk objective: return - portfolio volatility - downside risk - individual-stock volatility - concentration")
    print("Age has a material effect on lambda; horizon has a smaller effect.")
    print("Saved: data/model_cache.pkl")
