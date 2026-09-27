from pathlib import Path
import time
import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# NIFTY 50 constituents (Yahoo Finance symbols)
NIFTY50 = [
    "ADANIENT.NS","ADANIPORTS.NS","APOLLOHOSP.NS","ASIANPAINT.NS",
    "AXISBANK.NS","BAJAJ-AUTO.NS","BAJFINANCE.NS","BAJAJFINSV.NS",
    "BEL.NS","BHARTIARTL.NS","CIPLA.NS","COALINDIA.NS","DRREDDY.NS",
    "EICHERMOT.NS","ETERNAL.NS","GRASIM.NS","HCLTECH.NS","HDFCBANK.NS",
    "HDFCLIFE.NS","HEROMOTOCO.NS","HINDALCO.NS","HINDUNILVR.NS",
    "ICICIBANK.NS","INDUSINDBK.NS","INFY.NS","ITC.NS","JIOFIN.NS",
    "JSWSTEEL.NS","KOTAKBANK.NS","LT.NS","M&M.NS","MARUTI.NS",
    "MAXHEALTH.NS","NESTLEIND.NS","NTPC.NS","ONGC.NS","POWERGRID.NS",
    "RELIANCE.NS","SBILIFE.NS","SBIN.NS","SHRIRAMFIN.NS","SUNPHARMA.NS",
    "TATACONSUM.NS","TATASTEEL.NS","TCS.NS","TECHM.NS","TITAN.NS",
    "TRENT.NS","ULTRACEMCO.NS","WIPRO.NS"
]

def main():
    print("Downloading 5 years of NIFTY 50 data from Yahoo Finance...")
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=5)

    successful = []
    failed = []

    frames = []
    for i, ticker in enumerate(NIFTY50, 1):
        print(f"[{i:02d}/{len(NIFTY50)}] {ticker}", end=" ... ")
        try:
            df = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if df.empty:
                raise ValueError("No data returned")
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close.name = ticker
            frames.append(close)
            successful.append(ticker)
            print("OK")
        except Exception as e:
            failed.append((ticker, str(e)))
            print(f"FAILED: {e}")
        time.sleep(0.15)

    nifty = yf.download(
        "^NSEI",
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if nifty.empty:
        raise RuntimeError("Could not download ^NSEI.")

    nifty_close = nifty["Close"]
    if isinstance(nifty_close, pd.DataFrame):
        nifty_close = nifty_close.iloc[:, 0]
    nifty_close.name = "^NSEI"

    prices = pd.concat(frames + [nifty_close], axis=1).sort_index()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)

    prices.to_parquet(DATA_DIR / "nifty50_5y_prices.parquet")
    prices.to_csv(DATA_DIR / "nifty50_5y_prices.csv")

    returns = prices.pct_change()
    returns.to_parquet(DATA_DIR / "nifty50_5y_returns.parquet")

    metadata = pd.DataFrame({
        "successful_ticker": successful + [ "^NSEI" ],
    })
    metadata.to_csv(DATA_DIR / "download_metadata.csv", index=False)

    print("\nDOWNLOAD COMPLETE")
    print(f"Successful stocks: {len(successful)} / {len(NIFTY50)}")
    if failed:
        print("\nFailed downloads:")
        for t, e in failed:
            print(f"  {t}: {e}")
    print(f"\nSaved to: {DATA_DIR}")

if __name__ == "__main__":
    main()
