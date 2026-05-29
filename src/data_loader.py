import pandas as pd
import yfinance as yf

from config import TICKERS, START_DATE, END_DATE


def load_price_data(
    tickers: list[str] = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE
) -> pd.DataFrame:
    raw_data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    if raw_data.empty:
        raise ValueError("No price data downloaded. Check tickers or date range.")

    if isinstance(raw_data.columns, pd.MultiIndex):
        prices = raw_data["Close"]
    else:
        prices = raw_data[["Close"]]
        prices.columns = tickers

    prices = prices.reindex(columns=tickers)
    prices = prices.dropna(how="all")
    prices = prices.ffill()
    prices = prices.dropna()

    return prices


if __name__ == "__main__":
    prices = load_price_data()
    print(prices.head())