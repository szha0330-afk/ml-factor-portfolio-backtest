import numpy as np
import pandas as pd

from config import TOP_N, TRANSACTION_COST, BENCHMARK


def build_portfolio_weights(
    prices: pd.DataFrame,
    predictions: pd.DataFrame,
    top_n: int = TOP_N
) -> pd.DataFrame:
    weights = pd.DataFrame(
        np.nan,
        index=prices.index,
        columns=prices.columns
    )

    for date, prediction_group in predictions.groupby("date"):
        if date not in weights.index:
            continue

        selected = (
            prediction_group
            .sort_values("predicted_forward_return", ascending=False)
            .head(top_n)
        )

        weights.loc[date, :] = 0.0

        if len(selected) > 0:
            selected_tickers = selected["ticker"].tolist()
            weights.loc[date, selected_tickers] = 1 / len(selected_tickers)

    weights = weights.ffill()
    weights = weights.fillna(0.0)

    return weights


def calculate_portfolio_returns(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    transaction_cost: float = TRANSACTION_COST
) -> pd.DataFrame:
    asset_returns = prices.pct_change().fillna(0)

    shifted_weights = weights.shift(1).fillna(0)
    gross_returns = (shifted_weights * asset_returns).sum(axis=1)

    turnover = weights.diff().abs().sum(axis=1).fillna(0)
    costs = turnover * transaction_cost

    net_returns = gross_returns - costs
    equity_curve = (1 + net_returns).cumprod()

    result = pd.DataFrame(index=prices.index)
    result["gross_return"] = gross_returns
    result["transaction_cost"] = costs
    result["portfolio_return"] = net_returns
    result["turnover"] = turnover
    result["equity_curve"] = equity_curve

    return result


def calculate_benchmark_returns(
    prices: pd.DataFrame,
    benchmark: str = BENCHMARK
) -> pd.DataFrame:
    benchmark_returns = prices[benchmark].pct_change().fillna(0)
    benchmark_equity = (1 + benchmark_returns).cumprod()

    result = pd.DataFrame(index=prices.index)
    result["benchmark_return"] = benchmark_returns
    result["benchmark_equity"] = benchmark_equity

    return result