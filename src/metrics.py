import numpy as np
import pandas as pd


def calculate_drawdown(equity_curve: pd.Series) -> pd.Series:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return drawdown


def calculate_performance_metrics(
    returns: pd.Series,
    equity_curve: pd.Series
) -> pd.Series:
    total_return = equity_curve.iloc[-1] - 1

    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25

    if years > 0:
        cagr = equity_curve.iloc[-1] ** (1 / years) - 1
    else:
        cagr = np.nan

    volatility = returns.std() * np.sqrt(252)

    if returns.std() != 0:
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
    else:
        sharpe_ratio = np.nan

    drawdown = calculate_drawdown(equity_curve)
    max_drawdown = drawdown.min()

    if max_drawdown != 0:
        calmar_ratio = cagr / abs(max_drawdown)
    else:
        calmar_ratio = np.nan

    metrics = pd.Series({
        "Total Return": total_return,
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Calmar Ratio": calmar_ratio
    })

    return metrics