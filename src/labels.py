import pandas as pd

from config import FORWARD_DAYS


def create_forward_return_labels(
    prices: pd.DataFrame,
    forward_days: int = FORWARD_DAYS
) -> pd.Series:
    forward_returns = prices.shift(-forward_days) / prices - 1
    labels = forward_returns.stack(future_stack=True).rename("target")

    labels.index.names = ["date", "ticker"]

    return labels