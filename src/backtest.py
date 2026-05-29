import os
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    TICKERS,
    BENCHMARK,
    START_DATE,
    END_DATE,
    FORWARD_DAYS,
    TOP_N,
    MIN_TRAINING_ROWS,
    FEATURE_COLUMNS,
    RESULTS_DIR
)
from data_loader import load_price_data
from features import create_feature_matrix
from labels import create_forward_return_labels
from model import train_model, predict_cross_section
from portfolio import (
    build_portfolio_weights,
    calculate_portfolio_returns,
    calculate_benchmark_returns
)
from metrics import calculate_performance_metrics, calculate_drawdown


def get_month_end_trading_dates(prices: pd.DataFrame) -> pd.DatetimeIndex:
    month_end_dates = prices.groupby(
        [prices.index.year, prices.index.month]
    ).tail(1).index

    return month_end_dates


def run_backtest():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Downloading price data...")
    prices = load_price_data(TICKERS, START_DATE, END_DATE)

    print("Creating factor features...")
    feature_matrix = create_feature_matrix(prices)

    print("Creating forward return labels...")
    labels = create_forward_return_labels(prices, FORWARD_DAYS)

    dataset = feature_matrix.join(labels)
    dataset = dataset.dropna()

    rebalance_dates = get_month_end_trading_dates(prices)

    prediction_records = []

    print("Running walk-forward ML backtest...")

    for date in rebalance_dates:
        if date not in prices.index:
            continue

        date_position = prices.index.get_loc(date)

        if date_position < 300:
            continue

        cutoff_position = max(0, date_position - FORWARD_DAYS)
        cutoff_date = prices.index[cutoff_position]

        train_data = dataset[
            dataset.index.get_level_values("date") <= cutoff_date
        ]

        if len(train_data) < MIN_TRAINING_ROWS:
            continue

        try:
            current_features = feature_matrix.xs(date, level="date")
        except KeyError:
            continue

        current_features = current_features.dropna()

        if current_features.empty:
            continue

        x_train = train_data[FEATURE_COLUMNS]
        y_train = train_data["target"]

        model = train_model(x_train, y_train)

        predictions = predict_cross_section(
            model,
            current_features[FEATURE_COLUMNS]
        )

        for ticker, predicted_return in predictions.items():
            prediction_records.append({
                "date": date,
                "ticker": ticker,
                "predicted_forward_return": predicted_return
            })

    if len(prediction_records) == 0:
        raise ValueError("No predictions generated. Check feature and training data.")

    predictions_df = pd.DataFrame(prediction_records)

    print("Building portfolio weights...")
    weights = build_portfolio_weights(
        prices,
        predictions_df,
        top_n=TOP_N
    )

    print("Calculating portfolio returns...")
    portfolio_result = calculate_portfolio_returns(prices, weights)
    benchmark_result = calculate_benchmark_returns(prices, BENCHMARK)

    combined = pd.concat(
        [
            portfolio_result,
            benchmark_result
        ],
        axis=1
    )

    portfolio_metrics = calculate_performance_metrics(
        combined["portfolio_return"],
        combined["equity_curve"]
    )

    benchmark_metrics = calculate_performance_metrics(
        combined["benchmark_return"],
        combined["benchmark_equity"]
    )

    performance_report = pd.DataFrame({
        "ML Factor Portfolio": portfolio_metrics,
        "Benchmark": benchmark_metrics
    })

    predictions_df.to_csv(f"{RESULTS_DIR}/predictions.csv", index=False)
    weights.to_csv(f"{RESULTS_DIR}/weights.csv")
    combined.to_csv(f"{RESULTS_DIR}/backtest_results.csv")
    performance_report.to_csv(f"{RESULTS_DIR}/performance_report.csv")

    plt.figure(figsize=(10, 6))
    plt.plot(
        combined.index,
        combined["equity_curve"],
        label="ML Factor Portfolio"
    )
    plt.plot(
        combined.index,
        combined["benchmark_equity"],
        label=f"{BENCHMARK} Buy and Hold"
    )
    plt.title("ML Factor Portfolio vs Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.savefig(f"{RESULTS_DIR}/equity_curve.png")
    plt.close()

    portfolio_drawdown = calculate_drawdown(combined["equity_curve"])
    benchmark_drawdown = calculate_drawdown(combined["benchmark_equity"])

    plt.figure(figsize=(10, 6))
    plt.plot(
        portfolio_drawdown.index,
        portfolio_drawdown,
        label="ML Factor Portfolio Drawdown"
    )
    plt.plot(
        benchmark_drawdown.index,
        benchmark_drawdown,
        label=f"{BENCHMARK} Drawdown"
    )
    plt.title("Drawdown Comparison")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.savefig(f"{RESULTS_DIR}/drawdown.png")
    plt.close()

    print("\nBacktest Completed")
    print("------------------")
    print(performance_report)


if __name__ == "__main__":
    run_backtest()