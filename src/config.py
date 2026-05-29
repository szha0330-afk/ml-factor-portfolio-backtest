TICKERS = [
    "SPY",  # S&P 500
    "QQQ",  # Nasdaq 100
    "IWM",  # Russell 2000
    "TLT",  # Long-Term Treasury
    "GLD",  # Gold
    "EFA",  # Developed Markets
    "EEM",  # Emerging Markets
    "VNQ",  # Real Estate
    "XLE",  # Energy
    "XLK"   # Technology
]

BENCHMARK = "SPY"

START_DATE = "2010-01-01"
END_DATE = "2024-01-01"

FORWARD_DAYS = 21
TOP_N = 3

TRANSACTION_COST = 0.001

MIN_TRAINING_ROWS = 500

RANDOM_STATE = 42
N_ESTIMATORS = 300
MAX_DEPTH = 5
MIN_SAMPLES_LEAF = 5

RESULTS_DIR = "results"

FEATURE_COLUMNS = [
    "momentum_21",
    "momentum_63",
    "momentum_126",
    "momentum_252",
    "volatility_21",
    "volatility_63",
    "drawdown_63",
    "ma_spread_50_200",
    "rsi_14"
]