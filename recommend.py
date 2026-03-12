"""
recommend.py — Live Trading Recommendation CLI
CM3070 Final Year Project — Intelligent Stock Trading Advisor

Usage:
    python recommend.py AAPL
    python recommend.py MSFT
    python recommend.py GOOGL

What it does:
    1. Downloads the last 60 days of market data for the given ticker
    2. Loads the trained DQN model from results/dqn_AAPL_model.keras
    3. Constructs the 11-dimensional state vector from current market data
    4. Runs the model and outputs an explainable recommendation

This script demonstrates the Financial Advisor Bot functionality:
a trained RL agent providing explainable BUY / HOLD / SELL recommendations
on live market data for any US equity.

References:
    Mnih et al. (2015). Human-level control through deep reinforcement learning.
    Wilder (1978). New concepts in technical trading systems.
    Puiutta & Veith (2020). Explainable reinforcement learning: A survey.

DISCLAIMER: This is a research prototype for academic purposes only.
            It does not constitute financial advice.
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Suppress TF noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Import our DQN components
# Use dqn_trading_fast if available (predict() fix applied), else dqn_trading
try:
    from dqn_trading_fast import (
        DQNTradingEnvironment, DQNAgent, explain_action
    )
    _src = "dqn_trading_fast"
except ImportError:
    from dqn_trading import (
        DQNTradingEnvironment, DQNAgent, explain_action
    )
    _src = "dqn_trading"


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH   = "results/dqn_AAPL_model.keras"
LOOKBACK     = 60      # Days of history needed to compute all indicators
ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}
ACTION_EMOJI = {0: "⏸ ", 1: "📈", 2: "📉"}


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

def fetch_recent_data(ticker, days=LOOKBACK):
    """
    Download the last `days` trading days for the given ticker.
    Requires internet connection (Yahoo Finance via yfinance).
    """
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    end   = datetime.today()
    start = end - timedelta(days=days * 2)   # Extra buffer for weekends/holidays

    print(f"  Fetching {ticker} data from Yahoo Finance...")
    data = yf.download(ticker, start=start.strftime('%Y-%m-%d'),
                       end=end.strftime('%Y-%m-%d'), progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if len(data) < 30:
        print(f"ERROR: Only {len(data)} days returned for {ticker}.")
        print("Check the ticker symbol and your internet connection.")
        sys.exit(1)

    # Keep only the last `days` rows
    data = data.tail(days).reset_index(drop=True)
    print(f"  Loaded {len(data)} trading days  ✓")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    """Load trained DQN from disk."""
    if not Path(MODEL_PATH).exists():
        print(f"\nERROR: Model not found at '{MODEL_PATH}'")
        print("Run 'python dqn_trading.py' first to train and save the model.")
        sys.exit(1)

    agent = DQNAgent(state_size=DQNTradingEnvironment.STATE_SIZE, n_actions=3)
    agent.load(MODEL_PATH)
    agent.epsilon = 0.0   # Pure exploitation — no random actions
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# STATE CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def get_current_state(data, assume_position=False):
    """
    Build the 11-dimensional state vector from the most recent market data.

    We use DQNTradingEnvironment's own feature engineering methods to ensure
    the state vector is identical to what the agent saw during training.
    No reimplementation — reuse the exact same code.

    Args:
        data: DataFrame with at least 'Close' and 'Volume' columns
        assume_position: bool, whether to assume the user holds stock (True)
                         or is in cash (False). Default: cash.

    Returns:
        state: numpy array of shape (11,)
        env:   DQNTradingEnvironment instance (for feature method access)
    """
    env = DQNTradingEnvironment(data, initial_capital=10000)

    # Advance the environment to the last available day
    # so that _get_state() uses all available history
    env.current_step = len(data) - 2   # -2 because step() will advance by 1

    # Override position to reflect user's actual situation
    if assume_position:
        env.shares = 1      # Treat as holding
        env.capital = 0.0
    else:
        env.shares = 0
        env.capital = 10000.0

    state = env._get_state()
    return state, env


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

def print_recommendation(ticker, state, action, q_values, data):
    """Print a formatted, human-readable recommendation."""

    explanation = explain_action(state, action, q_values)
    action_name = ACTION_NAMES[action]
    emoji       = ACTION_EMOJI[action]

    current_price = float(data['Close'].iloc[-1])
    prev_price    = float(data['Close'].iloc[-2])
    day_change    = (current_price - prev_price) / prev_price * 100

    # Colour coding for terminal (works on most terminals)
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

    action_colour = GREEN if action == 1 else (RED if action == 2 else YELLOW)
    day_colour    = GREEN if day_change >= 0 else RED
    day_sign      = "+" if day_change >= 0 else ""

    width = 58

    print()
    print(BOLD + "═" * width + RESET)
    print(BOLD + f"  {ticker} — AI Trading Recommendation" + RESET)
    print(f"  {datetime.today().strftime('%d %B %Y')}")
    print("═" * width)
    print()
    print(f"  Current Price : {BOLD}${current_price:.2f}{RESET}  "
          f"({day_colour}{day_sign}{day_change:.2f}% today{RESET})")
    print()
    print(f"  Recommendation : "
          f"{action_colour}{BOLD}{emoji} {action_name}{RESET}")
    print(f"  Confidence     : {BOLD}{explanation['confidence']:.0f}%{RESET}")
    print()
    print("─" * width)
    print(f"  {BLUE}Reasoning:{RESET}")
    for reason in explanation['reasons']:
        print(f"    • {reason}")
    print()
    print("─" * width)
    print(f"  {BLUE}Q-Values (agent's action preferences):{RESET}")
    q = explanation['q_values']
    for act_name, val in q.items():
        bar_len = max(0, int((val - min(q.values())) /
                             (max(q.values()) - min(q.values()) + 1e-10) * 20))
        bar = "█" * bar_len
        highlight = BOLD if act_name == action_name else ""
        print(f"    {highlight}{act_name:<6}{RESET}  {val:>7.3f}  {BLUE}{bar}{RESET}")
    print()
    print("─" * width)
    print(f"  {YELLOW}⚠  Research prototype. Not financial advice.{RESET}")
    print("═" * width)
    print()


def print_market_snapshot(ticker, data):
    """Print a quick summary of current market conditions."""

    closes  = data['Close'].values.astype(float)
    volumes = data['Volume'].values.astype(float)

    rsi_period = 14
    deltas = np.diff(closes)
    seed   = deltas[:rsi_period]
    up     = seed[seed > 0].sum() / rsi_period
    down   = -seed[seed < 0].sum() / rsi_period
    rs     = up / (down + 1e-10)
    rsi    = 100 - (100 / (1 + rs))

    ma5  = np.mean(closes[-5:])
    ma20 = np.mean(closes[-20:])
    trend = "Uptrend ↑" if ma5 > ma20 else "Downtrend ↓"

    vol_20 = np.mean(volumes[-20:])
    vol_today = volumes[-1]
    vol_rel = vol_today / (vol_20 + 1e-10)

    print()
    print(f"  Market Snapshot ({ticker}, last {len(data)} days):")
    print(f"    RSI (14):      {rsi:.1f}  "
          f"{'[Oversold]' if rsi < 30 else '[Overbought]' if rsi > 70 else '[Neutral]'}")
    print(f"    Trend (5/20d): {trend}")
    print(f"    Volume ratio:  {vol_rel:.2f}x 20-day average")
    print(f"    52-day high:   ${max(closes):.2f}")
    print(f"    52-day low:    ${min(closes):.2f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Parse ticker from command line
    if len(sys.argv) < 2:
        print("\nUsage: python recommend.py <TICKER>")
        print("Example: python recommend.py AAPL")
        print("         python recommend.py MSFT")
        print("         python recommend.py GOOGL")
        sys.exit(0)

    ticker = sys.argv[1].upper().strip()

    # ── Optional: ask user if they hold the stock
    assume_position = False
    if len(sys.argv) >= 3 and sys.argv[2].lower() in ('--holding', '-h', 'holding'):
        assume_position = True

    print()
    print(f"  Intelligent Stock Trading Advisor — CM3070 FYP")
    print(f"  Analysing {ticker}...")
    print(f"  (Using model source: {_src})")
    print()

    # ── Fetch data
    data = fetch_recent_data(ticker, days=LOOKBACK)

    # ── Load model
    print(f"  Loading trained DQN model from '{MODEL_PATH}'...")
    agent = load_model()
    print(f"  Model loaded  ✓")

    # ── Construct state
    state, env = get_current_state(data, assume_position=assume_position)

    # ── Get recommendation
    action   = agent.get_action(state, training=False)
    q_values = agent.online_network(
        np.array([state], dtype=np.float32), training=False
    ).numpy()[0]

    # ── Print market snapshot
    print_market_snapshot(ticker, data)

    # ── Print recommendation
    print_recommendation(ticker, state, action, q_values, data)

    # ── Position note
    if assume_position:
        print("  Note: Recommendation generated assuming you currently HOLD this stock.")
    else:
        print("  Note: Recommendation generated assuming you are currently in CASH.")
        print("  Run with '--holding' flag if you already hold this stock:")
        print(f"    python recommend.py {ticker} --holding")
    print()


if __name__ == "__main__":
    main()
