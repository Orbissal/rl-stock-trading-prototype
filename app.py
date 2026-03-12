"""
app.py — Flask API Server
CM3070 Final Year Project — Intelligent Stock Trading Advisor

Endpoints:
    GET  /api/recommendation?ticker=AAPL&position=cash
    GET  /api/performance
    GET  /api/health

Usage (Windows):
    pip install flask flask-cors
    python app.py

Then open dashboard.html in your browser.

References:
    Mnih et al. (2015). Human-level control through deep reinforcement learning.
    Puiutta & Veith (2020). Explainable reinforcement learning: A survey.

DISCLAIMER: Research prototype for academic purposes only.
            Not financial advice.
"""

import os
import sys
import warnings
import numpy as np
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS

# ── Suppress TensorFlow noise ─────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# ── Import DQN components from existing codebase ──────────────────────────────
try:
    from dqn_trading_fast import DQNTradingEnvironment, DQNAgent, explain_action
    _src = "dqn_trading_fast"
except ImportError:
    from dqn_trading import DQNTradingEnvironment, DQNAgent, explain_action
    _src = "dqn_trading"

# ── Import data + state helpers from recommend.py ─────────────────────────────
from recommend import fetch_recent_data, load_model, get_current_state

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "results/dqn_AAPL_model.keras"
ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}
SUPPORTED_TICKERS = ["AAPL"]

# ── Pre-load model at startup (avoids reload on every request) ────────────────
print(f"\n  Loading DQN model from '{MODEL_PATH}'...")
try:
    _agent = load_model()
    print(f"  Model loaded  ✓  (source: {_src})\n")
except SystemExit:
    print("  ERROR: Could not load model. Ensure results/dqn_AAPL_model.keras exists.")
    sys.exit(1)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # Allow dashboard.html served from file:// to call this API


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_market_snapshot(data, state=None):
    """Return a dict of current market indicators for display."""
    closes  = data['Close'].values.astype(float)
    volumes = data['Volume'].values.astype(float)

    # RSI: use state[0] (the value the model actually used) so the snapshot
    # always matches the explain_action reasoning displayed in the UI.
    if state is not None:
        rsi = float(state[0] * 100)   # state[0] is normalised RSI in [0,1]
    else:
        deltas = np.diff(closes)
        seed   = deltas[:14]
        up     = seed[seed > 0].sum() / 14
        down   = -seed[seed < 0].sum() / 14
        rs     = up / (down + 1e-10)
        rsi    = float(100 - (100 / (1 + rs)))

    ma5  = float(np.mean(closes[-5:]))
    ma20 = float(np.mean(closes[-20:]))

    vol_20    = float(np.mean(volumes[-20:]))
    vol_today = float(volumes[-1])

    current_price = float(closes[-1])
    prev_price    = float(closes[-2])
    day_change_pct = (current_price - prev_price) / prev_price * 100

    return {
        "current_price":   round(current_price, 2),
        "day_change_pct":  round(day_change_pct, 2),
        "rsi":             round(rsi, 1),
        "rsi_signal":      "Oversold"   if rsi < 30 else
                           "Overbought" if rsi > 70 else "Neutral",
        "ma5":             round(ma5, 2),
        "ma20":            round(ma20, 2),
        "trend":           "Uptrend"    if ma5 > ma20 else "Downtrend",
        "volume_ratio":    round(vol_today / (vol_20 + 1e-10), 2),
        "high_52d":        round(float(np.max(closes)), 2),
        "low_52d":         round(float(np.min(closes)), 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    """Simple health check."""
    return jsonify({
        "status":    "ok",
        "model":     MODEL_PATH,
        "source":    _src,
        "timestamp": datetime.now().isoformat(),
        "tickers":   SUPPORTED_TICKERS,
    })


@app.route("/api/recommendation", methods=["GET"])
def recommendation():
    """
    Generate a live trading recommendation.

    Query params:
        ticker   : str  — stock symbol (default: AAPL)
        position : str  — 'cash' or 'holding' (default: cash)

    Returns JSON:
        ticker, action, action_name, confidence, q_values,
        reasons, market, timestamp
    """
    ticker   = request.args.get("ticker",   "AAPL").upper().strip()
    position = request.args.get("position", "cash").lower().strip()

    # Validate
    if ticker not in SUPPORTED_TICKERS:
        return jsonify({
            "error": f"Ticker '{ticker}' not supported. "
                     f"Supported: {SUPPORTED_TICKERS}"
        }), 400

    assume_holding = (position == "holding")

    try:
        # 1. Fetch live data
        data = fetch_recent_data(ticker, days=60)

        # 2. Build state vector
        state, env = get_current_state(data, assume_position=assume_holding)

        # 3. Run inference
        action   = _agent.get_action(state, training=False)
        q_values = _agent.online_network(
            np.array([state], dtype=np.float32), training=False
        ).numpy()[0]

        # 4. Generate explanation
        explanation = explain_action(state, action, q_values)

        # 5. Market snapshot
        market = compute_market_snapshot(data, state=state)

        return jsonify({
            "ticker":      ticker,
            "action":      int(action),
            "action_name": ACTION_NAMES[action],
            "confidence":  round(float(explanation["confidence"]), 1),
            "q_values": {
                "HOLD": round(float(q_values[0]), 4),
                "BUY":  round(float(q_values[1]), 4),
                "SELL": round(float(q_values[2]), 4),
            },
            "reasons":   explanation["reasons"],
            "position":  "holding" if assume_holding else "cash",
            "market":    market,
            "timestamp": datetime.now().isoformat(),
            "disclaimer": "Research prototype. Not financial advice.",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/performance", methods=["GET"])
def performance():
    """
    Return pre-computed multi-regime performance metrics.
    Data sourced from evaluate_regimes.py output (regime_analysis.txt).
    No live computation needed — these are the evaluated results.
    """
    return jsonify({
        "description": "Multi-regime evaluation: Tabular Q-Learning vs DQN vs Buy-and-Hold (AAPL)",
        "regimes": [
            {
                "name":    "2022 Bear Market",
                "period":  "2022-01-01 to 2022-12-31",
                "context": "High volatility. AAPL fell ~27%.",
                "results": {
                    "tabular": {
                        "total_return": -38.80,
                        "sharpe":       -1.57,
                        "max_drawdown": -45.49,
                        "trades":        17,
                    },
                    "dqn": {
                        "total_return":  43.59,
                        "sharpe":         1.61,
                        "max_drawdown": -12.53,
                        "trades":         69,
                    },
                    "buy_and_hold": {
                        "total_return": -28.20,
                        "sharpe":        -0.76,
                        "max_drawdown": -30.35,
                        "trades":          2,
                    },
                },
            },
            {
                "name":    "2023 Bull Run",
                "period":  "2023-01-01 to 2023-12-31",
                "context": "Strong uptrend. AAPL gained ~49%.",
                "results": {
                    "tabular": {
                        "total_return":  39.74,
                        "sharpe":         1.96,
                        "max_drawdown": -12.27,
                        "trades":          7,
                    },
                    "dqn": {
                        "total_return":  42.63,
                        "sharpe":         2.59,
                        "max_drawdown":  -7.54,
                        "trades":         47,
                    },
                    "buy_and_hold": {
                        "total_return":  54.80,
                        "sharpe":         2.32,
                        "max_drawdown": -14.93,
                        "trades":          2,
                    },
                },
            },
            {
                "name":    "2024 Holdout (Test Set)",
                "period":  "2024-01-01 to 2024-10-31",
                "context": "Standard 20% holdout. Neither agent saw this data during training.",
                "results": {
                    "tabular": {
                        "total_return":  17.67,
                        "sharpe":         1.02,
                        "max_drawdown": -15.53,
                        "trades":          3,
                    },
                    "dqn": {
                        "total_return":   5.67,
                        "sharpe":         0.46,
                        "max_drawdown": -11.42,
                        "trades":         43,
                    },
                    "buy_and_hold": {
                        "total_return":  24.42,
                        "sharpe":         1.24,
                        "max_drawdown": -15.35,
                        "trades":          2,
                    },
                },
            },
        ],
        "summary": {
            "dqn_wins_regimes":     2,
            "tabular_wins_regimes": 1,
            "key_finding": (
                "DQN outperforms buy-and-hold in bear markets (+43.59% vs -28.20% in 2022) "
                "but underperforms in trending markets due to overtrading. "
                "Consistent with Fischer (2018) and Sun et al. (2023)."
            ),
        },
        "source": "evaluate_regimes.py — CM3070 FYP",
    })


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("  ══════════════════════════════════════════════")
    print("  CM3070 Intelligent Stock Trading Advisor API")
    print("  ══════════════════════════════════════════════")
    print("  Endpoints:")
    print("    GET /api/health")
    print("    GET /api/recommendation?ticker=AAPL&position=cash")
    print("    GET /api/performance")
    print()
    print("  Open dashboard.html in your browser to use the UI.")
    print("  Press Ctrl+C to stop the server.")
    print("  ══════════════════════════════════════════════\n")

    app.run(host="127.0.0.1", port=5000, debug=False)
