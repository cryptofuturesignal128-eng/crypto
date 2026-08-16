import os
import time
import logging
from datetime import datetime, timezone

import requests
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

BINANCE_URL = "https://fapi.binance.com"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TOP_COINS = 100
MIN_SCORE = 80

SCAN_INTERVAL = 15 * 60

REQUEST_TIMEOUT = 15

# Maximum number of signals sent in one scan
MAX_SIGNALS_PER_SCAN = 5


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("crypto-signal-bot")


# ============================================================
# SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent": "CryptoSignalBot/1.0"
})


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Telegram variables are missing.")
        return False

    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }

    try:

        response = session.post(
            url,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        if not data.get("ok"):
            logger.error(
                "Telegram error: %s",
                data
            )
            return False

        return True

    except Exception as e:

        logger.error(
            "Telegram request failed: %s",
            e
        )

        return False


# ============================================================
# BINANCE REQUEST
# ============================================================

def binance_get(endpoint, params=None):

    url = BINANCE_URL + endpoint

    response = session.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    data = response.json()

    if isinstance(data, dict) and "code" in data:
        raise RuntimeError(
            f"Binance API error: {data}"
        )

    return data


# ============================================================
# EXCHANGE INFO
# ============================================================

def get_valid_symbols():

    data = binance_get(
        "/fapi/v1/exchangeInfo"
    )

    symbols = set()

    for item in data.get("symbols", []):

        if (
            item.get("contractType") == "PERPETUAL"
            and item.get("quoteAsset") == "USDT"
            and item.get("status") == "TRADING"
        ):
            symbols.add(item["symbol"])

    logger.info(
        "Found %d valid USDT perpetuals.",
        len(symbols)
    )

    return symbols


# ============================================================
# TOP 100
# ============================================================

def get_top_symbols(valid_symbols):

    data = binance_get(
        "/fapi/v1/ticker/24hr"
    )

    coins = []

    for item in data:

        symbol = item.get("symbol")

        if symbol not in valid_symbols:
            continue

        try:
            quote_volume = float(
                item.get("quoteVolume", 0)
            )
        except:
            continue

        if quote_volume <= 0:
            continue

        coins.append(
            (
                symbol,
                quote_volume
            )
        )

    coins.sort(
        key=lambda x: x[1],
        reverse=True
    )

    top = [
        symbol
        for symbol, volume
        in coins[:TOP_COINS]
    ]

    logger.info(
        "Scanning top %d coins.",
        len(top)
    )

    return top


# ============================================================
# KLINES
# ============================================================

def get_klines(
    symbol,
    interval,
    limit=250
):

    data = binance_get(
        "/fapi/v1/klines",
        {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
    )

    if not data:
        raise RuntimeError(
            f"No candle data for {symbol}"
        )

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore"
    ]

    df = pd.DataFrame(
        data,
        columns=columns
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["ema20"] = (
        df["close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    df["ema50"] = (
        df["close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    df["ema200"] = (
        df["close"]
        .ewm(
            span=200,
            adjust=False
        )
        .mean()
    )

    # --------------------------------------------------------
    # RSI 14
    # --------------------------------------------------------

    delta = df["close"].diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    df["rsi"] = (
        100 -
        (100 / (1 + rs))
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema12 = (
        df["close"]
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema26 = (
        df["close"]
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    df["macd"] = (
        ema12 - ema26
    )

    df["macd_signal"] = (
        df["macd"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    df["macd_hist"] = (
        df["macd"] -
        df["macd_signal"]
    )

    # --------------------------------------------------------
    # ATR 14
    # --------------------------------------------------------

    previous_close = (
        df["close"].shift(1)
    )

    tr1 = (
        df["high"] -
        df["low"]
    )

    tr2 = (
        df["high"] -
        previous_close
    ).abs()

    tr3 = (
        df["low"] -
        previous_close
    ).abs()

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3
        ],
        axis=1
    ).max(axis=1)

    df["atr"] = (
        true_range
        .rolling(14)
        .mean()
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    df["volume_avg"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    # --------------------------------------------------------
    # Candle body / volatility
    # --------------------------------------------------------

    df["candle_range"] = (
        df["high"] -
        df["low"]
    )

    df["atr_percent"] = (
        df["atr"] /
        df["close"]
    ) * 100

    return df


# ============================================================
# SIGNAL ANALYSIS
# ============================================================

def analyze_symbol(symbol):

    df15 = get_klines(
        symbol,
        "15m",
        250
    )

    df1h = get_klines(
        symbol,
        "1h",
        250
    )

    df15 = calculate_indicators(
        df15
    )

    df1h = calculate_indicators(
        df1h
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Use last CLOSED candle.
    # -1 can still be forming.
    # --------------------------------------------------------

    candle15 = df15.iloc[-2]
    candle1h = df1h.iloc[-2]

    values = [
        candle15["close"],
        candle15["ema20"],
        candle15["ema50"],
        candle15["ema200"],
        candle15["rsi"],
        candle15["macd"],
        candle15["macd_signal"],
        candle15["atr"],
        candle15["volume_avg"],
        candle1h["ema50"],
        candle1h["ema200"]
    ]

    if not all(
        np.isfinite(x)
        for x in values
    ):
        return None

    long_score = 0
    short_score = 0

    long_reasons = []
    short_reasons = []

    # ========================================================
    # 1H TREND
    # ========================================================

    if (
        candle1h["ema50"] >
        candle1h["ema200"]
    ):

        long_score += 20

        long_reasons.append(
            "1H EMA50 > EMA200"
        )

    elif (
        candle1h["ema50"] <
        candle1h["ema200"]
    ):

        short_score += 20

        short_reasons.append(
            "1H EMA50 < EMA200"
        )

    # ========================================================
    # 15M EMA STRUCTURE
    # ========================================================

    if (
        candle15["ema20"] >
        candle15["ema50"] >
        candle15["ema200"]
    ):

        long_score += 25

        long_reasons.append(
            "15M EMA20 > EMA50 > EMA200"
        )

    elif (
        candle15["ema20"] <
        candle15["ema50"] <
        candle15["ema200"]
    ):

        short_score += 25

        short_reasons.append(
            "15M EMA20 < EMA50 < EMA200"
        )

    # ========================================================
    # PRICE VS EMA20
    # ========================================================

    if (
        candle15["close"] >
        candle15["ema20"]
    ):

        long_score += 10

        long_reasons.append(
            "Price above EMA20"
        )

    elif (
        candle15["close"] <
        candle15["ema20"]
    ):

        short_score += 10

        short_reasons.append(
            "Price below EMA20"
        )

    # ========================================================
    # RSI
    # ========================================================

    rsi = float(
        candle15["rsi"]
    )

    if 50 <= rsi <= 68:

        long_score += 15

        long_reasons.append(
            f"RSI {rsi:.1f}"
        )

    elif 32 <= rsi < 50:

        short_score += 15

        short_reasons.append(
            f"RSI {rsi:.1f}"
        )

    # ========================================================
    # MACD
    # ========================================================

    if (
        candle15["macd"] >
        candle15["macd_signal"]
    ):

        long_score += 15

        long_reasons.append(
            "MACD bullish"
        )

    elif (
        candle15["macd"] <
        candle15["macd_signal"]
    ):

        short_score += 15

        short_reasons.append(
            "MACD bearish"
        )

    # ========================================================
    # VOLUME
    # ========================================================

    volume_ratio = (
        candle15["volume"] /
        candle15["volume_avg"]
    )

    if volume_ratio >= 1.0:

        # Give volume confirmation to
        # whichever direction is currently stronger.

        if long_score > short_score:

            long_score += 10

            long_reasons.append(
                f"Volume {volume_ratio:.2f}x average"
            )

        elif short_score > long_score:

            short_score += 10

            short_reasons.append(
                f"Volume {volume_ratio:.2f}x average"
            )

    # ========================================================
    # DETERMINE SIGNAL
    # ========================================================

    if long_score >= MIN_SCORE:

        signal = "LONG"

        score = long_score

        reasons = long_reasons

    elif short_score >= MIN_SCORE:

        signal = "SHORT"

        score = short_score

        reasons = short_reasons

    else:

        return None

    # ========================================================
    # ENTRY
    # ========================================================

    entry = float(
        candle15["close"]
    )

    atr = float(
        candle15["atr"]
    )

    if atr <= 0:
        return None

    # ========================================================
    # SL / TP
    #
    # SL = 1.5 ATR
    # TP1 = 1R
    # TP2 = 2R
    # TP3 = 3R
    # ========================================================

    if signal == "LONG":

        stop_loss = (
            entry -
            (atr * 1.5)
        )

        risk = (
            entry -
            stop_loss
        )

        tp1 = entry + risk
        tp2 = entry + (risk * 2)
        tp3 = entry + (risk * 3)

    else:

        stop_loss = (
            entry +
            (atr * 1.5)
        )

        risk = (
            stop_loss -
            entry
        )

        tp1 = entry - risk
        tp2 = entry - (risk * 2)
        tp3 = entry - (risk * 3)

    # ========================================================
    # RESULT
    # ========================================================

    return {
        "symbol": symbol,
        "signal": signal,
        "score": score,
        "entry": entry,
        "sl": stop_loss,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "rsi": rsi,
        "volume_ratio": volume_ratio,
        "atr_percent": float(
            candle15["atr_percent"]
        ),
        "reasons": reasons
    }


# ============================================================
# FORMAT PRICE
# ============================================================

def format_price(price):

    if price >= 1000:
        return f"{price:,.2f}"

    if price >= 1:
        return f"{price:.4f}"

    if price >= 0.01:
        return f"{price:.6f}"

    return f"{price:.8f}"


# ============================================================
# FORMAT TELEGRAM MESSAGE
# ============================================================

def format_signal(signal):

    if signal["signal"] == "LONG":
        direction = "🟢 LONG"
    else:
        direction = "🔴 SHORT"

    reasons = "\n".join(
        "✅ " + reason
        for reason in signal["reasons"]
    )

    now = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M UTC"
    )

    return (
        "🚨 CRYPTO FUTURES SIGNAL\n\n"

        f"{direction}\n"
        f"🪙 {signal['symbol']}\n\n"

        f"⭐ Score: {signal['score']}/100\n"
        f"📊 RSI: {signal['rsi']:.1f}\n"
        f"📊 Volume: "
        f"{signal['volume_ratio']:.2f}x average\n"
        f"📊 ATR: "
        f"{signal['atr_percent']:.2f}%\n\n"

        "💰 ENTRY\n"
        f"{format_price(signal['entry'])}\n\n"

        "🛑 STOP LOSS\n"
        f"{format_price(signal['sl'])}\n\n"

        "🎯 TAKE PROFIT\n"
        f"TP1: {format_price(signal['tp1'])}\n"
        f"TP2: {format_price(signal['tp2'])}\n"
        f"TP3: {format_price(signal['tp3'])}\n\n"

        "📈 CONFIRMATIONS\n"
        f"{reasons}\n\n"

        "⏱ Timeframe: 15M\n"
        "🔎 Trend: 1H confirmation\n\n"

        f"🕐 {now}\n\n"

        "⚠️ Educational signal only. "
        "Use proper risk management."
    )


# ============================================================
# SCANNER
# ============================================================

def scan_market(valid_symbols):

    try:

        top_symbols = get_top_symbols(
            valid_symbols
        )

    except Exception as e:

        logger.error(
            "Could not get top symbols: %s",
            e
        )

        return []

    signals = []

    for index, symbol in enumerate(
        top_symbols,
        start=1
    ):

        try:

            result = analyze_symbol(
                symbol
            )

            if result:

                signals.append(
                    result
                )

                logger.info(
                    "[%d/%d] %s -> %s %d",
                    index,
                    len(top_symbols),
                    symbol,
                    result["signal"],
                    result["score"]
                )

        except Exception as e:

            logger.warning(
                "%s analysis failed: %s",
                symbol,
                e
            )

        # Avoid unnecessary API pressure
        time.sleep(0.08)

    signals.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return signals


# ============================================================
# DUPLICATE PROTECTION
# ============================================================

sent_signals = {}


def signal_key(signal):

    # Round entry so tiny price differences
    # don't create duplicate alerts.

    entry_bucket = round(
        signal["entry"],
        4
    )

    return (
        signal["symbol"],
        signal["signal"],
        entry_bucket
    )


def should_send(signal):

    key = signal_key(
        signal
    )

    current_time = time.time()

    previous_time = sent_signals.get(
        key
    )

    # Do not repeat same setup
    # within 4 hours.

    if previous_time:

        if (
            current_time -
            previous_time
        ) < 4 * 60 * 60:

            return False

    sent_signals[key] = current_time

    # Prevent memory growing forever.
    if len(sent_signals) > 1000:

        oldest = sorted(
            sent_signals.items(),
            key=lambda x: x[1]
        )[:300]

        for key_to_delete, _ in oldest:

            del sent_signals[
                key_to_delete
            ]

    return True


# ============================================================
# STARTUP MESSAGE
# ============================================================

def send_startup_message():

    message = (
        "🤖 CRYPTO SIGNAL BOT ONLINE\n\n"

        "📊 Market: Binance USDT-M Futures\n"
        "🪙 Universe: Top 100 by 24H volume\n"
        "⏱ Entry: 15M\n"
        "🔎 Confirmation: 1H\n"
        "📈 EMA: 20 / 50 / 200\n"
        "📊 RSI: 14\n"
        "📊 MACD\n"
        "📊 Volume\n"
        "📊 ATR\n"
        f"⭐ Minimum Score: {MIN_SCORE}/100\n\n"

        "Only qualifying LONG/SHORT setups "
        "will be sent."
    )

    send_telegram(
        message
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not TELEGRAM_TOKEN:

        raise RuntimeError(
            "TELEGRAM_TOKEN is not set."
        )

    if not TELEGRAM_CHAT_ID:

        raise RuntimeError(
            "TELEGRAM_CHAT_ID is not set."
        )

    logger.info(
        "Starting Crypto Signal Bot..."
    )

    # Get valid symbols once at startup
    try:

        valid_symbols = (
            get_valid_symbols()
        )

    except Exception as e:

        raise RuntimeError(
            f"Could not connect to Binance: {e}"
        )

    send_startup_message()

    while True:

        cycle_start = time.time()

        logger.info(
            "========== NEW SCAN =========="
        )

        try:

            signals = scan_market(
                valid_symbols
            )

            logger.info(
                "Found %d qualifying signals.",
                len(signals)
            )

            sent_count = 0

            for signal in signals:

                if sent_count >= MAX_SIGNALS_PER_SCAN:
                    break

                if not should_send(
                    signal
                ):
                    continue

                message = format_signal(
                    signal
                )

                if send_telegram(
                    message
                ):

                    sent_count += 1

                    logger.info(
                        "Sent %s %s signal. Score=%d",
                        signal["symbol"],
                        signal["signal"],
                        signal["score"]
                    )

            if sent_count == 0:

                logger.info(
                    "No new qualifying signals."
                )

        except Exception as e:

            logger.exception(
                "Scan error: %s",
                e
            )

        elapsed = (
            time.time() -
            cycle_start
        )

        sleep_time = max(
            30,
            SCAN_INTERVAL - elapsed
        )

        logger.info(
            "Scan finished in %.1f seconds. "
            "Next scan in %.1f minutes.",
            elapsed,
            sleep_time / 60
        )

        time.sleep(
            sleep_time
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
