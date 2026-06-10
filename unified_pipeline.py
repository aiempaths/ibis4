#!/usr/bin/env python3
"""Main entrypoint for the Universal Topological Coherence Operator."""

import argparse
import json
import os
import sys
from typing import Dict, Any, Optional

import requests
import numpy as np

from src.domain_adapter import DomainAdapter
from validation.validation_suite import run_validation_suite


MARKET_IDS = {
    "eth": "ethereum",
    "sol": "solana",
    "bitcoin": "bitcoin",
    "btc": "bitcoin"
}


def save_report(report: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as out:
        json.dump(report, out, indent=2)
    print(f"✅ Saved report to {path}")


def fetch_market_series(symbol: str, days: int = 30) -> Dict[str, Any]:
    symbol_key = symbol.lower()
    asset_id = MARKET_IDS.get(symbol_key, symbol_key)
    url = f"https://api.coingecko.com/api/v3/coins/{asset_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    body = response.json()

    prices = body.get("prices", [])
    if not prices:
        raise RuntimeError(f"No market history returned for {symbol}")

    timestamps = [int(p[0]) for p in prices]
    closes = [float(p[1]) for p in prices]
    return {"symbol": symbol.upper(), "timestamps": timestamps, "close": closes}


def run_synthetic_examples(adapter: DomainAdapter) -> Dict[str, Any]:
    t = np.linspace(0, 10, 250)
    coherent = np.sin(2 * np.pi * 1.0 * t) + 0.05 * np.random.normal(size=t.shape)
    drifting = np.linspace(0.0, 1.0, 250) + 0.5 * np.random.normal(size=t.shape)
    regime_shift = np.concatenate([
        np.sin(2 * np.pi * 0.8 * t[:100]),
        0.3 * np.random.normal(size=80),
        np.sin(2 * np.pi * 1.2 * t[100:])
    ])

    return {
        "coherent": adapter.process(coherent, source="synthetic_coherent"),
        "drifting": adapter.process(drifting, source="synthetic_drifting"),
        "regime_shift": adapter.process(regime_shift, source="synthetic_regime_shift"),
    }


def run_text_example(adapter: DomainAdapter, text: str) -> Dict[str, Any]:
    return {"text_input": adapter.process(text, source="text")}


def main(args: Optional[Any] = None) -> int:
    parser = argparse.ArgumentParser(description="Universal Topological Coherence Operator runner")
    parser.add_argument("--validate", action="store_true", help="Run the validation suite")
    parser.add_argument("--market", type=str, help="Fetch real market data for a symbol (e.g. ETH, SOL)")
    parser.add_argument("--text", type=str, help="Run narrative/text coherence analysis")
    parser.add_argument("--output", type=str, default="report.json", help="Path to save the JSON report")
    parser.add_argument("--days", type=int, default=30, help="Lookback window in days for market fetch")
    parser.add_argument("--verbose", action="store_true", help="Print verbose report output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parsed = parser.parse_args(args=args)

    np.random.seed(parsed.seed)
    adapter = DomainAdapter()
    report: Dict[str, Any] = {"version": "1.0", "reports": {}}

    try:
        if parsed.validate:
            validation_report = run_validation_suite(seed=parsed.seed, verbose=parsed.verbose)
            save_report(validation_report, parsed.output)
            if parsed.verbose:
                print(json.dumps(validation_report, indent=2))
            return 0

        if parsed.market:
            market_payload = fetch_market_series(parsed.market, days=parsed.days)
            market_report = adapter.process_market_dataframe(market_payload)
            report["reports"][f"market_{parsed.market.upper()}"] = market_report

        if parsed.text:
            report["reports"]["narrative"] = run_text_example(adapter, parsed.text)["text_input"]

        if not parsed.market and not parsed.text:
            report["reports"]["synthetic"] = run_synthetic_examples(adapter)

        save_report(report, parsed.output)
        print(json.dumps(report, indent=2))
        return 0

    except requests.RequestException as re:
        print(f"❌ Network error while fetching market data: {re}")
        return 2
    except Exception as exc:
        print(f"❌ Error executing pipeline: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
