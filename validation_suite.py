"""Validation suite for the Universal Topological Coherence Operator."""

import json
import os
from typing import Dict, Any

import numpy as np

from src.domain_adapter import DomainAdapter
from src.geometry_engine import PhaseSpaceEngine
from src.phase_space_reconstructor import PhaseSpaceReconstructor
from engines.signal_clarity import clean_covariance_rmt

VALIDATION_OUTPUT = "validation_results.json"


def _synthetic_sine(length: int = 250) -> np.ndarray:
    t = np.linspace(0.0, 10.0, length)
    return np.sin(2.0 * np.pi * 1.0 * t) + 0.05 * np.random.normal(size=length)


def _synthetic_drifting_noise(length: int = 250) -> np.ndarray:
    return np.linspace(0.0, 1.0, length) + 0.5 * np.random.normal(size=length)


def _synthetic_regime_shift(length: int = 250) -> np.ndarray:
    first = np.sin(2.0 * np.pi * 0.8 * np.linspace(0.0, 4.0, length // 3))
    second = 0.4 * np.random.normal(size=length // 3)
    third = np.sin(2.0 * np.pi * 1.5 * np.linspace(0.0, 4.0, length - 2 * (length // 3)))
    return np.concatenate([first, second, third])


def _summarize(values):
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _compute_alignment_score(coherent_mean: float, drifting_mean: float, geometry_gain: float, ablation_gap: float) -> int:
    score = 0
    score += 2 if coherent_mean >= 0.7 else 0
    score += 2 if drifting_mean <= 0.35 else 0
    score += 2 if (coherent_mean - drifting_mean) >= 0.3 else 0
    score += 2 if geometry_gain >= 0.1 else 0
    score += 2 if ablation_gap >= 0.08 else 0
    return min(score, 10)


def _run_experiment(adapter: DomainAdapter, series: np.ndarray) -> Dict[str, Any]:
    result = adapter.process(series, source="validation")
    return {
        "C": result["coherence"]["C_score"],
        "phase_analysis": result["phase_analysis"],
        "geometry": result["geometry_analysis"],
        "telemetry": result["telemetry"],
        "coherence": result["coherence"],
    }


def _write_markdown_report(results: Dict[str, Any], path: str = "validation/validation_report.md") -> None:
    lines = [
        "# Validation Report: Universal Topological Coherence Operator",
        "",
        "## Synthetic Discrimination Results",
        "",
        "| Category | Mean C | Mean c | Mean r | Mean H | Mean Geometry | Regime |",
        "|---|---|---|---|---|---|---|",
        f"| Coherent | {results['coherent']['mean']:.4f} | {results['coherent']['c_mean']:.4f} | {results['coherent']['r_mean']:.4f} | {results['coherent']['H_mean']:.4f} | {results['coherent']['geometry_mean']:.4f} | {results['coherent']['regime']} |",
        f"| Drifting | {results['drifting']['mean']:.4f} | {results['drifting']['c_mean']:.4f} | {results['drifting']['r_mean']:.4f} | {results['drifting']['H_mean']:.4f} | {results['drifting']['geometry_mean']:.4f} | {results['drifting']['regime']} |",
        "",
        "## Ablation Summary",
        "",
        f"- Chiasm-only mean C: {results['ablation']['chiasm_only']['mean']:.4f}",
        f"- Sync-only mean C: {results['ablation']['sync_only']['mean']:.4f}",
        f"- Geometry coherence mean: {results['ablation']['geometry_coherence']['mean']:.4f}",
        "",
        "## Alignment Assessment",
        "",
        f"- Coherent vs drifting ΔC: {results['discrimination_power']['delta_C']:.4f}",
        f"- Alignment score: {results['alignment_score']} / 10",
        f"- Coherent target (>0.70): {results['summary']['target_coherent_gt_0_7']}",
        f"- Drifting target (<0.35): {results['summary']['target_drifting_lt_0_35']}",
        "",
        "## Recommendations",
        "",
        "- Apply the same operator to real Solana/ETH market sequences with 30-day windows.",
        "- Compare geometry coherence and phase coherence alongside price returns.",
        "- Use the operator to flag regime transitions before volatility bursts.",
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as md:
        md.write("\n".join(lines))


def run_validation_suite(seed: int = 42, verbose: bool = False) -> Dict[str, Any]:
    np.random.seed(seed)
    adapter = DomainAdapter()
    geometry_engine = PhaseSpaceEngine()

    coherent = [_run_experiment(adapter, _synthetic_sine()) for _ in range(10)]
    drifting = [_run_experiment(adapter, _synthetic_drifting_noise()) for _ in range(10)]
    regime_shift = [_run_experiment(adapter, _synthetic_regime_shift()) for _ in range(10)]

    coherent_scores = [x["C"] for x in coherent]
    drifting_scores = [x["C"] for x in drifting]
    regime_scores = [x["C"] for x in regime_shift]
    coherent_geometry = [x["geometry"]["coherence_index"] for x in coherent]
    drifting_geometry = [x["geometry"]["coherence_index"] for x in drifting]

    coherent_c = [x["phase_analysis"]["c"] for x in coherent]
    coherent_r = [x["phase_analysis"]["r"] for x in coherent]
    coherent_H = [x["phase_analysis"]["H"] for x in coherent]
    drifting_c = [x["phase_analysis"]["c"] for x in drifting]
    drifting_r = [x["phase_analysis"]["r"] for x in drifting]
    drifting_H = [x["phase_analysis"]["H"] for x in drifting]
    coherent_regimes = [x["coherence"]["regime"] for x in coherent]
    drifting_regimes = [x["coherence"]["regime"] for x in drifting]

    # Ablation tests
    coherent_chiasm = [adapter.phase_analyzer.analyze(series=_synthetic_sine(), alpha=1.0, beta=0.0)["C"] for _ in range(10)]
    coherent_sync = [adapter.phase_analyzer.analyze(series=_synthetic_sine(), alpha=0.0, beta=0.0)["C"] for _ in range(10)]
    geometry_coherent = [geometry_engine.process_stream(_synthetic_sine().tolist())["coherence_index"] for _ in range(10)]

    geometry_gain = float(np.mean(geometry_coherent) - np.mean(coherent_scores))
    ablation_gap = float(abs(np.mean(coherent_chiasm) - np.mean(coherent_sync)))
    alignment_score = _compute_alignment_score(np.mean(coherent_scores), np.mean(drifting_scores), geometry_gain, ablation_gap)

    results = {
        "metadata": {
            "description": "Validation of the Universal Topological Coherence Operator against synthetic domains.",
            "coherence_formula": "C = αc + (1-α)r - βH",
            "seed": seed,
        },
        "coherent": {
            **_summarize(coherent_scores),
            "c_mean": float(np.mean(coherent_c)),
            "r_mean": float(np.mean(coherent_r)),
            "H_mean": float(np.mean(coherent_H)),
            "geometry_mean": float(np.mean(coherent_geometry)),
            "regime": max(set(coherent_regimes), key=coherent_regimes.count) if coherent_regimes else "UNKNOWN",
        },
        "drifting": {
            **_summarize(drifting_scores),
            "c_mean": float(np.mean(drifting_c)),
            "r_mean": float(np.mean(drifting_r)),
            "H_mean": float(np.mean(drifting_H)),
            "geometry_mean": float(np.mean(drifting_geometry)),
            "regime": max(set(drifting_regimes), key=drifting_regimes.count) if drifting_regimes else "UNKNOWN",
        },
        "regime_shift": _summarize(regime_scores),
        "ablation": {
            "chiasm_only": _summarize(coherent_chiasm),
            "sync_only": _summarize(coherent_sync),
            "geometry_coherence": _summarize(geometry_coherent),
        },
        "discrimination_power": {
            "delta_C": float(np.mean(coherent_scores) - np.mean(drifting_scores)),
            "coherent_mean": float(np.mean(coherent_scores)),
            "drifting_mean": float(np.mean(drifting_scores)),
        },
        "alignment_score": alignment_score,
        "summary": {
            "target_coherent_gt_0_7": bool(np.mean(coherent_scores) >= 0.7),
            "target_drifting_lt_0_35": bool(np.mean(drifting_scores) <= 0.35),
            "geometry_improves": geometry_gain >= 0.1,
            "ablation_gap_significant": ablation_gap >= 0.08,
        },
    }

    os.makedirs("validation", exist_ok=True)
    with open(os.path.join("validation", VALIDATION_OUTPUT), "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    _write_markdown_report(results)

    if verbose:
        print(json.dumps(results, indent=2))

    print("\n=== Validation Summary ===")
    print(f"Coherent mean C: {results['coherent']['mean']:.4f}")
    print(f"Drifting mean C: {results['drifting']['mean']:.4f}")
    print(f"Delta C: {results['discrimination_power']['delta_C']:.4f}")
    print(f"Alignment Score: {alignment_score}/10")
    print("==========================\n")

    return results
