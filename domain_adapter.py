"""Domain adapter that converts raw inputs into phase-coherence reports with Takens integration."""

import json
import numpy as np
from typing import Any, Dict, List, Optional
from .coherence import UnifiedPhaseAnalyzer
from .geometry_engine import PhaseSpaceEngine
from .phase_space_reconstructor import PhaseSpaceReconstructor
from engines import (
    calculate_signal_clarity,
    evaluate_identity_resonance,
    calculate_infection_rate,
    calculate_entanglement_field,
    detect_phase_criticality,
)
from engines.signal_clarity import clean_covariance_rmt

POSITIVE_WORDS = {
    "coherent", "stable", "strong", "emergent", "aligned", "positive", "growth", "surge"
}
NEGATIVE_WORDS = {
    "drift", "uncertain", "fragile", "weak", "collapse", "noise", "fall", "crash"
}


def _text_to_series(text: str) -> np.ndarray:
    words = [w.lower() for w in text.split() if w.isalpha()]
    if len(words) == 0:
        return np.array([0.0])

    lengths = np.array([len(w) for w in words], dtype=float)
    sentiment = np.array([
        1.0 if w in POSITIVE_WORDS else -1.0 if w in NEGATIVE_WORDS else 0.0
        for w in words
    ], dtype=float)
    signal = 0.5 + 0.5 * sentiment
    weights = 1.0 + (lengths / (lengths.max() + 1e-10))
    return np.clip(signal * weights, 0.0, 2.0)


class DomainAdapter:
    """
    Converts market, time series, and narrative inputs into coherence reports.
    
    Enhanced with:
    - RMT-based signal cleaning
    - Takens delay embedding and phase space reconstruction
    - Geometric invariants for enhanced coherence scoring
    
    Use use_takens=True to enable the full geometry-enhanced pipeline.
    """

    def __init__(self, 
                 alpha: float = 0.25, 
                 beta: float = 0.08, 
                 gamma: float = 0.1,
                 geometry_dim: int = 3, 
                 geometry_delay: int = 2,
                 use_takens: bool = False):
        """
        Initialize the domain adapter.
        
        Parameters:
        -----------
        alpha : float
            Weight for topological chiasm coherence
        beta : float
            Weight for entropy penalty
        gamma : float
            Weight for geometric features (new)
        geometry_dim : int
            Embedding dimension for legacy PhaseSpaceEngine
        geometry_delay : int
            Delay for legacy PhaseSpaceEngine
        use_takens : bool
            If True, use enhanced Takens-based reconstruction pipeline
        """
        self.phase_analyzer = UnifiedPhaseAnalyzer(alpha=alpha, beta=beta, gamma=gamma)
        self.geometry_engine = PhaseSpaceEngine(dimension=geometry_dim, delay=geometry_delay)
        self.use_takens = use_takens
        
        if use_takens:
            self.takens_reconstructor = PhaseSpaceReconstructor(
                max_delay=30,
                max_dimension=8,
                delay_method='mutual_info',
                dimension_method='cao'
            )
        else:
            self.takens_reconstructor = None

    def _normalize_series(self, raw_series: Any) -> np.ndarray:
        arr = np.asarray(raw_series, dtype=float)
        if arr.ndim > 1 and arr.shape[0] > arr.shape[1]:
            arr = arr.T
        if arr.ndim == 0:
            arr = np.array([arr])
        if arr.ndim == 1:
            arr = arr
        return arr

    def _create_text_trajectory(self, text: str) -> np.ndarray:
        series = _text_to_series(text)
        if series.size < 5:
            series = np.pad(series, (0, 5 - series.size), mode="edge")
        return series

    def process(self, 
                input_data: Any, 
                source: Optional[str] = None,
                use_takens: Optional[bool] = None) -> Dict[str, Any]:
        """
        Main processing pipeline.
        
        Parameters:
        -----------
        input_data : Any
            Time series, text, or market data
        source : str, optional
            Data source type ('text', 'market', 'numeric')
        use_takens : bool, optional
            Override instance use_takens setting
        
        Returns:
        --------
        dict
            Comprehensive coherence report with optional geometric enhancements
        """
        if isinstance(input_data, str):
            source = source or "text"
            series = self._create_text_trajectory(input_data)
        elif isinstance(input_data, dict) and "close" in input_data:
            source = source or "market"
            series = self._normalize_series(input_data["close"])
        else:
            source = source or "numeric"
            series = self._normalize_series(input_data)

        # Determine which pipeline to use
        use_takens_pipeline = use_takens if use_takens is not None else self.use_takens
        
        # Initialize report structure
        report = {
            "source": source,
            "use_takens": use_takens_pipeline,
        }
        
        if use_takens_pipeline and len(series) > 10:
            # ===== ENHANCED TAKENS PIPELINE =====
            
            # Step 1: RMT cleaning
            rmt_result = clean_covariance_rmt(series, return_phase_space=False)
            cleaned_series = rmt_result['cleaned_series']
            
            # Step 2: Takens reconstruction on cleaned series
            try:
                trajectory, invariants = self.takens_reconstructor.reconstruct(cleaned_series)
                
                # Step 3: Phase analysis with geometric features
                phase_report = self.phase_analyzer.analyze(
                    series,  # Use original for topological analysis
                    geometric_invariants=invariants
                )
                
                # Extract geometric features for report
                geometric_features = {
                    'system_radius': invariants.get('system_radius', 0.0),
                    'compactness': invariants.get('compactness', 0.0),
                    'recurrence_rate': invariants.get('recurrence_rate', 0.0),
                    'divergence_proxy': invariants.get('divergence_proxy', 0.0),
                    'embedding_delay': invariants.get('delay', 1.0),
                    'embedding_dimension': invariants.get('dimension', 3.0),
                }
                
                report["phase_analysis"] = phase_report
                report["geometric_features"] = geometric_features
                report["rmt_cleaning"] = {
                    "mp_threshold": rmt_result['mp_threshold'],
                    "q_ratio": rmt_result['q_ratio'],
                    "num_signal_eigenvalues": int(np.sum(rmt_result['filtered_eigenvalues'] > 0))
                }
                
                # Use enhanced C-score if available
                c_score = phase_report.get('C_enhanced', phase_report.get('C', 0.0))
                regime_threshold_high = 0.65
                regime_threshold_low = 0.35
                
            except Exception as e:
                # Fallback to standard pipeline if Takens fails
                phase_report = self.phase_analyzer.analyze(series)
                report["phase_analysis"] = phase_report
                report["takens_error"] = str(e)
                c_score = phase_report['C']
                regime_threshold_high = 0.7
                regime_threshold_low = 0.35
        
        else:
            # ===== STANDARD PIPELINE (Backward Compatible) =====
            phase_report = self.phase_analyzer.analyze(series)
            geometry_report = self.geometry_engine.process_stream(series.tolist())
            
            report["phase_analysis"] = phase_report
            report["geometry_analysis"] = geometry_report
            
            c_score = phase_report['C']
            regime_threshold_high = 0.7
            regime_threshold_low = 0.35
        
        # Common metrics (both pipelines)
        signal_score = calculate_signal_clarity(np.asarray(series))
        identity_report = evaluate_identity_resonance(np.asarray(series))
        narrative_report = calculate_infection_rate(np.asarray(series))
        entanglement_report = calculate_entanglement_field(
            np.atleast_2d(series).T if np.ndim(series) == 1 else np.asarray(series)
        )
        phase_transition_report = detect_phase_criticality(np.asarray(series))

        # Regime determination
        if c_score >= regime_threshold_high:
            regime = "COHERENT"
        elif c_score <= regime_threshold_low:
            regime = "DRIFTING"
        else:
            regime = "TRANSITION"
        
        geometry_state = "CLOSED" if c_score >= 0.65 else "OPEN"

        report.update({
            "telemetry": {
                "signal_clarity": float(round(signal_score, 4)),
                "identity_resonance": identity_report,
                "narrative_infection": narrative_report,
                "structural_entanglement": entanglement_report,
                "phase_transition": phase_transition_report,
            },
            "coherence": {
                "C_score": float(round(c_score, 4)),
                "regime": regime,
                "geometry_state": geometry_state,
                "c_component": phase_report.get("c", 0.0),
                "r_component": phase_report.get("r", 0.0),
                "entropy_component": phase_report.get("H", 0.0),
            },
        })

        if source == "text":
            report["narrative_series"] = series.tolist()

        return report

    def process_market_dataframe(self, dataframe, use_takens: Optional[bool] = None) -> Dict[str, Any]:
        """Process a market dataframe."""
        if hasattr(dataframe, "get") and "close" in dataframe:
            series = dataframe.get("close")
        elif hasattr(dataframe, "loc") and "close" in dataframe.columns:
            series = dataframe["close"].astype(float).values
        else:
            series = np.asarray(dataframe, dtype=float)
        return self.process(series, source="market", use_takens=use_takens)
