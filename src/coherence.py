"""Unified topological coherence operator implementation with geometric integration."""

import numpy as np
from scipy.signal import hilbert
from scipy.stats import entropy
from typing import Any, Dict, Optional
from engines.signal_clarity import clean_covariance_rmt
from engines.phase_transitions import detect_phase_criticality
from sklearn.decomposition import PCA


def _ensure_2d(series: np.ndarray) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def _normalize_channels(series: np.ndarray) -> np.ndarray:
    standardized = np.zeros_like(series)
    for idx, row in enumerate(series):
        std = np.std(row)
        if std > 0:
            standardized[idx] = (row - np.mean(row)) / std
        else:
            standardized[idx] = row - np.mean(row)
    return standardized


class UnifiedPhaseAnalyzer:
    """
    Combines Hilbert phase extraction, PLV, PLI, Kuramoto sync, chiasm coherence, 
    and geometric invariants from Takens reconstruction.
    
    Mathematical Foundation:
    - Topological measures (c, r, H) detect phase-space coherence and synchronization
    - Geometric measures (radius, compactness, recurrence, divergence) capture 
      attractor structure and non-local correlations
    - Enhanced C-score integrates both topology and geometry:
      C_enhanced = α*c + (1-α)*r - β*H + γ*(geometric terms)
    """

    def __init__(self, alpha: float = 0.25, beta: float = 0.08, gamma: float = 0.1, hist_bins: int = 16):
        """
        Initialize analyzer with weighting parameters.
        
        Parameters:
        -----------
        alpha : float
            Weight for chiasm coherence (topology)
        beta : float
            Weight for entropy penalty
        gamma : float
            Weight for geometric features (new)
        hist_bins : int
            Histogram bins for entropy computation
        """
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.beta = float(max(0.0, beta))
        self.gamma = float(np.clip(gamma, 0.0, 1.0))
        self.hist_bins = max(8, hist_bins)

    def _delay_embed(self, series: np.ndarray, dimension: int = 4, delay: int = 2) -> np.ndarray:
        n = len(series)
        max_lag = (dimension - 1) * delay
        if n <= max_lag:
            # Fallback to simple channel replication if not enough length
            return np.vstack([series for _ in range(2)])
        indices = np.arange(n - max_lag)
        return np.vstack([series[indices + i * delay] for i in range(dimension)])

    def _compute_similarity_matrix(self, series: np.ndarray) -> np.ndarray:
        channels = _normalize_channels(series)
        if channels.shape[0] == 1:
            channels = self._delay_embed(channels[0])
        return np.corrcoef(channels)

    def _compute_chiasm_coherence(self, similarity: np.ndarray) -> float:
        """
        Compute anti-diagonal (chiasm) coherence.
        Encodes non-local correlations and boundary inversions.
        """
        m = similarity.shape[0]
        if m <= 1:
            return 1.0

        anti_vals = [similarity[i, m - 1 - i] for i in range(m // 2)]
        mu_chi = float(np.mean(anti_vals)) if anti_vals else 0.0
        off_diag_mask = ~np.eye(m, dtype=bool)
        mu_n = float(np.mean(np.abs(similarity[off_diag_mask]))) if m > 1 else 0.0
        return float(np.clip((mu_chi - mu_n + 1.0) / 2.0, 0.0, 1.0))

    def _compute_phase_matrix(self, series: np.ndarray) -> np.ndarray:
        analytic = hilbert(series, axis=1)
        phases = np.angle(analytic)
        return phases

    def _compute_sync(self, phases: np.ndarray) -> float:
        if phases.size == 0:
            return 0.0
        phase_order = np.abs(np.mean(np.exp(1j * phases), axis=0))
        return float(np.clip(np.mean(phase_order), 0.0, 1.0))

    def _compute_plv_pli(self, phases: np.ndarray) -> Dict[str, float]:
        m = phases.shape[0]
        if m <= 1:
            return {"plv": 1.0, "pli": 1.0}

        diffs = []
        for i in range(m):
            for j in range(i + 1, m):
                phase_diff = phases[i] - phases[j]
                diffs.append(phase_diff)
        diffs = np.vstack(diffs)
        plv = float(np.abs(np.mean(np.exp(1j * diffs))))
        pli = float(np.abs(np.mean(np.sign(np.sin(diffs)))))
        return {"plv": np.clip(plv, 0.0, 1.0), "pli": np.clip(pli, 0.0, 1.0)}

    def _compute_entropy(self, similarity: np.ndarray) -> float:
        if similarity.size <= 1:
            return 0.0

        off_diag = similarity[~np.eye(similarity.shape[0], dtype=bool)]
        hist, _ = np.histogram(off_diag, bins=self.hist_bins,
                       range=(-1.0, 1.0), density=False)
        prob = hist.astype(float) + 1e-6   # add noise floor before filtering
        prob = prob / prob.sum()            # normalize to true probability
        prob = prob[prob > 1e-7]
        if prob.size == 0:
            return 0.0

        ent = entropy(prob, base=2)
        return float(np.clip(ent / np.log2(self.hist_bins), 0.0, 1.0))
    
    def _compute_geometric_coherence(self, geometric_invariants: Dict) -> float:
        """
        Compute a unified geometric coherence score from invariants.
        
        Strategy:
        - High compactness (low C_geom) → trajectory is concentrated → coherent
        - High recurrence (high R_geom) → system revisits same regions → structured
        - Low divergence (low D_geom) → predictable dynamics → coherent
        
        Returns:
        --------
        float
            Geometric coherence score in [0, 1]
        """
        if not geometric_invariants:
            return 0.5
        
        # Normalize geometric features to [0, 1]
        compactness = float(geometric_invariants.get('compactness', 1.0))
        recurrence = float(geometric_invariants.get('recurrence_rate', 0.5))
        divergence = float(geometric_invariants.get('divergence_proxy', 1.0))
        
        # Normalize compactness (lower is better, but clip at reasonable range)
        compactness_score = 1.0 / (1.0 + np.clip(compactness, 0, 10))
        
        # Recurrence score (higher is better, already in [0,1])
        recurrence_score = np.clip(recurrence, 0.0, 1.0)
        
        # Divergence score (lower is better)
        divergence_score = 1.0 / (1.0 + np.clip(divergence, 0, 10))
        
        # Weighted combination
        g_score = 0.33 * compactness_score + 0.33 * recurrence_score + 0.33 * divergence_score
        
        return float(np.clip(g_score, 0.0, 1.0))

    def analyze(self, series, alpha: Optional[float] = None, beta: Optional[float] = None,
                geometric_invariants: Optional[Dict] = None) -> Dict[str, float]:
        """
        Comprehensive coherence analysis combining topology and geometry.
        
        Parameters:
        -----------
        series : array-like
            Input time series
        alpha : float, optional
            Override alpha weighting
        beta : float, optional
            Override beta weighting
        geometric_invariants : dict, optional
            Geometric invariants from Takens reconstruction
        
        Returns:
        --------
        dict
            Coherence metrics including enhanced C-score
        """
        arr = _ensure_2d(np.asarray(series, dtype=float))
        if arr.shape[0] == 1:
            arr = self._delay_embed(arr[0])

        # Topological measures
        similarity = self._compute_similarity_matrix(arr)
        c = self._compute_chiasm_coherence(similarity)
        phases = self._compute_phase_matrix(arr)
        kuramoto = self._compute_sync(phases)
        plv_pli = self._compute_plv_pli(phases)
        r = float(np.clip(
            0.2 * kuramoto +
            0.4 * plv_pli["plv"] +
            0.4 * plv_pli["pli"],
            0.0,
            1.0,
        ))
        H = self._compute_entropy(similarity)

        # Use provided or instance weights
        alpha = self.alpha if alpha is None else float(np.clip(alpha, 0.0, 1.0))
        beta = self.beta if beta is None else float(max(0.0, beta))
        
        # Base topological C-score
        C_base = float(np.clip(alpha * c + (1.0 - alpha) * r - beta * H, 0.0, 1.0))
        
        # Geometric enhancement
        result = {
            "alpha": alpha,
            "beta": beta,
            "c": round(c, 4),
            "r": round(r, 4),
            "kuramoto": round(kuramoto, 4),
            "plv": round(plv_pli["plv"], 4),
            "pli": round(plv_pli["pli"], 4),
            "H": round(H, 4),
            "C": round(C_base, 4),
        }
        
        # Integrate geometric features if available
        if geometric_invariants is not None:
            g_score = self._compute_geometric_coherence(geometric_invariants)
            result["gamma"] = float(self.gamma)
            result["geometric_score"] = round(g_score, 4)
            
            # Enhanced C-score: topological + geometric
            C_enhanced = float(np.clip(
                (1.0 - self.gamma) * C_base + self.gamma * g_score,
                0.0, 1.0
            ))
            result["C_enhanced"] = round(C_enhanced, 4)
            
            # Include individual geometric components
            result["compactness"] = round(geometric_invariants.get('compactness', 0.0), 4)
            result["recurrence_rate"] = round(geometric_invariants.get('recurrence_rate', 0.0), 4)
            result["divergence_proxy"] = round(geometric_invariants.get('divergence_proxy', 0.0), 4)
        
        return result


# === Kuramoto Extension on Takens Phase Space (Refined) ===
# Usage:
#   from src.coherence import SyncModule
#   sync = SyncModule()
#   report = sync.order_parameter(price_series)
class SyncModule:
    """Kuramoto synchronization on Takens-reconstructed attractor.

    This module lazily initializes the phase space reconstructor only when the
    `order_parameter` method is called, avoiding import-time circular dependencies.
    """

    def __init__(self, analyzer: Optional['UnifiedPhaseAnalyzer'] = None, use_takens: bool = True):
        self.analyzer = analyzer or UnifiedPhaseAnalyzer(gamma=0.12)
        self.use_takens = use_takens
        # Lazy reconstructor initialization to reduce import and circularity risk.
        self._reconstructor = None
        self.reconstructor = None

    def order_parameter(self, price_series: np.ndarray, window: int = 64) -> Dict[str, Any]:
        """Full Kuramoto metrics using existing pipeline."""
        series = np.asarray(price_series[-window:]).flatten()
        if len(series) < 10:
            raise ValueError("price_series window must have at least 10 samples")

        if self.use_takens and self._reconstructor is None:
            try:
                from src.phase_space_reconstructor import PhaseSpaceReconstructor
                self._reconstructor = PhaseSpaceReconstructor()
                self.reconstructor = self._reconstructor
            except Exception:
                self._reconstructor = None
                self.reconstructor = None

        cleaned = clean_covariance_rmt(series).get('cleaned_series', series)
        geometric_invariants = {}
        if self.use_takens and self._reconstructor is not None and len(cleaned) > 10:
            try:
                trajectory, geometric_invariants = self._reconstructor.reconstruct(cleaned)
            except Exception:
                trajectory = cleaned.reshape(-1, 1)
                geometric_invariants = {}
        else:
            trajectory = cleaned.reshape(-1, 1)

        if trajectory.ndim == 1 or trajectory.shape[1] == 1:
            phases = np.angle(np.exp(1j * trajectory.flatten()))
        else:
            pca = PCA(n_components=1)
            proj = pca.fit_transform(trajectory).flatten()
            phases = np.arctan2(proj, np.gradient(proj))

        report = self.analyzer.analyze(series, geometric_invariants=geometric_invariants)
        criticality = detect_phase_criticality(trajectory)
        critical_flag = criticality.get('approaching_criticality', False)

        global_r = float(np.abs(np.mean(np.exp(1j * phases))))
        phase_diffs = np.diff(phases)
        local_desync = float(np.mean(np.abs(phase_diffs) > (np.pi / 3))) if len(phase_diffs) > 0 else 0.0

        regime = "locked" if global_r > 0.78 else "searching" if global_r > 0.45 else "drifting"
        high_sync = global_r > 0.82
        low_desync = local_desync < 0.15
        high_desync = local_desync > 0.30

        if high_sync and critical_flag:
            alert = "PHASE_TRANSITION_RISK"
        elif high_sync and low_desync:
            alert = "HERDING_ALERT"
        elif high_desync or critical_flag:
            alert = "DESYNC_WARNING"
        else:
            alert = "NOMINAL"

        return {
            "kuramoto_r": round(global_r, 4),
            "collective_phase": round(float(np.angle(np.mean(np.exp(1j * phases)))), 4),
            "local_desync": round(local_desync, 4),
            "C_enhanced": report.get("C_enhanced", report.get("C", 0.0)),
            "regime": regime,
            "alert": alert,
            "geometric_invariants": geometric_invariants,
            "phase_criticality": criticality,
        }
