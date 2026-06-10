import numpy as np
from typing import List, Dict, Any

class PhaseSpaceEngine:
    """
    A domain-agnostic engine that reconstructs the phase space of an arbitrary 
    numerical time-series to detect structural anomalies and system coherence.
    """
    def __init__(self, dimension: int = 3, delay: int = 2, target_metric: str = "euclidean"):
        self.dimension = dimension  # Embedding dimension (d)
        self.delay = delay          # Time lag (tau)
        self.target_metric = target_metric

    def reconstruct_space(self, series: np.ndarray) -> np.ndarray:
        """
        Transforms a 1D array into a d-dimensional trajectory matrix.
        Using Takens' Embedding Theorem: Y_t = [X_t, X_{t-tau}, ..., X_{t-(d-1)tau}]
        """
        n = len(series)
        max_lag = (self.dimension - 1) * self.delay
        if n <= max_lag:
            raise ValueError(f"Time series length ({n}) must be greater than maximum lag ({max_lag})")
            
        # Dynamically generate phase space vectors
        indices = np.arange(n - max_lag)
        vectors = np.column_stack([
            series[indices + i * self.delay] for i in range(self.dimension)
        ])
        return vectors

    def analyze_coherence(self, vectors: np.ndarray) -> Dict[str, float]:
        """
        Measures the geometric stability/entropy of the current attractor shape.
        Returns a dictionary with a normalized Coherence Index and structural metrics.
        """
        if len(vectors) < 3:
            return {"coherence_index": 1.0, "dispersion": 0.0}

        # Calculate step-by-step velocities along the attractor trajectory
        diffs = np.diff(vectors, axis=0)
        velocities = np.linalg.norm(diffs, axis=1)
        
        # Calculate localized geometric dispersion (variance of trajectory steps)
        mean_velocity = np.mean(velocities)
        velocity_variance = np.var(velocities) if len(velocities) > 1 else 0.0
        
        # Determine global density bounding radius
        centroid = np.mean(vectors, axis=0)
        distances_to_centroid = np.linalg.norm(vectors - centroid, axis=1)
        radius = np.max(distances_to_centroid) if len(distances_to_centroid) > 0 else 1.0
        
        # Map variance and divergence into a clean 0.0 to 1.0 Coherence Score
        # High volatility/unpredictability drops the score close to 0
        stability = 1.0 / (1.0 + velocity_variance)
        compactness = 1.0 / (1.0 + (radius / (mean_velocity + 1e-6)))
        coherence_index = float(np.clip(stability * 0.7 + compactness * 0.3, 0.0, 1.0))

        return {
            "coherence_index": round(coherence_index, 4),
            "trajectory_velocity": float(round(mean_velocity, 4)),
            "system_radius": float(round(radius, 4))
        }

    def process_stream(self, data_stream: List[float]) -> Dict[str, Any]:
        """
        High-level pipeline entrypoint for TaaS consumers.
        """
        try:
            arr = np.array(data_stream, dtype=float)
            # Filter out constants or extreme dead noise
            if np.all(arr == arr[0]):
                return {"coherence_index": 1.0, "status": "static_input"}
                
            vectors = self.reconstruct_space(arr)
            metrics = self.analyze_coherence(vectors)
            metrics["status"] = "success"
            return metrics
        except Exception as e:
            return {"coherence_index": 0.0, "status": f"error: {str(e)}"}