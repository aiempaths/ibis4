#!/usr/bin/env python3
"""
Phase Space Reconstructor: Takens Embedding & Geometric Invariants
====================================================================

This module implements Takens delay embedding theorem as a foundational 
preprocessor for quantum circuits and topological coherence analysis.

Mathematical Foundation:
- Takens' Embedding Theorem: A d-dimensional attractor can be reconstructed 
  from a 1D time series using delay embedding Y_t = [X_t, X_{t-τ}, ..., X_{t-(d-1)τ}]
  if d ≥ 2*D_A + 1, where D_A is the attractor dimension.

- Adaptive Delay Selection: Use mutual information (Shannon) or autocorrelation 
  to identify the first local minimum—this τ maximizes information gain.

- Embedding Dimension: Use false nearest neighbors (FNN) or Cao's method 
  to estimate minimal embedding dimension avoiding overfitting.

- Geometric Invariants: Capture emergent structure:
  * System Radius: Maximum distance from trajectory centroid (compactness)
  * Trajectory Compactness: Concentration of trajectory around center
  * Recurrence Rate: Fraction of near-return pairs (phase space density)
  * Approximate Divergence: Rate of information spread (Lyapunov proxy)

References:
- Takens, F. (1981). "Detecting strange attractors in turbulence"
- Fraser & Swinney (1986). "Independent coordinates for strange attractors 
  from mutual information"
- Cao, L. (1997). "Practical method for determining the minimum embedding dimension"
"""

import numpy as np
from scipy.signal import correlate
from scipy.stats import entropy
from typing import Dict, Tuple, Optional


class PhaseSpaceReconstructor:
    """
    Reconstructs the phase space of a 1D time series using Takens delay embedding.
    
    Features:
    - Adaptive delay selection (mutual information or autocorrelation)
    - Embedding dimension estimation (FNN-like or Cao's method)
    - Geometric invariants computation
    - Non-local correlation → boundary inversion → emergent geometry pipeline
    
    Usage Example:
    --------
    reconstructor = PhaseSpaceReconstructor()
    trajectory, invariants = reconstructor.reconstruct(time_series)
    
    # Access individual components
    radius = invariants['system_radius']
    compactness = invariants['compactness']
    divergence = invariants['divergence_proxy']
    """
    
    def __init__(self, 
                 max_delay: int = 20,
                 max_dimension: int = 10,
                 delay_method: str = 'mutual_info',
                 dimension_method: str = 'cao'):
        """
        Initialize the reconstructor.
        
        Parameters:
        -----------
        max_delay : int
            Maximum delay to search for optimal τ (default 20)
        max_dimension : int
            Maximum embedding dimension to test (default 10)
        delay_method : str
            Method for delay selection: 'mutual_info', 'autocorr', 'fixed' (default: 'mutual_info')
        dimension_method : str
            Method for dimension estimation: 'cao', 'fnn_lite', 'fixed' (default: 'cao')
        """
        self.max_delay = max_delay
        self.max_dimension = max_dimension
        self.delay_method = delay_method
        self.dimension_method = dimension_method
        self.optimal_delay = None
        self.optimal_dimension = None
    
    def _compute_mutual_information(self, series: np.ndarray, delay: int, bins: int = 8) -> float:
        """
        Compute mutual information I(X_t, X_{t+τ}) using histogram-based estimation.
        
        MI quantifies the information content between the original and delayed series.
        The first minimum indicates the delay at which the series becomes sufficiently 
        decorrelated (optimal τ for embedding).
        """
        # Create delayed pairs
        n = len(series)
        if n - delay < 1:
            return 0.0
            
        x1 = series[:-delay]
        x2 = series[delay:]
        
        # Normalize to [0, 1] for binning
        x1_norm = (x1 - x1.min()) / (x1.max() - x1.min() + 1e-10)
        x2_norm = (x2 - x2.min()) / (x2.max() - x2.min() + 1e-10)
        
        # 2D histogram
        hist_2d, _, _ = np.histogram2d(x1_norm, x2_norm, bins=bins)
        pxy = hist_2d / hist_2d.sum()
        
        # Marginal distributions
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)
        
        # Compute MI: I(X,Y) = sum( P(x,y) * log(P(x,y) / (P(x)*P(y))) )
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if pxy[i, j] > 0:
                    mi += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j] + 1e-10) + 1e-10)
        
        return float(np.clip(mi, 0.0, None))
    
    def _compute_autocorrelation(self, series: np.ndarray, delay: int) -> float:
        """
        Compute autocorrelation at lag τ using normalized cross-correlation.
        
        Lower autocorrelation indicates independent information; we seek the first minimum.
        """
        n = len(series)
        if n - delay < 1:
            return 1.0
        
        x1 = series[:-delay]
        x2 = series[delay:]
        
        # Pearson correlation coefficient
        mean1, mean2 = x1.mean(), x2.mean()
        cov = np.mean((x1 - mean1) * (x2 - mean2))
        std1, std2 = x1.std(), x2.std()
        
        if std1 > 0 and std2 > 0:
            acf = cov / (std1 * std2)
        else:
            acf = 1.0
        
        return float(np.clip(acf, -1.0, 1.0))
    
    def _select_optimal_delay(self, series: np.ndarray) -> int:
        """
        Adaptively select the optimal delay τ.
        
        Strategy:
        1. For 'mutual_info': find first local minimum in MI curve
        2. For 'autocorr': find first crossing below 0.2 (or first minimum)
        3. For 'fixed': return a heuristic (sqrt(len(series)) / 10)
        """
        if self.delay_method == 'fixed':
            return max(1, int(np.sqrt(len(series)) / 10))
        
        n = len(series)
        delays = np.arange(1, min(self.max_delay + 1, n // 3))
        
        if self.delay_method == 'mutual_info':
            mis = [self._compute_mutual_information(series, d) for d in delays]
            # Find first local minimum
            if len(mis) > 2:
                for i in range(1, len(mis) - 1):
                    if mis[i] < mis[i-1] and mis[i] < mis[i+1]:
                        return int(delays[i])
            # Fallback: find first minimum
            return int(delays[np.argmin(mis)]) if len(mis) > 0 else 1
        
        elif self.delay_method == 'autocorr':
            acfs = [self._compute_autocorrelation(series, d) for d in delays]
            # Find first crossing below 0.2
            for i, acf in enumerate(acfs):
                if acf < 0.2:
                    return int(delays[i])
            # Fallback: find first minimum
            return int(delays[np.argmin(acfs)]) if len(acfs) > 0 else 1
        
        return 1
    
    def _estimate_embedding_dimension_cao(self, series: np.ndarray, delay: int) -> int:
        """
        Cao's method for embedding dimension estimation.
        
        Algorithm:
        1. For each dimension d, compute average nearest neighbor distance ratio E(d+1)/E(d)
        2. The dimension where E(d) saturates (stops decreasing significantly) is optimal
        3. More robust than FNN, computationally lighter
        """
        n = len(series)
        max_dim = min(self.max_dimension, (n - 1) // delay)
        
        if max_dim < 2:
            return 2
        
        def _compute_cao_statistic(series, d, tau):
            """Compute Cao's E(d) statistic."""
            max_lag = (d - 1) * tau
            if n <= max_lag:
                return None
            
            # Reconstruct phase space
            indices = np.arange(n - max_lag)
            X = np.column_stack([series[indices + i * tau] for i in range(d)])
            
            # Compute nearest neighbors (using Euclidean distance)
            distances = np.zeros(len(X))
            for i in range(len(X)):
                dists = np.linalg.norm(X - X[i], axis=1)
                # Find nearest neighbor (excluding self)
                dists[i] = np.inf
                distances[i] = np.min(dists)
            
            return np.mean(distances)
        
        # Compute E(d) for d = 1 to max_dim
        e_values = []
        for d in range(1, max_dim + 1):
            e_d = _compute_cao_statistic(series, d, delay)
            if e_d is not None:
                e_values.append(e_d)
            else:
                break
        
        if len(e_values) < 2:
            return 2
        
        # Find saturation point: where E(d+1)/E(d) stays close to 1.0
        ratios = []
        for i in range(len(e_values) - 1):
            if e_values[i] > 1e-10:
                ratio = e_values[i + 1] / (e_values[i] + 1e-10)
                ratios.append(ratio)
        
        # Dimension is where ratio first exceeds 0.95 (saturation)
        optimal_dim = 2
        for i, ratio in enumerate(ratios):
            if ratio > 0.95:
                optimal_dim = i + 2
                break
        
        return int(np.clip(optimal_dim, 2, max_dim))
    
    def _estimate_embedding_dimension_fnn_lite(self, series: np.ndarray, delay: int) -> int:
        """
        Simplified FNN (False Nearest Neighbors) method.
        
        Intuition: As embedding dimension increases, true neighbors stay close
        while false neighbors diverge. Optimal d is where FNN% drops below threshold.
        """
        n = len(series)
        max_dim = min(self.max_dimension, (n - 1) // delay)
        
        if max_dim < 2:
            return 2
        
        fnn_thresholds = [0.1, 0.05, 0.01]  # Relaxed for real data
        
        for d in range(2, max_dim + 1):
            max_lag = (d - 1) * delay
            if n <= max_lag:
                break
            
            # Reconstruct at dimension d and d+1
            indices = np.arange(n - max_lag)
            X_d = np.column_stack([series[indices + i * delay] for i in range(d)])
            
            max_lag_next = d * delay
            if n <= max_lag_next:
                return d
            
            indices_next = np.arange(n - max_lag_next)
            X_next = np.column_stack([series[indices_next + i * delay] for i in range(d + 1)])
            
            # Find nearest neighbors in X_d
            n_d = len(X_d)
            fnn_count = 0
            
            for i in range(min(n_d, 100)):  # Subsample for speed
                dists_d = np.linalg.norm(X_d - X_d[i], axis=1)
                dists_d[i] = np.inf
                
                nn_idx = np.argmin(dists_d)
                dist_d = dists_d[nn_idx]
                
                # Compare to distance in higher dimension
                if i < len(X_next) and nn_idx < len(X_next):
                    dist_next = np.linalg.norm(X_next[i] - X_next[nn_idx])
                    
                    # If distance increased significantly, it's a false neighbor
                    if dist_d > 1e-10:
                        ratio = dist_next / dist_d
                        if ratio > 10.0:  # Relaxed threshold
                            fnn_count += 1
            
            fnn_pct = fnn_count / min(n_d, 100)
            
            if fnn_pct < 0.1:  # Stop when FNN% < 10%
                return d
        
        return int(np.clip(max_dim, 2, max_dim))
    
    def _estimate_embedding_dimension(self, series: np.ndarray, delay: int) -> int:
        """Dispatch to selected dimension estimation method."""
        if self.dimension_method == 'cao':
            return self._estimate_embedding_dimension_cao(series, delay)
        elif self.dimension_method == 'fnn_lite':
            return self._estimate_embedding_dimension_fnn_lite(series, delay)
        else:  # 'fixed'
            return max(3, min(self.max_dimension, len(series) // (2 * delay)))
    
    def _reconstruct_trajectory(self, series: np.ndarray, delay: int, dimension: int) -> np.ndarray:
        """
        Reconstruct the phase space trajectory.
        
        Takens Embedding:
        Y = [X_t, X_{t-τ}, X_{t-2τ}, ..., X_{t-(d-1)τ}]
        
        This creates a d-dimensional point cloud whose geometry encodes
        the dynamics of the original system.
        """
        n = len(series)
        max_lag = (dimension - 1) * delay
        
        if n <= max_lag:
            raise ValueError(f"Series too short: {n} <= {max_lag}")
        
        indices = np.arange(n - max_lag)
        trajectory = np.column_stack([
            series[indices + i * delay] for i in range(dimension)
        ])
        
        return trajectory
    
    def _compute_geometric_invariants(self, trajectory: np.ndarray) -> Dict[str, float]:
        """
        Compute geometric invariants that capture emergent structure.
        
        Returns:
        --------
        system_radius : float
            Maximum distance from trajectory centroid (measures compactness)
        
        compactness : float
            Normalized concentration metric: radius / (mean inter-point distance)
            Lower values = more compact, higher values = more dispersed
        
        recurrence_rate : float
            Fraction of near-return pairs (repeat visits to same phase regions)
            Higher = more structured, lower = more chaotic
        
        divergence_proxy : float
            Approximate divergence rate (Lyapunov-like measure)
            Measures information spreading along the trajectory
        """
        n = len(trajectory)
        
        # Compute system radius (compactness)
        centroid = np.mean(trajectory, axis=0)
        distances_from_center = np.linalg.norm(trajectory - centroid, axis=1)
        system_radius = float(np.max(distances_from_center))
        
        # Compute mean inter-point distance
        if n > 1:
            # Pairwise distances (sample for large trajectories)
            sample_size = min(n, 50)
            sample_indices = np.random.choice(n, sample_size, replace=False)
            sample = trajectory[sample_indices]
            
            distances = np.zeros((len(sample), len(sample)))
            for i in range(len(sample)):
                distances[i] = np.linalg.norm(sample - sample[i], axis=1)
            
            mean_distance = np.mean(distances[distances > 1e-10])
        else:
            mean_distance = 1.0
        
        # Compactness: normalized concentration
        compactness = float(system_radius / (mean_distance + 1e-10))
        
        # Recurrence rate: fraction of near-return pairs
        recurrence_threshold = np.percentile(distances_from_center[distances_from_center > 0], 25)
        near_returns = 0
        for i in range(n - 1):
            for j in range(i + 1, min(i + 20, n)):  # Look ahead window
                dist = np.linalg.norm(trajectory[i] - trajectory[j])
                if dist < recurrence_threshold:
                    near_returns += 1
        
        max_possible_returns = (n - 1) * 19 / 2
        recurrence_rate = float(near_returns / (max_possible_returns + 1e-10))
        
        # Divergence proxy: spread of consecutive derivatives
        diffs = np.diff(trajectory, axis=0)
        velocities = np.linalg.norm(diffs, axis=1)
        
        if len(velocities) > 1:
            mean_velocity = np.mean(velocities)
            velocity_std = np.std(velocities)
            divergence_proxy = float(velocity_std / (mean_velocity + 1e-10))
        else:
            divergence_proxy = 0.0
        
        return {
            'system_radius': float(np.clip(system_radius, 0.0, None)),
            'compactness': float(np.clip(compactness, 0.0, 100.0)),
            'recurrence_rate': float(np.clip(recurrence_rate, 0.0, 1.0)),
            'divergence_proxy': float(np.clip(divergence_proxy, 0.0, 100.0)),
            'mean_distance': float(np.clip(mean_distance, 0.0, None)),
        }
    
    def reconstruct(self, series: np.ndarray, 
                   use_adaptive_delay: bool = True,
                   use_adaptive_dimension: bool = True) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Main reconstruction pipeline.
        
        Parameters:
        -----------
        series : np.ndarray
            Input 1D time series (e.g., price returns, signal)
        use_adaptive_delay : bool
            If True, adaptively select τ; else use fixed heuristic
        use_adaptive_dimension : bool
            If True, adaptively estimate d; else use fixed value
        
        Returns:
        --------
        trajectory : np.ndarray
            Reconstructed phase space of shape (N, dimension)
        invariants : Dict
            Geometric invariants: radius, compactness, recurrence, divergence
        """
        series = np.asarray(series, dtype=float)
        
        if len(series) < 10:
            raise ValueError("Series must have at least 10 points")
        
        # Select optimal delay
        if use_adaptive_delay:
            delay = self._select_optimal_delay(series)
        else:
            delay = max(1, int(np.sqrt(len(series)) / 10))
        
        self.optimal_delay = delay
        
        # Select optimal dimension
        if use_adaptive_dimension:
            dimension = self._estimate_embedding_dimension(series, delay)
        else:
            dimension = max(3, min(self.max_dimension, len(series) // (2 * delay)))
        
        self.optimal_dimension = dimension
        
        # Reconstruct trajectory
        trajectory = self._reconstruct_trajectory(series, delay, dimension)
        
        # Compute geometric invariants
        invariants = self._compute_geometric_invariants(trajectory)
        
        # Add metadata
        invariants['delay'] = float(delay)
        invariants['dimension'] = float(dimension)
        
        return trajectory, invariants
    
    def get_flattened_features(self, trajectory: np.ndarray, max_features: int = 20) -> np.ndarray:
        """
        Flatten reconstructed trajectory for quantum circuit encoding.
        
        Strategy:
        1. Use PCA-like approach (principal components of covariance matrix)
        2. Or use row means/STDs for each dimension
        3. Keep bounded to max_features for circuit compatibility
        """
        if trajectory.shape[0] == 0:
            return np.zeros(max_features)
        
        # Compute column-wise statistics
        means = np.mean(trajectory, axis=0)
        stds = np.std(trajectory, axis=0)
        mins = np.min(trajectory, axis=0)
        maxs = np.max(trajectory, axis=0)
        
        # Flatten: means, stds, ranges
        features = []
        features.extend(means)
        features.extend(stds)
        features.extend(maxs - mins)
        
        features = np.array(features)
        
        # Pad or truncate to max_features
        if len(features) < max_features:
            features = np.pad(features, (0, max_features - len(features)), mode='edge')
        else:
            features = features[:max_features]
        
        return features


# ========== Example Usage ==========

if __name__ == "__main__":
    # Generate synthetic time series (e.g., chaotic Lorenz projection)
    np.random.seed(42)
    t = np.linspace(0, 100, 500)
    signal = np.sin(0.1 * t) * np.cos(0.05 * t) + 0.1 * np.random.randn(len(t))
    
    # Initialize reconstructor
    reconstructor = PhaseSpaceReconstructor(
        max_delay=30,
        max_dimension=8,
        delay_method='mutual_info',
        dimension_method='cao'
    )
    
    # Reconstruct
    trajectory, invariants = reconstructor.reconstruct(signal, 
                                                       use_adaptive_delay=True,
                                                       use_adaptive_dimension=True)
    
    print("=== Takens Phase Space Reconstruction ===")
    print(f"Input series length: {len(signal)}")
    print(f"Optimal delay (τ): {invariants['delay']}")
    print(f"Optimal dimension (d): {invariants['dimension']}")
    print(f"Trajectory shape: {trajectory.shape}")
    print("\n=== Geometric Invariants ===")
    for key, value in sorted(invariants.items()):
        print(f"{key:20s}: {value:.6f}")
    
    # Get flattened features for quantum circuit
    features = reconstructor.get_flattened_features(trajectory, max_features=16)
    print(f"\nFlattened features for QC: {features[:8]} ... (len={len(features)})")
