#!/usr/bin/env python3
"""
IBIS Engine 1: Signal Clarity with RMT Cleaning & Takens Integration
======================================================================

Quantifies the signal-to-noise ratio (SNR) and filters out market entropy.
Enhanced with Random Matrix Theory (RMT) cleaning to remove noise eigenvalues
and optional Takens phase space reconstruction for structural analysis.

Background:
- RMT principle: In pure noise, eigenvalue distribution follows Marchenko-Pastur law
- Signal eigenvalues stick out above the MP threshold
- We shrink noise eigenvalues to zero, preserving signal structure
"""
from pennylane import numpy as np
from typing import Dict, Tuple, Optional


def calculate_signal_clarity(forecast_trajectory: np.ndarray) -> float:
    """
    Computes a signal clarity metric between 0.0 (pure noise) and 1.0 (perfect signal).
    Uses the ratio of trajectory variance to mean absolute deviation.
    
    Parameters:
    -----------
    forecast_trajectory : np.ndarray
        Input signal (1D or 2D)
    
    Returns:
    --------
    float
        Clarity score in [0, 1]
    """
    if forecast_trajectory.size == 0:
        return 0.0
        
    mean_signal = np.mean(forecast_trajectory)
    variance = np.var(forecast_trajectory)
    
    if variance == 0:
        return 1.0
        
    # Quantify structural noise floor
    noise_floor = np.mean(np.abs(forecast_trajectory - mean_signal))
    
    # Calculate clarity score bounded between 0 and 1
    clarity_score = 1.0 / (1.0 + (noise_floor / (variance + 1e-6)))
    return float(clarity_score)


def _marchenko_pastur_threshold(eigenvalues: np.ndarray, q: float) -> float:
    """
    Compute the Marchenko-Pastur threshold for eigenvalue filtering.
    
    Parameters:
    -----------
    eigenvalues : np.ndarray
        Eigenvalues of covariance matrix (sorted descending)
    q : float
        Ratio of observations to features (T / N)
    
    Returns:
    --------
    float
        Threshold: eigenvalues above this are signal, below is noise
    """
    if q <= 0 or q >= 1:
        return 0.0
    
    # Marchenko-Pastur distribution parameters
    # Upper edge: lambda_+ = (1 + sqrt(q))^2 / q
    # Lower edge: lambda_- = (1 - sqrt(q))^2 / q
    
    lambda_plus = ((1.0 + np.sqrt(q))**2) / q
    
    # More robust: use 95th percentile of noise eigenvalues
    # Typically, use 1.2 * lambda_plus for real data
    return float(1.2 * lambda_plus)


def clean_covariance_rmt(series: np.ndarray, 
                        q_ratio: Optional[float] = None,
                        return_phase_space: bool = False) -> Dict[str, any]:
    """
    Clean covariance matrix using Random Matrix Theory.
    
    Algorithm:
    1. Compute covariance matrix from series
    2. Eigenvalue decomposition
    3. Threshold eigenvalues using Marchenko-Pastur law
    4. Reconstruct cleaned covariance
    5. (Optional) Apply Takens reconstruction to cleaned series
    
    Parameters:
    -----------
    series : np.ndarray
        Input time series (1D or 2D, shape (T,) or (T, N))
    q_ratio : float, optional
        Observation-to-feature ratio T/N. If None, estimated from data.
    return_phase_space : bool
        If True, also return Takens reconstructed phase space
    
    Returns:
    --------
    dict with keys:
    - 'cleaned_cov': Cleaned covariance matrix
    - 'cleaned_series': Cleaned time series (flattened)
    - 'eigenvalues': All eigenvalues
    - 'filtered_eigenvalues': Eigenvalues after thresholding
    - 'phase_space': (optional) Reconstructed trajectory
    - 'phase_invariants': (optional) Geometric invariants
    """
    series = np.asarray(series, dtype=float)
    
    # Handle 1D vs 2D
    if series.ndim == 1:
        series_2d = series.reshape(-1, 1)
    else:
        series_2d = series
    
    T, N = series_2d.shape
    
    # Estimate q_ratio if not provided
    if q_ratio is None:
        q_ratio = float(T) / float(N)
    else:
        q_ratio = float(q_ratio)
    
    # Compute covariance
    cov_matrix = np.cov(series_2d, rowvar=False)
    
    if cov_matrix.ndim == 0:  # Single variable case
        cov_matrix = np.array([[float(cov_matrix)]])
    
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
    
    # Compute MP threshold
    mp_threshold = _marchenko_pastur_threshold(eigenvalues, q_ratio)
    
    # Filter eigenvalues: shrink noise eigenvalues to near-zero
    filtered_eigenvalues = np.copy(eigenvalues)
    noise_indices = eigenvalues < mp_threshold
    filtered_eigenvalues[noise_indices] = 0.0
    
    # Reconstruct cleaned covariance (only if we have signal eigenvalues)
    if np.sum(filtered_eigenvalues > 0) > 0:
        # Reorder eigenvectors to match sorted eigenvalues
        idx_sort = np.argsort(eigenvalues)[::-1]
        eigenvectors_sorted = eigenvectors[:, idx_sort]
        
        cleaned_cov = eigenvectors_sorted @ np.diag(filtered_eigenvalues) @ eigenvectors_sorted.T
    else:
        cleaned_cov = np.eye(N) * 1e-6
    
    # Apply cleaned covariance to denoise series (Wiener-like filtering)
    if np.linalg.matrix_rank(cleaned_cov) > 0:
        try:
            cov_inv = np.linalg.pinv(cleaned_cov)
            cov_clean = np.linalg.inv(np.eye(N) + cov_inv @ np.linalg.inv(cov_matrix + 1e-6))
            cleaned_series_2d = series_2d @ cov_clean.T
        except:
            cleaned_series_2d = series_2d
    else:
        cleaned_series_2d = series_2d
    
    result = {
        'cleaned_cov': cleaned_cov,
        'cleaned_series': cleaned_series_2d.ravel() if series.ndim == 1 else cleaned_series_2d,
        'eigenvalues': eigenvalues,
        'filtered_eigenvalues': filtered_eigenvalues,
        'mp_threshold': float(mp_threshold),
        'q_ratio': float(q_ratio),
    }
    
    # Optionally, return phase space reconstruction
    if return_phase_space:
        try:
            # Import here to avoid circular dependency
            from ..src.phase_space_reconstructor import PhaseSpaceReconstructor
            
            reconstructor = PhaseSpaceReconstructor(
                max_delay=20,
                max_dimension=8,
                delay_method='mutual_info',
                dimension_method='cao'
            )
            
            cleaned_1d = cleaned_series_2d.ravel() if cleaned_series_2d.ndim > 1 else cleaned_series_2d
            
            if len(cleaned_1d) > 10:
                trajectory, invariants = reconstructor.reconstruct(cleaned_1d)
                result['phase_space'] = trajectory
                result['phase_invariants'] = invariants
        except Exception as e:
            # If reconstruction fails, just skip it
            pass
    
    return result


# Example usage
if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    t = np.linspace(0, 10, 200)
    signal = np.sin(2 * np.pi * 0.5 * t)
    noise = 0.3 * np.random.randn(len(t))
    noisy_signal = signal + noise
    
    result = clean_covariance_rmt(noisy_signal, return_phase_space=False)
    
    print("=== Signal Clarity with RMT Cleaning ===")
    print(f"Original clarity: {calculate_signal_clarity(noisy_signal):.4f}")
    print(f"Cleaned clarity: {calculate_signal_clarity(result['cleaned_series']):.4f}")
    print(f"MP threshold: {result['mp_threshold']:.6f}")
    print(f"Eigenvalues (top 5): {result['eigenvalues'][:5]}")
    print(f"Filtered eigenvalues (top 5): {result['filtered_eigenvalues'][:5]}")