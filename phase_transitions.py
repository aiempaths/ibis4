#!/usr/bin/env python3
"""
IBIS Engine 5: Phase Transition Dynamics
Monitors thermodynamic volatility shifts to flag imminent market tipping points.
"""
from pennylane import numpy as np

def detect_phase_criticality(forecast_trajectory: np.ndarray) -> dict:
    """
    Analyzes systemic variance profiles to detect critical tipping points.
    Returns high-risk warnings if metrics near phase transition boundaries.
    """
    # Calculate step variations across time windows
    step_variations = np.diff(forecast_trajectory, axis=0) if forecast_trajectory.ndim > 1 else np.diff(forecast_trajectory)
    
    susceptibility_metric = float(np.var(step_variations)) if step_variations.size > 0 else 0.0
    
    # Define a criticality index scaling parameter
    criticality_index = float(np.tanh(susceptibility_metric * 100.0))
    
    # Issue systemic alert if system reaches within 8% of the extreme critical boundary
    approaching_criticality = bool(criticality_index > 0.92)
    
    return {
        "criticality_index": criticality_index,
        "approaching_criticality": approaching_criticality,
        "system_regime": "CRITICAL_BURST" if approaching_criticality else "STABLE_PERCOLATION"
    }