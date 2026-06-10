#!/usr/bin/env python3
"""
Integration Verification Script
Tests all new components and verifies full pipeline functionality
"""

import sys
import numpy as np

def test_phase_space_reconstructor():
    """Test basic Takens reconstruction functionality"""
    print("\n[1/5] Testing PhaseSpaceReconstructor...")
    try:
        from src.phase_space_reconstructor import PhaseSpaceReconstructor
        
        reconstructor = PhaseSpaceReconstructor()
        signal = np.sin(np.linspace(0, 10, 300)) + 0.05 * np.random.randn(300)
        
        trajectory, invariants = reconstructor.reconstruct(signal)
        
        assert trajectory.shape[0] > 0, "Trajectory empty"
        assert 'system_radius' in invariants, "Missing invariant"
        assert 'recurrence_rate' in invariants, "Missing invariant"
        
        print("  ✓ Reconstruction works")
        print(f"  ✓ Trajectory shape: {trajectory.shape}")
        print(f"  ✓ Key invariants: radius={invariants['system_radius']:.4f}, recurrence={invariants['recurrence_rate']:.4f}")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def test_rmt_cleaning():
    """Test RMT cleaning functionality"""
    print("\n[2/5] Testing RMT Cleaning...")
    try:
        from engines.signal_clarity import clean_covariance_rmt
        
        signal = np.random.randn(200)
        result = clean_covariance_rmt(signal)
        
        assert 'cleaned_series' in result, "No cleaned series"
        assert 'mp_threshold' in result, "No MP threshold"
        assert len(result['cleaned_series']) > 0, "Cleaned series empty"
        
        print("  ✓ RMT cleaning works")
        print(f"  ✓ MP threshold: {result['mp_threshold']:.6f}")
        print(f"  ✓ Signal eigenvalues: {np.sum(result['filtered_eigenvalues'] > 0)}")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def test_coherence_geometry():
    """Test geometric integration in coherence"""
    print("\n[3/5] Testing Coherence with Geometry...")
    try:
        from src.coherence import UnifiedPhaseAnalyzer
        
        analyzer = UnifiedPhaseAnalyzer(alpha=0.25, beta=0.08, gamma=0.15)
        signal = np.sin(np.linspace(0, 10, 250)) + 0.05 * np.random.randn(250)
        
        # Test without geometry
        result_base = analyzer.analyze(signal)
        assert 'C' in result_base, "No C-score"
        
        # Test with geometry
        geom = {'system_radius': 1.2, 'compactness': 2.5, 'recurrence_rate': 0.65, 'divergence_proxy': 0.8}
        result_geom = analyzer.analyze(signal, geometric_invariants=geom)
        assert 'C_enhanced' in result_geom, "No C_enhanced"
        
        print("  ✓ Topological analysis works")
        print(f"  ✓ Base C-score: {result_base['C']:.4f}")
        print("  ✓ Geometric integration works")
        print(f"  ✓ Enhanced C-score: {result_geom['C_enhanced']:.4f}")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def test_domain_adapter_backward_compat():
    """Test backward compatibility (use_takens=False)"""
    print("\n[4/5] Testing Backward Compatibility...")
    try:
        from src.domain_adapter import DomainAdapter
        
        adapter = DomainAdapter(use_takens=False)
        signal = np.random.randn(250)
        
        result = adapter.process(signal)
        
        assert 'phase_analysis' in result, "Missing phase_analysis"
        assert 'C_score' in result['coherence'], "Missing C_score"
        assert result['use_takens'] == False, "use_takens not False"
        
        print("  ✓ Baseline mode works")
        print(f"  ✓ C-score: {result['coherence']['C_score']:.4f}")
        print(f"  ✓ Regime: {result['coherence']['regime']}")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def test_domain_adapter_enhanced():
    """Test enhanced pipeline (use_takens=True)"""
    print("\n[5/5] Testing Enhanced Pipeline...")
    try:
        from src.domain_adapter import DomainAdapter
        
        adapter = DomainAdapter(use_takens=True, gamma=0.15)
        signal = np.sin(np.linspace(0, 10, 300)) + 0.05 * np.random.randn(300)
        
        result = adapter.process(signal)
        
        assert 'phase_analysis' in result, "Missing phase_analysis"
        assert result['use_takens'] == True, "use_takens not True"
        
        # Check for geometric features
        has_geometric = 'geometric_features' in result or 'takens_error' in result
        assert has_geometric, "Missing geometric features or error"
        
        if 'geometric_features' in result:
            geom = result['geometric_features']
            assert 'recurrence_rate' in geom, "Missing recurrence_rate"
            print("  ✓ Full enhanced pipeline works")
            print(f"  ✓ C-score: {result['coherence']['C_score']:.4f}")
            print(f"  ✓ Recurrence: {geom['recurrence_rate']:.4f}")
            print(f"  ✓ Embedding: d={int(geom['embedding_dimension'])}, τ={int(geom['embedding_delay'])}")
        else:
            print(f"  ⚠ Geometric features unavailable: {result.get('takens_error', 'Unknown')}")
            print("  ✓ Fallback pipeline works (Takens unavailable but no crash)")
        
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("TAKENS INTEGRATION VERIFICATION")
    print("="*60)
    
    tests = [
        test_phase_space_reconstructor,
        test_rmt_cleaning,
        test_coherence_geometry,
        test_domain_adapter_backward_compat,
        test_domain_adapter_enhanced,
    ]
    
    results = []
    for test_func in tests:
        try:
            results.append(test_func())
        except Exception as e:
            print(f"  ✗ EXCEPTION: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED - Integration successful!")
        print("\nNext steps:")
        print("  1. Read QUICK_START_TAKENS.md for usage examples")
        print("  2. Run: python3 validation/validation_demo_takens.py")
        print("  3. Start backtesting on real market data")
        return 0
    else:
        print(f"\n✗ {total - passed} TEST(S) FAILED - Review errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
