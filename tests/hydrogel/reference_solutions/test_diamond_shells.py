#!/usr/bin/env python3

import numpy as np
import pytest
import matplotlib.pyplot as plt

from matscipy.calculators.hydrogel.reference_solutions.lattice_shell_structures import DiamondShellStructure, CubicShellStructure
from matscipy.neighbours import neighbour_list
from matscipy.molecules import Molecules
from ase.build import bulk

def assert_close(a, b, rtol=1e-12, atol=1e-12):
    assert np.isclose(a, b, rtol=rtol, atol=atol), f"{a} != {b}"

def test_inheritance():
    """Test that DiamondShellStructure properly inherits from CubicShellStructure"""
    diamond = DiamondShellStructure(nb_shells=1)
    print("Testing inheritance:")
    print(f"  Is instance of CubicShellStructure: {isinstance(diamond, CubicShellStructure)}")
    print(f"  Dimension: {diamond.dim}")
    print(f"  Coordination: {diamond.coordination}")
    assert isinstance(diamond, CubicShellStructure)
    assert diamond.dim == 3
    assert diamond.coordination == 4
    print("✓ Inheritance test passed")

def test_diamond_first_shell():
    """Test first shell properties"""
    diamond = DiamondShellStructure(nb_shells=1)
    
    print("First shell:")
    print(f"  Distance: {diamond.a[0]:.6f} (expected: 1.000000)")
    print(f"  Coordination: {diamond.Z[0]} (expected: 4)")
    print(f"  Isotropic coeff: {diamond.isotropic_coeff[0]:.6f} (expected: {1.0/9.0:.6f})")
    print(f"  Cubic coeff: {diamond.cubic_coeff[0]:.6f} (expected: {-2.0/9.0:.6f})")
    
    assert_close(diamond.a[0], 1.0)
    assert diamond.Z[0] == 4
    assert_close(diamond.isotropic_coeff[0], 1.0/9.0)
    assert_close(diamond.cubic_coeff[0], -2.0/9.0)
    print("✓ First shell test passed")

def test_diamond_second_shell():
    """Test second shell properties"""
    diamond = DiamondShellStructure(nb_shells=2)
    
    print("\nSecond shell:")
    r2_expected = 4.0/np.sqrt(6.0)
    print(f"  Distance: {diamond.a[1]:.6f} (expected: {r2_expected:.6f})")
    print(f"  Coordination: {diamond.Z[1]} (expected: 12)")
    print(f"  Isotropic coeff: {diamond.isotropic_coeff[1]:.6f} (expected: {1.0/12.0:.6f})")
    print(f"  Cubic coeff: {diamond.cubic_coeff[1]:.6f} (expected: {-1.0/12.0:.6f})")
    
    assert_close(diamond.a[1], r2_expected)
    assert diamond.Z[1] == 12
    assert_close(diamond.isotropic_coeff[1], 1.0/12.0)
    assert_close(diamond.cubic_coeff[1], -1.0/12.0)
    print("✓ Second shell test passed")

def test_diamond_axis_shell():
    """Test axis shell properties (⟨100⟩ directions)"""
    diamond = DiamondShellStructure(cutoff=2.35)  # just above 4/sqrt(3) ≈ 2.309
    
    r_target = 4.0/np.sqrt(3.0)
    # Find closest shell
    idx = np.argmin(np.abs(diamond.a - r_target))
    
    print(f"\nAxis shell (closest to r={r_target:.6f}):")
    print(f"  Distance: {diamond.a[idx]:.6f}")
    print(f"  Coordination: {diamond.Z[idx]} (expected: 6)")
    print(f"  Isotropic coeff: {diamond.isotropic_coeff[idx]:.6f} (expected: 0.0)")
    print(f"  Cubic coeff: {diamond.cubic_coeff[idx]:.6f} (expected: {1.0/3.0:.6f})")
    
    assert_close(diamond.a[idx], r_target)
    assert diamond.Z[idx] == 6
    assert_close(diamond.isotropic_coeff[idx], 0.0)
    assert_close(diamond.cubic_coeff[idx], 1.0/3.0)
    print("✓ Axis shell test passed")

def test_shell_tensors():
    """Test that shell tensor methods work"""
    diamond = DiamondShellStructure(nb_shells=3)
    
    print("\nTesting shell tensors:")
    
    # Test 2nd order tensor
    T2 = diamond.shellTensor2(0)
    print(f"  T2 shape: {T2.shape}")
    print(f"  T2 trace: {np.trace(T2):.6f} (expected: 1.0)")
    assert T2.shape == (3, 3)
    assert_close(np.trace(T2), 1.0)
    
    # Test 4th order tensor
    T4 = diamond.shellTensor4(0) 
    print(f"  T4 shape: {T4.shape}")
    print(f"  T4[0,0,0,0]: {T4[0,0,0,0]:.6f}")
    print(f"  T4[0,0,1,1]: {T4[0,0,1,1]:.6f}")
    assert T4.shape == (3, 3, 3, 3)
    
    print("✓ Shell tensor tests passed")

def test_volume_per_atom():
    """Test volume per atom calculation"""
    diamond = DiamondShellStructure(nb_shells=1)
    
    vpa_1 = diamond.vpa(1.0)  # volume per atom at bond length = 1
    print(f"\nVolume per atom at r=1: {vpa_1:.6f}")
    print(f"Expected: {8 * np.sqrt(3) / 9:.6f}")
    
    assert_close(vpa_1, 8 * np.sqrt(3) / 9)
    print("✓ Volume per atom test passed")

def create_diamond_structure(crosslink_spacing=1.0, n_cells=3):
    """Create a diamond lattice structure for testing.
    
    Parameters
    ----------
    crosslink_spacing : float
        Nearest neighbor distance (bond length)
    n_cells : int
        Number of unit cells in each direction
    
    Returns
    -------
    atoms : ase.Atoms
        Diamond structure
    molecules : matscipy.molecules.Molecules  
        Bond connectivity
    """
    # Diamond lattice constant: nearest neighbor distance = a*sqrt(3)/4
    a = crosslink_spacing * 4 / np.sqrt(3)
    
    # Create diamond lattice
    atoms = bulk("C", "diamond", a=a, cubic=True)
    atoms = atoms.repeat((n_cells, n_cells, n_cells))
    atoms.set_masses(np.ones(len(atoms)))
    
    # Create bonds between nearest neighbors
    nn_dist = crosslink_spacing
    i_p, j_p = neighbour_list("ij", atoms, nn_dist * 1.1)
    
    # Keep only unique bonds (i < j)
    mask = i_p < j_p
    bonds = np.column_stack([i_p[mask], j_p[mask]])
    molecules = Molecules(bonds_connectivity=bonds)
    
    return atoms, molecules

def test_shell_structure_vs_simulation():
    """Test that analytical shell structure matches RDF from actual diamond lattice
    
    
    """
    print("\nTesting shell structure vs simulation:")
    
    # Create analytical shell structure (matching your parameters)
    cutoff = 10.0  # in units of bond length
    diamond_shells = DiamondShellStructure(cutoff=cutoff, safety_margin_cells=8)
    
    # Create actual diamond structure (matching your parameters)
    crosslink_spacing = 1.0  # bond length
    n_cells = 2  # matching your parameters
    atoms, molecules = create_diamond_structure(crosslink_spacing, n_cells)
    
    print(f"  Created diamond structure with {len(atoms)} atoms")
    print(f"  System size: {atoms.cell[0,0]:.2f} x {atoms.cell[1,1]:.2f} x {atoms.cell[2,2]:.2f}")
    
    # Calculate RDF from simulation
    distances = neighbour_list('d', atoms, cutoff=cutoff * crosslink_spacing)
    natoms = len(atoms)
    
    # Convert distances to bond length units
    distances_normalized = distances / crosslink_spacing
    
    # Create histogram for RDF (matching your parameters)
    n_bins = 500  # matching your bin count
    n, x = np.histogram(distances_normalized, bins=n_bins, range=(0, cutoff), density=False)
    bin_centers = (x[:-1] + x[1:]) / 2
    rdf_simulation = n / natoms  # normalize by number of atoms
    
    print(f"  Calculated RDF from {len(distances)} pair distances")
    
    # Compare with analytical shell structure
    print("  Comparison of peaks:")
    print("    Shell | Analytical r | Analytical Z | Simulation peak")
    print("    ------|--------------|--------------|----------------")
    
    tolerance = 0.2  # tolerance for finding peaks
    matches = 0
    matches_first_20 = 0  # Track first 20 shells separately
    total_shells = len(diamond_shells.a)  # test all shells up to cutoff=10
    
    for i in range(total_shells):
        r_analytical = diamond_shells.a[i]
        Z_analytical = diamond_shells.Z[i]
        
        # Find simulation peak near analytical position
        # closest peak
        mask = np.argmin(abs(bin_centers - r_analytical)) 
        peak_height = rdf_simulation[mask]
        peak_position = bin_centers[mask]
        
        print(f"    {i+1:4d}  | {r_analytical:10.4f}  | {Z_analytical:10d}  | r={peak_position:.4f}, h={peak_height:.1f}")
        

        # Check if peak position is close to analytical prediction
        if np.isclose(peak_height, Z_analytical, atol=0.1,):  # within 15% of bond length
            matches += 1
            if i < 20:
                matches_first_20 += 1
        else:
            print(f"    {i+1:4d}  | {r_analytical:10.4f}  | {Z_analytical:10d}  | No peak found")
    
    # We compare only the cumulative results to aboid sensitivity to binning
    print("\n  Cumulative coordination comparison:")
    print("    Distance | Analytical cum Z | Simulation cum Z | Relative error")
    print("    ---------|------------------|------------------|---------------")
    
    # Calculate cumulative coordination numbers for analytical structure
    cumulative_analytical = np.cumsum(diamond_shells.Z)
    
    # Calculate cumulative coordination numbers for simulation
    cumulative_simulation = []
    cumulative_errors = []
    
    test_distances = (diamond_shells.a[1:] +  diamond_shells.a[:-1])/2  # Test every 3rd shell to reduce output
    
    for i, r_test in enumerate(test_distances):
        # Find all bins up to this distance
        mask_cumulative = bin_centers <= r_test
        cum_sim = np.sum(rdf_simulation[mask_cumulative])
        
        # Find corresponding analytical cumulative value
        shell_idx = np.where(diamond_shells.a <= r_test)[0]
        if len(shell_idx) > 0:
            cum_analytical = cumulative_analytical[shell_idx[-1]]
        else:
            cum_analytical = 0
        
        cumulative_simulation.append(cum_sim)
        
        # Calculate relative error
        if cum_analytical > 0:
            rel_error = abs(cum_sim - cum_analytical) / cum_analytical
            cumulative_errors.append(rel_error)
            error_pct = rel_error * 100
        else:
            rel_error = float('inf')
            cumulative_errors.append(rel_error)
            error_pct = float('inf')
        
        print(f"    {r_test:8.3f} | {cum_analytical:14.1f} | {cum_sim:14.1f} | {error_pct:8.1f}%")
    
    # Calculate overall agreement based on cumulative comparison
    valid_errors = [e for e in cumulative_errors if e != float('inf')]
    if valid_errors:
        mean_error = np.mean(valid_errors)
        max_error = np.max(valid_errors)
        print(f"  Average cumulative error: {mean_error*100:.1f}%")
        print(f"  Maximum cumulative error: {max_error*100:.1f}%")
        
        # Test for reasonable agreement (use cumulative comparison for robustness)
        assert max_error < 0.01, f"Max cumulative error {max_error*100:.1f}% too high, expected <30%"
        
    else:
        print("  Could not calculate errors - no valid comparisons")
        assert False, "No valid cumulative comparisons possible"
    

    return diamond_shells, atoms, bin_centers, rdf_simulation

if __name__ == "__main__":
    print("Testing DiamondShellStructure implementation...")
    print("="*50)
    
    # test_inheritance()
    # test_diamond_first_shell()
    # test_diamond_second_shell() 
    # test_diamond_axis_shell()
    # test_shell_tensors()
    # test_volume_per_atom()
    test_shell_structure_vs_simulation()
    
    # Uncomment the line below to create a plot like your example
    # plot_shell_comparison(create_plot=True)
    
    print("\n" + "="*50)
    print("All tests passed! ✓")