

import numpy as np
from matscipy.calculators.hydrogel.embedding_constant_coordination import FloryHugginsPotential



# ============== Flory-Huggins Tests ==============


class TestFloryHugginsPotential :
    """Tests for Flory-Huggins embedding energy."""

    def test_energy_at_low_density(self):
        """Test that energy is finite at low density."""
        fh = FloryHugginsPotential (
            chain_monomers=50,
            monomer_volume=4 * np.pi / 3,
            flory_chi=0.5,
            coordination=4,
        )
        w0 = 105.0 / (16.0 * np.pi * 20**3)


        phi = 0.01 
        rho = phi / (fh.vchain * fh.coord / 2)
        E = fh(rho)

        assert np.isfinite(E).all()

    def test_derivative_numerical(self):
        """Test derivative against numerical differentiation."""
        fh = FloryHugginsPotential (
            chain_monomers=50,
            monomer_volume=4 * np.pi / 3,
            flory_chi=0.5,
            coordination=4,
        )
        rho = np.linspace(0.01, 0.9, 10) * fh.max_crosslink_density
        h = 1e-8

        # Numerical derivative
        dF_num = (fh(rho + h) - fh(rho - h)) / (2 * h)

        # Analytical derivative
        dF_ana = fh.derivative(rho)

        np.testing.assert_allclose(dF_ana, dF_num, rtol=1e-4)

    def test_second_derivative_numerical(self):
        """Test second derivative against numerical differentiation."""
        fh = FloryHugginsPotential (
            chain_monomers=50,
            monomer_volume=4 * np.pi / 3,
            flory_chi=0.5,
            coordination=4,
        )
        rho = np.linspace(0.01, 0.9, 10) *  fh.max_crosslink_density
        h = 1e-7

        # Numerical second derivative
        d2F_num = (fh.derivative(rho + h) - fh.derivative(rho - h)) / (2 * h)

        # Analytical second derivative
        d2F_ana = fh.second_derivative(rho)

        np.testing.assert_allclose(d2F_ana, d2F_num, rtol=1e-4)

    def test_energy_consistency(self):
        """Test that energy, derivative, and second derivative are consistent."""
        fh = FloryHugginsPotential (
            chain_monomers=50,
            monomer_volume=4 * np.pi / 3,
            flory_chi=0.5,
            coordination=4,
        )

        rho = np.linspace(0.01, 0.9, 10) * fh.max_crosslink_density

        # energy per crosslink
        E = fh(rho)

        # energy per volume
        E_vol = E * rho

        # Implementation of energy per volume
        E_vol_impl = fh.per_volume(rho)

        assert np.allclose(E_vol, E_vol_impl, rtol=1e-8)


    def test_pressure_comp_with_fd(self):
        """Test that pressure calculation using composition rule is consistent with finite differences of the energy"""
        fh = FloryHugginsPotential (
            chain_monomers=50,
            monomer_volume=4 * np.pi / 3,
            flory_chi=0.5,
            coordination=4,
        )

        rho = np.linspace(0.01, 0.9, 10) * fh.max_crosslink_density

        # numerical derivative of energy per volume
        h = 1e-8
        
        # Pi = - df* / dJ 
        # and f* = J per_volume(rho0 / J)
        fstar = lambda J: J * fh.per_volume(rho / J)

        dE_dV_num = - (fstar(1 + h) - fstar(1 - h)) / (2 * h)

        P_num = dE_dV_num

        Pcomp = - fh.per_volume(rho) + fh.polymer_volume_fraction_from_crosslink_density(rho) * fh.per_volume(rho, der='phi')

        np.testing.assert_allclose(Pcomp, P_num, rtol=1e-4)

    def test_pressure_ana_with_fd(self):
        """Test that direct analytical expression for pressure calculation is consistent with finite differences of the energy """
        fh = FloryHugginsPotential (
            chain_monomers=50,
            monomer_volume=4 * np.pi / 3,
            flory_chi=0.5,
            coordination=4,
        )

        rho = np.linspace(0.01, 0.9, 10) * fh.max_crosslink_density

        # direct analytical expression for the pressure.
        P_calc = fh.pressure(rho)

        # numerical derivative of energy per volume
        h = 1e-8
        
        # Pi = - df* / dJ 
        # and f* = J per_volume(rho0 / J)
        fstar = lambda J: J * fh.per_volume(rho / J)

        dE_dV_num = - (fstar(1 + h) - fstar(1 - h)) / (2 * h)

        P_num = dE_dV_num

        np.testing.assert_allclose(P_calc, P_num, rtol=1e-4)

    def test_per_volume_phi_derivative_numerical(self):
        """Test per_volume phi derivatives against numerical differentiation."""
        fh = FloryHugginsPotential (
            chain_monomers=50,
            monomer_volume=4 * np.pi / 3,
            flory_chi=0.5,
            coordination=4,
        )
        
        # Test multiple density ranges
        rho_values = [
            np.linspace(0.01, 0.2, 10) * fh.max_crosslink_density,
            np.linspace(0.2, 0.5, 10) * fh.max_crosslink_density,
            np.linspace(0.5, 0.9, 10) * fh.max_crosslink_density,
        ]
        
        for rho in rho_values:
            # Convert to phi for numerical differentiation
            phi = fh.polymer_volume_fraction_from_crosslink_density(rho)
            h = 1e-8
            
            # Numerical first derivative w.r.t. phi
            f_plus = fh.per_volume(fh.crosslink_density_from_polymer_volume_fraction(phi + h), der="0")
            f_minus = fh.per_volume(fh.crosslink_density_from_polymer_volume_fraction(phi - h), der="0")
            df_dphi_num = (f_plus - f_minus) / (2 * h)
            
            # Analytical first derivative w.r.t. phi
            df_dphi_ana = fh.per_volume(rho, der="phi")
            
            np.testing.assert_allclose(df_dphi_ana, df_dphi_num, rtol=1e-5, 
                                     err_msg=f"Failed for phi derivative at rho range")

    def test_per_volume_phi2_derivative_numerical(self):
        """Test per_volume phi second derivatives against numerical differentiation."""
        fh = FloryHugginsPotential (
            chain_monomers=50,
            monomer_volume=4 * np.pi / 3,
            flory_chi=0.5,
            coordination=4,
        )
        
        # Test multiple density ranges
        rho_values = [
            np.linspace(0.01, 0.2, 10) * fh.max_crosslink_density,
            np.linspace(0.2, 0.5, 10) * fh.max_crosslink_density,
            np.linspace(0.5, 0.9, 10) * fh.max_crosslink_density,
        ]
        
        for rho in rho_values:
            # Convert to phi for numerical differentiation
            phi = fh.polymer_volume_fraction_from_crosslink_density(rho)
            h = 1e-6
            
            # Numerical second derivative w.r.t. phi
            df_plus = fh.per_volume(fh.crosslink_density_from_polymer_volume_fraction(phi + h), der="phi")
            df_minus = fh.per_volume(fh.crosslink_density_from_polymer_volume_fraction(phi - h), der="phi")
            d2f_dphi2_num = (df_plus - df_minus) / (2 * h)
            
            # Analytical second derivative w.r.t. phi
            d2f_dphi2_ana = fh.per_volume(rho, der="phi2")
            
            np.testing.assert_allclose(d2f_dphi2_ana, d2f_dphi2_num, rtol=1e-4,
                                     err_msg=f"Failed for phi2 derivative at rho range")

    def test_per_volume_rho_derivative_numerical(self):
        """Test per_volume rho derivatives against numerical differentiation."""
        fh = FloryHugginsPotential (
            chain_monomers=50,
            monomer_volume=4 * np.pi / 3,
            flory_chi=0.5,
            coordination=4,
        )
        
        # Test multiple density ranges  
        rho_values = [
            np.linspace(0.01, 0.2, 10) * fh.max_crosslink_density,
            np.linspace(0.2, 0.5, 10) * fh.max_crosslink_density,
            np.linspace(0.5, 0.9, 10) * fh.max_crosslink_density,
        ]
        
        for rho in rho_values:
            h = 1e-8
            
            # Numerical first derivative w.r.t. rho
            f_plus = fh.per_volume(rho + h, der="0")
            f_minus = fh.per_volume(rho - h, der="0")
            df_drho_num = (f_plus - f_minus) / (2 * h)
            
            # Analytical first derivative w.r.t. rho
            df_drho_ana = fh.per_volume(rho, der="rho")
            
            np.testing.assert_allclose(df_drho_ana, df_drho_num, rtol=1e-5,
                                     err_msg=f"Failed for rho derivative at rho range")

    def test_per_volume_rho2_derivative_numerical(self):
        """Test per_volume rho second derivatives against numerical differentiation."""
        fh = FloryHugginsPotential (
            chain_monomers=50,
            monomer_volume=4 * np.pi / 3,
            flory_chi=0.5,
            coordination=4,
        )
        
        # Test multiple density ranges
        rho_values = [
            np.linspace(0.01, 0.2, 10) * fh.max_crosslink_density,
            np.linspace(0.2, 0.5, 10) * fh.max_crosslink_density,
            np.linspace(0.5, 0.9, 10) * fh.max_crosslink_density,
        ]
        
        for rho in rho_values:
            h = 1e-6
            
            # Numerical second derivative w.r.t. rho
            df_plus = fh.per_volume(rho + h, der="rho")
            df_minus = fh.per_volume(rho - h, der="rho")
            d2f_drho2_num = (df_plus - df_minus) / (2 * h)
            
            # Analytical second derivative w.r.t. rho
            d2f_drho2_ana = fh.per_volume(rho, der="rho2")
            
            np.testing.assert_allclose(d2f_drho2_ana, d2f_drho2_num, rtol=1e-3,
                                     err_msg=f"Failed for rho2 derivative at rho range")

    def test_per_volume_derivative_chain_rule(self):
        """Test that chain rule is correctly applied for rho derivatives."""
        fh = FloryHugginsPotential (
            chain_monomers=50,
            monomer_volume=4 * np.pi / 3,
            flory_chi=0.5,
            coordination=4,
        )
        
        rho = np.linspace(0.01, 0.8, 20) * fh.max_crosslink_density
        
        # Get phi and dphi/drho
        phi = fh.polymer_volume_fraction_from_crosslink_density(rho)
        dphi_drho = fh.coord / 2.0 * fh.vchain  # From phi = rho * coord/2 * vchain
        
        # df/drho should equal df/dphi * dphi/drho (chain rule)
        df_dphi = fh.per_volume(rho, der="phi")
        df_drho_chain = df_dphi * dphi_drho
        df_drho_direct = fh.per_volume(rho, der="rho")
        
        np.testing.assert_allclose(df_drho_direct, df_drho_chain, rtol=1e-10,
                                 err_msg="Chain rule not satisfied for rho derivative")

    def test_per_volume_array_vs_scalar_consistency(self):
        """Test that per_volume gives consistent results for arrays vs scalars."""
        fh = FloryHugginsPotential (
            chain_monomers=50,
            monomer_volume=4 * np.pi / 3,
            flory_chi=0.5,
            coordination=4,
        )
        
        rho_array = np.array([0.1, 0.2, 0.5]) * fh.max_crosslink_density
        derivative_types = ["0", "phi", "phi2", "rho", "rho2"]
        
        for der_type in derivative_types:
            # Array result
            result_array = fh.per_volume(rho_array, der=der_type)
            
            # Scalar results
            result_scalars = np.array([fh.per_volume(rho_val, der=der_type) 
                                     for rho_val in rho_array])
            
            np.testing.assert_allclose(result_array, result_scalars, rtol=1e-12,
                                     err_msg=f"Array vs scalar inconsistency for der='{der_type}'")
