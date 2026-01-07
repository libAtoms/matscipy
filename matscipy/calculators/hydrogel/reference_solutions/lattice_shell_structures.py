

from abc import ABC, abstractmethod
from typing import Literal
import numpy as np

class ShellStructure(ABC):
    dim: Literal[2, 3] = 3  # dimension of the lattice
    coordination: int  # coordination number of the lattice
    vpa_factor: float  # volume per atom for an interatomic distance of 1
    a: np.ndarray  # array of shell distances in units of the bond length
    Z: np.ndarray  # array of shell coordination numbers

    @abstractmethod
    def shellTensor2(self, s) -> np.ndarray:
        """Return the 2nd order shell tensor of the shell structure at distance s"""
        pass
    
    @abstractmethod
    def shellTensor4(self, s) -> np.ndarray:
        """Return the 4th order shell tensor of the shell structure at distance s"""
        pass
    
    def vpa(self, r):
        """Volume per atom at distance r"""
        return self.vpa_factor * r ** (self.dim)


########### Defining some constants

#### 2D 


dij = np.eye(2).reshape((2, 2, 1, 1))
dkl = np.eye(2).reshape((1, 1, 2, 2))
dik = np.eye(2).reshape((2, 1, 2, 1))
djl = np.eye(2).reshape((1, 2, 1, 2))
dil = np.eye(2).reshape((2, 1, 1, 2))
djk = np.eye(2).reshape((1, 2, 2, 1))

shellTensor4_iso2 = 1 / 8 * (dij * dkl + dik * djl + dil * djk)
shellTensor2_iso2 = 1 / 2 * np.eye(2) 

#### 3D 

dij = np.eye(3).reshape((3, 3, 1, 1))
dkl = np.eye(3).reshape((1, 1, 3, 3))
dik = np.eye(3).reshape((3, 1, 3, 1))
djl = np.eye(3).reshape((1, 3, 1, 3))
dil = np.eye(3).reshape((3, 1, 1, 3))
djk = np.eye(3).reshape((1, 3, 3, 1))
shellTensor4_iso3 = (dij * dkl + dik * djl + dil * djk)

shellTensor4_cubic = np.zeros((3, 3, 3, 3))
for i in range(3):
    shellTensor4_cubic[i, i, i, i] = 1.0

shellTensor2_cubic = shellTensor2_iso3 = np.eye(3) / 3.0

class Isotropic3DShellStructure(ShellStructure):
    dim = 3
    def shellTensor2(self, s):
        return shellTensor2_iso3

    def shellTensor4(self, s):
        return shellTensor4_iso3 / 15.0  # Normalized for 3D isotropic

class CubicShellStructure(ShellStructure):
    dim = 3
    def shellTensor2(self, s):
        return shellTensor2_cubic

class Isotropic2DShellStructure(ShellStructure):
    dim = 2
    def shellTensor2(self, s):
        return shellTensor2_iso2

    def shellTensor4(self, s):
        return shellTensor4_iso2
        
class GraphiteShellStructure(Isotropic2DShellStructure):
    dim = 2
    coordination = 3
    vpa_factor: float = 3 * np.sqrt(3) / 2 /2

    def __init__(self, nb_shells=None, cutoff=None):
        a_vals, Z_vals = self._enumerate_shells(nb_shells=nb_shells, cutoff=cutoff)
        self.a = np.array(a_vals) # array containing the shell distances in units of the bond length
        self.Z = np.array(Z_vals) # array containing the coordination numbers of each shell

    @staticmethod
    def _divisors(n: int):
        """Return sorted list of positive divisors of n."""
        if n <= 0:
            return []
        divs = set()
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
            i += 1
        return sorted(divs)

    @staticmethod
    def _D1_minus_D2(n: int) -> int:
        """D1(n) - D2(n): divisors ≡ 1 mod 3 minus divisors ≡ 2 mod 3."""
        d = GraphiteShellStructure._divisors(n)
        d1 = sum(1 for x in d if x % 3 == 1)
        d2 = sum(1 for x in d if x % 3 == 2)
        return d1 - d2

    @staticmethod
    def _enumerate_shells(nb_shells=None, cutoff=None):
        """Return first K shells (by increasing distance) for the 2D honeycomb lattice.

        nb_shells: number of shells to include
        cutoff: maximum distance to include shells, in units of the bond distances !
        
        Returns:
            a: list of shell radii
            Z: list of shell coordination numbers
        """

        assert not (nb_shells is None and cutoff is None), "Either nb_shells or cutoff must be provided"

        # Generate candidate M values by scanning N = m^2 - m n + n^2 up to a bound.
        Ms = set()
        B = 1
        while True:
            for m in range(-B, B+1):
                for n in range(-B, B+1):
                    M = m*m - m*n + n*n
                    if M > 0:
                        Ms.add(M)
            Ms_sorted = sorted(Ms)
            Ms_sorted = [M for M in Ms_sorted if M % 3 in (0, 1)]
            
            if nb_shells is not None and len(Ms_sorted) >= nb_shells:
                Ms_sorted = Ms_sorted[:nb_shells]
                break
            elif cutoff is not None:
                if np.sqrt(Ms_sorted[-1]) >= cutoff**2:
                    Ms_sorted = [M for M in Ms_sorted if np.sqrt(M) <= cutoff]
                    break
            B *= 2
        
        a_vals = []
        Z_vals = []

        for idx, M in enumerate(Ms_sorted, start=1):
            if M % 3 == 1:
                sub = "A→B"
                Z = 3 * __class__._D1_minus_D2(M)
            elif M % 3 == 0:
                sub = "A→A"
                Z = 6 * __class__._D1_minus_D2(M // 3)
            else:
                sub = "—"
                Z = 0
            a_vals.append(np.sqrt(M))
            Z_vals.append(Z)

        return a_vals, Z_vals


class DiamondShellStructure(CubicShellStructure):
    dim = 3
    coordination = 4
    vpa_factor: float = 8 * np.sqrt(3) / 9  # volume per atom for bond length = 1

    def __init__(self, nb_shells=None, cutoff=None, safety_margin_cells=2):
        shell_data = self._enumerate_shells(nb_shells=nb_shells, cutoff=cutoff, 
                                          safety_margin_cells=safety_margin_cells)
        self.a = np.array([d['r'] for d in shell_data])  # shell distances in units of bond length
        self.Z = np.array([d['Z'] for d in shell_data])  # coordination numbers
        self.isotropic_coeff = np.array([d['isotropic_coeff'] for d in shell_data])
        self.cubic_coeff = np.array([d['cubic_coeff'] for d in shell_data])

    def shellTensor4(self, s):
        """Return the 4th order shell tensor with cubic anisotropy"""
        if s >= len(self.isotropic_coeff):
            return np.zeros((3, 3, 3, 3))
        
        a_iso = self.isotropic_coeff[s]
        b_cub = self.cubic_coeff[s]
        
        # T4 = a*(δδ symmetric) + b*C where C has only diagonal terms
        # For cubic symmetry: use proper normalization
        return a_iso * shellTensor4_iso3 / 15.0 + b_cub * shellTensor4_cubic

    @staticmethod
    def _enumerate_shells(nb_shells=None, cutoff=None, safety_margin_cells=2):
        """Enumerate diamond lattice shells around an A-sublattice site.
        
        Parameters
        ----------
        nb_shells : int | None
            Number of shells to return (smallest radii first)
        cutoff : float | None  
            Maximum radius in units of bond length
        safety_margin_cells : int
            Extra conventional cells in each direction for completeness
            
        Returns
        -------
        list of dict with keys: r, Z, isotropic_coeff, cubic_coeff
        """
        assert not (nb_shells is None and cutoff is None), "Either nb_shells or cutoff must be provided"

        # Diamond conventional cell: FCC basis + shifted basis at (1/4,1/4,1/4)
        _FCC = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5], 
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ], dtype=float)
        
        _SHIFT = np.array([0.25, 0.25, 0.25], dtype=float)
        _DIAMOND_BASIS = np.vstack([_FCC, _FCC + _SHIFT])  # 8 atoms / conventional cell

        # Convert cutoff in bond units to conventional-cubic units
        # r_bond = (sqrt(3)/4) * a_conventional  
        if cutoff is not None:
            cutoff_a = cutoff * (np.sqrt(3.0) / 4.0)
            B = int(np.ceil(cutoff_a)) + safety_margin_cells
        else:
            B = 2

        def generate_vectors(Bcells):
            """Generate displacement vectors from origin to all atoms in [-B,B]^3 cells"""
            vecs = []
            for i in range(-Bcells, Bcells + 1):
                for j in range(-Bcells, Bcells + 1):
                    for k in range(-Bcells, Bcells + 1):
                        t = np.array([i, j, k], dtype=float)
                        for b in _DIAMOND_BASIS:
                            v = t + b
                            if np.allclose(v, 0.0):
                                continue
                            vecs.append(v)
            return np.array(vecs, dtype=float)

        while True:
            vecs = generate_vectors(B)
            
            # Use exact integer keys to avoid float grouping issues
            # All coordinates are multiples of 1/4, so u = 4*v is integer-valued
            u = np.rint(4.0 * vecs).astype(int)
            m2 = np.sum(u * u, axis=1).astype(int)
            
            # Exclude outside cutoff if requested
            if cutoff is not None:
                # r_shell = sqrt(m2/3) must be <= cutoff
                mask = (m2 <= int(np.floor(3.0 * cutoff * cutoff + 1e-12)))
                u = u[mask]
                m2 = m2[mask]

            uniq_m2 = np.unique(m2)
            uniq_m2.sort()

            if nb_shells is not None:
                if len(uniq_m2) >= nb_shells:
                    uniq_m2 = uniq_m2[:nb_shells]
                    break
                B *= 2
                continue
            else:
                break

        shells = []
        for mm in uniq_m2:
            shell_mask = (m2 == mm)
            u_shell = u[shell_mask]
            Z = u_shell.shape[0]

            # Unit vectors: e = u / sqrt(mm)
            e = u_shell.astype(float) / np.sqrt(float(mm))

            # 4th-rank tensor components
            T4 = np.einsum("ni,nj,nk,nl->ijkl", e, e, e, e) / Z
            A = T4[0, 0, 0, 0]       # <e_x^4>
            Bxy = T4[0, 0, 1, 1]     # <e_x^2 e_y^2>

            # Decomposition: T4 = a*(δδ sym) + b*C
            a_iso = Bxy
            b_cub = A - 3.0 * Bxy

            shells.append({
                "r": np.sqrt(mm / 3.0),     # shell radius in bond units
                "Z": int(Z),
                "isotropic_coeff": float(a_iso),
                "cubic_coeff": float(b_cub),
            })

        return shells


