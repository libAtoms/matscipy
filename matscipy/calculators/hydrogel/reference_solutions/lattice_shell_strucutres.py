


class ShellStructure(ABC):
    dim: int = 3  # dimension of the lattice
    coordination: int  # coordination number of the lattice
    vpa_factor: float  # volume per atom for an interatomic distance of 1

    @abstractmethod
    def shellTensor(self, s):
        """Return the 2nd order shell tensor of the shell structure at distance s"""
        pass
    
    @abstractmethod
    def shellTensor4(self, s):
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

class Isotropic2DShellStructure(ShellStructure):
    dim: int = 2
    def shellTensor2(s):
        return shellTensor4_iso2

    def shellTensor4(s):
        return shellTensor4_iso2
        
class GraphiteShellStructure(Isotropic2DShellStructure):
    dim: int = 2
    coordination: int = 3
    vpa_factor: float = 3 * np.sqrt(3) / 2 /2

    def __init__(self, nb_shells=None, cutoff=None):
        a_vals, Z_vals = self._enumerate_shells(nb_shells=nb_shells, cutoff=cutoff)
        self.a = np.array(a_vals) # array containing the shell distances in units of the bond length
        self.Z = np.array(Z_vals) # array containing the coordination numbers of each shell

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

    def _D1_minus_D2(n: int) -> int:
        """D1(n) - D2(n): divisors ≡ 1 mod 3 minus divisors ≡ 2 mod 3."""
        d = _divisors(n)
        d1 = sum(1 for x in d if x % 3 == 1)
        d2 = sum(1 for x in d if x % 3 == 2)
        return d1 - d2

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
                Z = 3 * _D1_minus_D2(M)
            elif M % 3 == 0:
                sub = "A→A"
                Z = 6 * _D1_minus_D2(M // 3)
            else:
                sub = "—"
                Z = 0
            a_vals.append(np.sqrt(M))
            Z_vals.append(Z)

        return a_vals, Z_vals
