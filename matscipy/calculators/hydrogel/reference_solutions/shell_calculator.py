
class ShellHydrogelCalculator():
    """Hydrogel calculator using shell reference solutions."""


    def __init__(self, shell_structure: ShellStructure, 
                       chain_potential: ChainPotential, 
                       embedding_potential: EmbeddingPotential,
                       weight_function: WeightFunction
                       , ):
        """
        Initialize the shell hydrogel calculator.

        Parameters
        ----------
        shell_structure : ShellStructure
            The shell structure to use.
        chain_potential : ChainPotential
            The chain potential to use.
        flory_chi : float
            The Flory-Huggins interaction parameter.
        """
        self.shell_structure = shell_structure
        self.chain_potential = chain_potential
        self.embedding_potential = embedding_potential
        self.weight_function = weight_function

    # some convenience functions to call weight function methods
    def f(self, r):
        self.weight_function(r)
    def fp(self, r):
        self.weight_function.derivative(r)
    def fpp(self, r):
        self.weight_function.second_derivative(r)

    def ft(self, r2):
        self.f(np.sqrt(r2))
    def ftp(self, r2):
        self.fp(np.sqrt(r2)) / ( 2 * np.sqrt(r2))
    def ftpp(self, r2):
        r = np.sqrt(r2)
        return self.fpp(r) / (4 * r2) - self.fp(r) / (4 * r2 * r)

    def F(self, rho):
        self.embedding_potential(rho)
    def Fp(self, rho):
        self.embedding_potential.derivative(rho)
    def Fpp(self, rho):
        self.embedding_potential.second_derivative(rho)

    def density_noself(self, r):
        """
        Electron density at distance r, excluding self-contribution
        """
        # First neighbors:
        rho = 0
        rn = self.shell_structure.a * r
        Zn = self.shell_structure.Z
        rho = jnp.sum( Zn * self.f(rn))
        return rho 
    
    def density(self, r):
        """
        Electron density at distance r, including self-contribution
        """
        return self.density_noself(r) + self.f(0)
    

    def stiffness_matrix(self, r):
        """
        Stiffness matrix from EAM potential
        """
        dim = self.dim
        ρ0 = self.density_noself(r)

        a = self.shell_structure.a
        Z = self.shell_structure.Z
        term1 = 0
        term2 = 0

        # TODO: vectorize this ! 
        for n in range(0, len(a) ):
            # Note that here we define the shell tensor as nu / Z , where nu is the shell tensor as defined in Muser, Sukhomlinov, Pastewka
            term1 += self.ftpp(r ** 2  * a[n]**2) * r**4 * a[n]**4 * self.shell_structure.shellTensor4(n) * Z[n]
            for m in range(0, len(a)): 

                term2 += self.ftp(r**2 * a[n]**2) * self.ftp(r ** 2 * a[m]**2) * r**4 *  a[n]**2 * a[m]**2 * Z[n] * Z[m] \
                  * self.shell_structure.shellTensor2(n).reshape(dim, dim, 1, 1) * self.shell_structure.shellTensor2(m).reshape(1, 1, dim, dim)
        
        term1 *= self.Fp(ρ0)
        term2 *= self.Fpp(ρ0)

        return 4  * (
            term1
            + term2
            ) * self.density(r)
    
