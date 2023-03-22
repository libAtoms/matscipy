#
# Copyright 2021 Jan Griesser (U. Freiburg)
#           2021 Lars Pastewka (U. Freiburg)
#
# matscipy - Materials science with Python at the atomic-scale
# https://github.com/libAtoms/matscipy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

"""
Pair potential + Ewald summation
"""

#
# Coding convention
# * All numpy arrays are suffixed with the array dimensions
# * The suffix stands for a certain type of dimension:
#   - n: Atomic index, i.e. array dimension of length nb_atoms
#   - p: Pair index, i.e. array dimension of length nb_pairs
#   - c: Cartesian index, array dimension of length 3
#   - l: Wave vector index, i.e. array of dimension length of k_lc

from collections import defaultdict
import numpy as np

from scipy.linalg import block_diag
from scipy.special import erfc

from ...calculators.pair_potential.calculator import (
    PairPotential,
    CutoffInteraction,
)

# Charges q are expressed as multiples of the elementary charge e: q = x*e
# e^2/(4*pi*epsilon0) = 14.399645 eV * Angström
conversion_prefactor = 14.399645


class EwaldShortRange(CutoffInteraction):
    """Short range term of Ewald summation."""

    def __init__(self, alpha, cutoff):
        super().__init__(cutoff)
        self.alpha = alpha

    def __call__(self, r, qi, qj):
        return conversion_prefactor * qi * qj * erfc(self.alpha * r) / r

    def first_derivative(self, r, qi, qj):
        a = self.alpha
        return (
            -conversion_prefactor
            * qi
            * qj
            * (
                erfc(a * r) / r**2
                + 2 * a * np.exp(-((a * r) ** 2)) / (np.sqrt(np.pi) * r)
            )
        )

    def second_derivative(self, r, qi, qj):
        a = self.alpha
        return (
            conversion_prefactor
            * qi
            * qj
            * (
                2 * erfc(a * r) / r**3
                + 4
                * a
                * np.exp((-((a * r) ** 2)))
                / np.sqrt(np.pi)
                * (1 / r**2 + a**2)
            )
        )


class Ewald(PairPotential):
    """Ewal summation calculator."""

    name = "Ewald"

    default_parameters = {
        "accuracy": 1e-6,
        "cutoff": 3,
        "verbose": True,
        "kspace": {},
    }

    def __init__(self):
        super().__init__(defaultdict(lambda: self.short_range))

        self.set(**self.parameters)
        self.kvectors = None
        self.initial_I = None
        self.initial_alpha = None

    @property
    def short_range(self):
        return EwaldShortRange(
            self.alpha,
            self.parameters["cutoff"],
        )

    @property
    def alpha(self):
        """Get alpha."""
        return self._alpha

    @alpha.setter
    def alpha(self, v):
        """Set alpha."""
        self._alpha = v

    def set(self, **kwargs):
        super().set(**kwargs)

        if "accuracy" in kwargs:
            self.reset()

    def reset(self):
        super().reset()
        self.dict = defaultdict(lambda: self.short_range.cutoff)
        self.df = defaultdict(lambda: self.short_range.derivative(1))
        self.df2 = defaultdict(lambda: self.short_range.derivative(2))

    def _mask_pairs(self, i_p, j_p):
        """Match all atom types to the (1, 1) object for pair interactions."""
        yield np.s_[:], (1, 1)

    @staticmethod
    def determine_alpha(charge, acc, cutoff, cell):
        """
        Determine an estimate for alpha on the basis of the cell, cutoff and desired accuracy
        (Adopted from LAMMPS)
        """

        # The kspace rms error is computed relative to the force that two unit point
        # charges exert on each other at a distance of 1 Angström
        accuracy_relative = acc * conversion_prefactor

        qsqsum = conversion_prefactor * np.sum(charge**2)

        a = (
            accuracy_relative
            * np.sqrt(
                len(charge) * cutoff * cell[0, 0] * cell[1, 1] * cell[2, 2]
            )
            / (2 * qsqsum)
        )

        if a >= 1.0:
            return (1.35 - 0.15 * np.log(accuracy_relative)) / cutoff

        else:
            return np.sqrt(-np.log(a)) / cutoff

    @staticmethod
    def determine_nk(charge, cell, acc, a, natoms):
        """
        Determine the maximal number of points in reciprocal space for each direction,
        and the cutoff in reciprocal space
        """

        # The kspace rms error is computed relative to the force that two unit point
        # charges exert on each other at a distance of 1 Angström
        accuracy_relative = acc * conversion_prefactor

        nxmax = 1
        nymax = 1
        nzmax = 1

        qsqsum = conversion_prefactor * np.sum(charge**2)

        error = Ewald.rms_kspace(nxmax, cell[0, 0], natoms, a, qsqsum)
        while error > (accuracy_relative):
            nxmax += 1
            error = Ewald.rms_kspace(nxmax, cell[0, 0], natoms, a, qsqsum)

        error = Ewald.rms_kspace(nymax, cell[1, 1], natoms, a, qsqsum)
        while error > (accuracy_relative):
            nymax += 1
            error = Ewald.rms_kspace(nymax, cell[1, 1], natoms, a, qsqsum)

        error = Ewald.rms_kspace(nzmax, cell[2, 2], natoms, a, qsqsum)
        while error > (accuracy_relative):
            nzmax += 1
            error = Ewald.rms_kspace(nzmax, cell[2, 2], natoms, a, qsqsum)

        kxmax = 2 * np.pi / cell[0, 0] * nxmax
        kymax = 2 * np.pi / cell[1, 1] * nymax
        kzmax = 2 * np.pi / cell[2, 2] * nzmax

        kmax = max(kxmax, kymax, kzmax)

        # Check if box is triclinic --> Scale lattice vectors for triclinic skew
        if np.count_nonzero(cell - np.diag(np.diagonal(cell))) != 9:
            vector = np.array(
                [nxmax / cell[0, 0], nymax / cell[1, 1], nzmax / cell[2, 2]]
            )
            scaled_nbk = np.dot(np.array(np.abs(cell)), vector)
            nxmax = max(1, np.int(scaled_nbk[0]))
            nymax = max(1, np.int(scaled_nbk[1]))
            nzmax = max(1, np.int(scaled_nbk[2]))

        return kmax, np.array([nxmax, nymax, nzmax])

    @staticmethod
    def determine_kc(cell, nk):
        """
        Determine maximal wave vector based in a given integer triplet
        """
        kxmax = 2 * np.pi / cell[0, 0] * nk[0]
        kymax = 2 * np.pi / cell[1, 1] * nk[1]
        kzmax = 2 * np.pi / cell[2, 2] * nk[2]

        return max(kxmax, kymax, kzmax)

    @staticmethod
    def rms_kspace(km, l, n, a, q2):
        """
        Compute the root mean square error of the force in reciprocal space

        Reference
        ------------------
        Henrik G. Petersen, The Journal of chemical physics 103.9 (1995)
        """

        return (
            2
            * q2
            * a
            / l
            * np.sqrt(1 / (np.pi * km * n))
            * np.exp(-((np.pi * km / (a * l)) ** 2))
        )

    @staticmethod
    def rms_rspace(charge, cell, a, rc):
        """
        Compute the root mean square error of the force in real space

        Reference
        ------------------
        Henrik G. Petersen, The Journal of chemical physics 103.9 (1995)
        """

        return (
            2
            * np.sum(charge**2)
            * np.exp(-((a * rc) ** 2))
            / np.sqrt(rc * len(charge) * cell[0, 0] * cell[1, 1] * cell[2, 2])
        )

    @staticmethod
    def allowed_wave_vectors(cell, km, a, nk):
        """
        Compute allowed wave vectors and the prefactor I
        """
        nx = np.arange(-nk[0], nk[0] + 1, 1)
        ny = np.arange(-nk[1], nk[1] + 1, 1)
        nz = np.arange(-nk[2], nk[2] + 1, 1)

        n_lc = np.array(np.meshgrid(nx, ny, nz)).T.reshape(-1, 3)

        k_lc = 2 * np.pi * np.dot(np.linalg.inv(np.array(cell)), n_lc.T).T

        k = np.linalg.norm(k_lc, axis=1)

        mask = np.logical_and(k <= km, k != 0)

        return np.exp(-((k[mask] / (2 * a)) ** 2)) / k[mask] ** 2, k_lc[mask]

    @staticmethod
    def self_energy(charge, a):
        """
        Return the self energy
        """
        return -conversion_prefactor * a * np.sum(charge**2) / np.sqrt(np.pi)

    @staticmethod
    def kspace_energy(charge, pos, vol, I, k):
        """
        Return the energy from the reciprocal space contribution
        """

        structure_factor_l = np.sum(
            charge * np.exp(1j * np.tensordot(k, pos, axes=((1), (1)))), axis=1
        )

        return (
            conversion_prefactor
            * 2
            * np.pi
            * np.sum(I * np.absolute(structure_factor_l) ** 2)
            / vol
        )

    @staticmethod
    def first_derivative_kspace(charge, natoms, vol, pos, I, k):
        """Return the kspace part of the force."""
        n = len(pos)

        phase_ln = np.tensordot(k, pos, axes=((1), (1)))

        cos_ln = np.cos(phase_ln)
        sin_ln = np.sin(phase_ln)

        cos_sin_ln = (cos_ln.T * np.sum(charge * sin_ln, axis=1)).T
        sin_cos_ln = (sin_ln.T * np.sum(charge * cos_ln, axis=1)).T

        prefactor_ln = (I * (cos_sin_ln - sin_cos_ln).T).T

        f_nc = np.sum(
            k.reshape(-1, 1, 3) * prefactor_ln.reshape(-1, n, 1), axis=0
        )

        return -conversion_prefactor * 4 * np.pi * (charge * f_nc.T).T / vol

    @staticmethod
    def stress_kspace(charge, pos, vol, a, I, k):
        """Return the stress contribution of the long-range Coulomb part."""
        sqk_l = np.sum(k * k, axis=1)

        structure_factor_l = np.sum(
            charge * np.exp(1j * np.tensordot(k, pos, axes=((1), (1)))), axis=1
        )

        wave_vectors_lcc = (k.reshape(-1, 3, 1) * k.reshape(-1, 1, 3)) * (
            1 / (2 * a**2) + 2 / sqk_l
        ).reshape(-1, 1, 1) - np.identity(3)

        stress_lcc = (I * np.absolute(structure_factor_l) ** 2).reshape(
            len(I), 1, 1
        ) * wave_vectors_lcc

        stress_cc = np.sum(stress_lcc, axis=0)

        stress_cc *= conversion_prefactor * 2 * np.pi / vol

        return np.array(
            [
                stress_cc[0, 0],  # xx
                stress_cc[1, 1],  # yy
                stress_cc[2, 2],  # zz
                stress_cc[1, 2],  # yz
                stress_cc[0, 2],  # xz
                stress_cc[0, 1],
            ]
        )  # xy

    def reset_kspace(self, atoms):
        """Reset kspace setup."""
        if not atoms.has("charge"):
            raise AttributeError(
                "Unable to load atom charges from atoms object!"
            )

        charge_n = atoms.get_array("charge")

        if np.abs(charge_n.sum()) > 1e-3:
            print("Net charge: ", np.sum(charge_n))
            raise AttributeError("System is not charge neutral!")

        if not all(atoms.get_pbc()):
            raise AttributeError(
                "This code only works for 3D systems with periodic boundaries!"
            )

        accuracy = self.parameters["accuracy"]
        rc = self.parameters["cutoff"]
        kspace_params = self.parameters["kspace"]

        self.alpha = kspace_params.get(
            "alpha",
            self.determine_alpha(charge_n, accuracy, rc, atoms.get_cell()),
        )

        alpha = self.alpha
        nb_atoms = len(atoms)

        if "nbk_c" in kspace_params:
            nbk_c = kspace_params["nbk_c"]
            kc = kspace_params.get(
                "cutoff", self.determine_kc(atoms.get_cell(), nbk_c)
            )
        else:
            kc, nbk_c = self.determine_nk(
                charge_n, atoms.get_cell(), accuracy, alpha, nb_atoms
            )

        self.set(cutoff_kspace=kc)
        self.initial_alpha = alpha

        I_l, k_lc = self.allowed_wave_vectors(
            atoms.get_cell(), kc, alpha, nbk_c
        )

        self.kvectors = k_lc
        self.initial_I = I_l

        # Priting info
        if self.parameters.get("verbose"):
            rms_rspace = self.rms_rspace(charge_n, atoms.get_cell(), alpha, rc)
            rms_kspace = [
                self.rms_kspace(
                    nbk_c[i],
                    atoms.get_cell()[i, i],
                    nb_atoms,
                    alpha,
                    conversion_prefactor * np.sum(charge_n**2),
                )
                for i in range(3)
            ]

            print("Estimated alpha: ", alpha)
            print("Number of wave vectors: ", k_lc.shape[0])
            print("Cutoff for kspace vectors: ", kc)
            print(
                "Estimated kspace triplets nx/ny/nx: ",
                nbk_c[0],
                "/",
                nbk_c[1],
                "/",
                nbk_c[2],
            )
            print(
                "Estimated absolute RMS force accuracy (Real space): ",
                np.absolute(rms_rspace),
            )
            print(
                "Estimated absolute RMS force accuracy (Kspace): ",
                np.linalg.norm(rms_kspace),
            )

    def calculate(self, atoms, properties, system_changes):
        """Compute Coulomb interactions with Ewald summation."""
        if "cell" in system_changes or getattr(self, 'alpha', None) is None:
            self.reset_kspace(atoms)

        super().calculate(atoms, properties, system_changes)

        nb_atoms = len(atoms)
        charge_n = atoms.get_array("charge")

        k_lc = self.kvectors
        I_l = self.initial_I
        alpha = self.alpha

        # Energy
        e_self = self.self_energy(charge_n, alpha)
        e_long = self.kspace_energy(
            charge_n, atoms.get_positions(), atoms.get_volume(), I_l, k_lc
        )

        self.results["energy"] += e_self + e_long
        self.results["free_energy"] += e_self + e_long

        # Forces
        self.results["forces"] += self.first_derivative_kspace(
            charge_n,
            nb_atoms,
            atoms.get_volume(),
            atoms.get_positions(),
            I_l,
            k_lc,
        )

        # Virial
        self.results["stress"] += (
            self.stress_kspace(
                charge_n,
                atoms.get_positions(),
                atoms.get_volume(),
                alpha,
                I_l,
                k_lc,
            )
            / atoms.get_volume()
        )

    def kspace_properties(self, atoms, prop="Hessian", divide_by_masses=False):
        """
        Calculate the recirprocal contributiom to the Hessian, the non-affine
        forces and the Born elastic constants

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        prop: "Hessian", "Born" or "Naforces"
            Compute either the Hessian/Dynamical matrix, the Born constants
            or the non-affine forces.

        divide_by_masses: bool
            if true return the dynamic matrix else Hessian matrix

        Restrictions
        ----------
        This method is currently only implemented for three dimensional systems
        """
        nb_atoms = len(atoms)
        alpha = self.alpha
        k_lc = self.kvectors
        I_l = self.initial_I
        charge_n = atoms.get_array("charge")

        if prop == "Hessian":
            H = np.zeros((3 * nb_atoms, 3 * nb_atoms))

            pos = atoms.get_positions()

            for i, k in enumerate(k_lc):
                phase_l = np.sum(k * pos, axis=1)

                I_sqcos_sqsin = I_l[i] * (
                    np.cos(phase_l).reshape(-1, 1)
                    * np.cos(phase_l).reshape(1, -1)
                    + np.sin(phase_l).reshape(-1, 1)
                    * np.sin(phase_l).reshape(1, -1)
                )

                I_sqcos_sqsin[range(nb_atoms), range(nb_atoms)] = 0.0

                H += np.concatenate(
                    np.concatenate(
                        k.reshape(1, 1, 3, 1)
                        * k.reshape(1, 1, 1, 3)
                        * I_sqcos_sqsin.reshape(nb_atoms, nb_atoms, 1, 1),
                        axis=2,
                    ),
                    axis=0,
                )

            H *= (
                (conversion_prefactor * 4 * np.pi / atoms.get_volume())
                * charge_n.repeat(3).reshape(-1, 1)
                * charge_n.repeat(3).reshape(1, -1)
            )

            Hdiag = np.zeros((3 * nb_atoms, 3))
            for x in range(3):
                Hdiag[:, x] = -np.sum(H[:, x::3], axis=1)

            Hdiag = block_diag(*Hdiag.reshape(nb_atoms, 3, 3))

            H += Hdiag

            if divide_by_masses:
                masses_p = (atoms.get_masses()).repeat(3)
                H /= np.sqrt(masses_p.reshape(-1, 1) * masses_p.reshape(1, -1))

            return H

        elif prop == "Born":
            delta_ab = np.identity(3)
            sqk_l = np.sum(k_lc * k_lc, axis=1)

            structure_factor_l = np.sum(
                charge_n
                * np.exp(
                    1j
                    * np.tensordot(
                        k_lc, atoms.get_positions(), axes=((1), (1))
                    )
                ),
                axis=1,
            )
            prefactor_l = (I_l * np.absolute(structure_factor_l) ** 2).reshape(
                -1, 1, 1, 1, 1
            )

            # First expression
            first_abab = delta_ab.reshape(1, 3, 3, 1, 1) * delta_ab.reshape(
                1, 1, 1, 3, 3
            ) + delta_ab.reshape(1, 1, 3, 3, 1) * delta_ab.reshape(
                1, 3, 1, 1, 3
            )

            # Second expression
            prefactor_second_l = -(1 / (2 * alpha**2) + 2 / sqk_l).reshape(
                -1, 1, 1, 1, 1
            )
            second_labab = (
                k_lc.reshape(-1, 1, 1, 3, 1)
                * k_lc.reshape(-1, 1, 1, 1, 3)
                * delta_ab.reshape(1, 3, 3, 1, 1)
                + k_lc.reshape(-1, 3, 1, 1, 1)
                * k_lc.reshape(-1, 1, 1, 3, 1)
                * delta_ab.reshape(1, 1, 3, 1, 3)
                + k_lc.reshape(-1, 3, 1, 1, 1)
                * k_lc.reshape(-1, 1, 3, 1, 1)
                * delta_ab.reshape(1, 1, 1, 3, 3)
                + k_lc.reshape(-1, 1, 3, 1, 1)
                * k_lc.reshape(-1, 1, 1, 3, 1)
                * delta_ab.reshape(1, 3, 1, 1, 3)
                + k_lc.reshape(-1, 3, 1, 1, 1)
                * k_lc.reshape(-1, 1, 1, 1, 3)
                * delta_ab.reshape(1, 1, 3, 3, 1)
            )

            # Third expression
            prefactor_third_l = (
                1 / (4 * alpha**4)
                + 2 / (alpha**2 * sqk_l)
                + 8 / sqk_l**2
            ).reshape(-1, 1, 1, 1, 1)
            third_labab = (
                k_lc.reshape(-1, 3, 1, 1, 1)
                * k_lc.reshape(-1, 1, 3, 1, 1)
                * k_lc.reshape(-1, 1, 1, 3, 1)
                * k_lc.reshape(-1, 1, 1, 1, 3)
            )

            C_labab = prefactor_l * (
                first_abab
                + prefactor_second_l * second_labab
                + prefactor_third_l * third_labab
            )

            return (
                conversion_prefactor
                * 2
                * np.pi
                * np.sum(C_labab, axis=0)
                / atoms.get_volume() ** 2
            )

        elif prop == "Naforces":
            delta_ab = np.identity(3)
            sqk_l = np.sum(k_lc * k_lc, axis=1)

            phase_ln = np.tensordot(
                k_lc, atoms.get_positions(), axes=((1), (1))
            )

            cos_ln = np.cos(phase_ln)
            sin_ln = np.sin(phase_ln)

            cos_sin_ln = (cos_ln.T * np.sum(charge_n * sin_ln, axis=1)).T
            sin_cos_ln = (sin_ln.T * np.sum(charge_n * cos_ln, axis=1)).T

            prefactor_ln = (I_l * (cos_sin_ln - sin_cos_ln).T).T

            # First expression
            first_lccc = (
                (1 / (2 * alpha**2) + 2 / sqk_l).reshape(-1, 1, 1, 1)
                * k_lc.reshape(-1, 1, 1, 3)
                * k_lc.reshape(-1, 3, 1, 1)
                * k_lc.reshape(-1, 1, 3, 1)
            )

            # Second expression
            second_lccc = -(
                k_lc.reshape(-1, 3, 1, 1) * delta_ab.reshape(-1, 1, 3, 3)
                + k_lc.reshape(-1, 1, 3, 1) * delta_ab.reshape(-1, 3, 1, 3)
            )

            naforces_nccc = np.sum(
                prefactor_ln.reshape(-1, nb_atoms, 1, 1, 1)
                * (first_lccc + second_lccc).reshape(-1, 1, 3, 3, 3),
                axis=0,
            )

            return (
                -conversion_prefactor
                * 4
                * np.pi
                * (charge_n * naforces_nccc.T).T
                / atoms.get_volume()
            )

    def get_hessian(self, atoms, format=""):
        """
        Compute the real space + kspace Hessian
        """
        # Ignore kspace here
        if format == "neighbour-list":
            return super().get_hessian(atoms, format=format)

        return super().get_hessian(
            atoms, format="sparse"
        ).todense() + self.kspace_properties(atoms, prop="Hessian")

    def get_nonaffine_forces(self, atoms):
        """
        Compute the non-affine forces which result from an affine deformation of atoms.

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        """
        return super().get_nonaffine_forces(atoms) + self.kspace_properties(
            atoms, prop="Naforces"
        )

    def get_born_elastic_constants(self, atoms):
        """
        Compute the Born elastic constants.

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        """
        C_abab = super().get_born_elastic_constants(
            atoms
        ) + self.kspace_properties(atoms, prop="Born")

        # Symmetrize elastic constant tensor
        C_abab = (
            C_abab
            + C_abab.swapaxes(0, 1)
            + C_abab.swapaxes(2, 3)
            + C_abab.swapaxes(0, 1).swapaxes(2, 3)
        ) / 4

        return C_abab

    def get_derivative_volume(self, atoms, d=1e-6):
        """
        Calculate the change of volume with strain using central differences
        """
        cell = atoms.cell.copy()
        vol = np.zeros((3, 3))

        for i in range(3):
            # Diagonal
            x = np.eye(3)
            x[i, i] += d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            Vplus = atoms.get_volume()

            x[i, i] -= 2 * d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            Vminus = atoms.get_volume()

            derivative_volume = (Vplus - Vminus) / (2 * d)

            vol[i, i] = derivative_volume

            # Off diagonal
            j = i - 2
            x[i, j] = d
            x[j, i] = d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            Vplus = atoms.get_volume()

            x[i, j] = -d
            x[j, i] = -d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            Vminus = atoms.get_volume()

            derivative_volume = (Vplus - Vminus) / (4 * d)
            vol[i, j] = derivative_volume
            vol[j, i] = derivative_volume

        return vol

    def get_derivative_wave_vector(self, atoms, d=1e-6):
        """
        Calculate the change of volume with strain using central differences
        """
        cell = atoms.cell.copy()

        e = np.ones(3)
        initial_k = 2 * np.pi * np.dot(np.linalg.inv(cell), e)
        print("Wave vector for n=(1,1,1): ", initial_k)

        def_k = np.zeros((3, 3, 3))

        for i in range(3):
            # Diagonal
            x = np.eye(3)
            x[i, i] += d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            k_pos = 2 * np.pi * np.dot(np.linalg.inv(atoms.get_cell()), e)

            x[i, i] -= 2 * d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            k_minus = 2 * np.pi * np.dot(np.linalg.inv(atoms.get_cell()), e)
            derivative_k = (k_pos - k_minus) / (2 * d)
            def_k[:, i, i] = derivative_k

            # Off diagonal --> xy, xz, yz
            j = i - 2
            x[i, j] = d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            k_pos = 2 * np.pi * np.dot(np.linalg.inv(atoms.get_cell()), e)

            x[i, j] = -d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            k_minus = 2 * np.pi * np.dot(np.linalg.inv(atoms.get_cell()), e)

            derivative_k = (k_pos - k_minus) / (2 * d)
            def_k[:, i, j] = derivative_k

            # Odd diagonal --> yx, zx, zy
            x[j, i] = d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            k_pos = 2 * np.pi * np.dot(np.linalg.inv(atoms.get_cell()), e)

            x[j, i] = -d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            k_minus = 2 * np.pi * np.dot(np.linalg.inv(atoms.get_cell()), e)

            derivative_k = (k_pos - k_minus) / (2 * d)
            def_k[:, j, i] = derivative_k

        return def_k
