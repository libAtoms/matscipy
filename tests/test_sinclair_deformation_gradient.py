#
# Copyright 2026 James Kermode (Warwick U.)
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

import types

import numpy as np
from ase import Atoms

from matscipy.fracture_mechanics.crack import CubicCrystalCrack, SinclairCrack


def test_cauchy_born_F_func_is_component_first():
    """SinclairCrack.get_deformation_gradient is the Cauchy-Born corrector's F_func, which takes
    F_ab = dx_a/dX_b (it forms the right polar decomposition F = RU and rotates the shifts by R).
    Check it against finite differences of the displacement field under mixed-mode loading, where
    the field's local rotation makes F non-symmetric."""
    crk = CubicCrystalCrack([1, 1, 1], [1, -1, 0], C11=151.4, C12=76.4, C44=56.4)
    cryst = Atoms(cell=np.diag([80.0, 80.0, 3.84]))
    kI, kII, alpha = 1.0, 0.5, 0.3
    stub = types.SimpleNamespace(alpha=alpha, kI=kI, kII=kII, crk=crk, cryst=cryst)
    tip_x, tip_y = 40.0 + alpha, 40.0

    rng = np.random.default_rng(0)
    r = rng.uniform(4.0, 20.0, 50)
    theta = rng.uniform(-0.9 * np.pi, 0.9 * np.pi, 50)
    x, y = tip_x + r * np.cos(theta), tip_y + r * np.sin(theta)

    F = SinclairCrack.get_deformation_gradient(stub, x, y)

    h = 1e-5
    def u(xx, yy):
        return np.array(crk.displacements(xx, yy, tip_x, tip_y, kI, kII))    # (2, N)
    du_dx = (u(x + h, y) - u(x - h, y)) / (2 * h)
    du_dy = (u(x, y + h) - u(x, y - h)) / (2 * h)
    F_fd = np.eye(2) + np.stack([du_dx, du_dy], axis=-1).transpose(1, 0, 2)  # F[n, a, b] = d x_a / d X_b

    assert np.allclose(F, F_fd, atol=1e-8)
    assert not np.allclose(F, np.swapaxes(F_fd, 1, 2), atol=1e-4)  # the load makes F non-symmetric
