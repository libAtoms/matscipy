"""
Minimal implementation of an embedded atom method (EAM) potential
"""
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
# import numpy as jnp
from ase.build import bulk

from matscipy.neighbours import neighbour_list


def F(density):
    return -jnp.sqrt(density)


def dF(density):
    return -0.5 / jnp.sqrt(density)


def f(distance, xi=1.224, q=2.278, r0=3.615 / jnp.sqrt(2)):
    return xi ** 2 * jnp.exp(-2 * q * (distance / r0 - 1))


# @jax.jit
def df(distance, xi=1.224, q=2.278, r0=3.615 / jnp.sqrt(2)):
    return -2 * xi ** 2 * q / r0 * jnp.exp(-2 * q * (distance / r0 - 1))


def rep(distance, A=0.0855, p=10.960, r0=3.615 / jnp.sqrt(2)):
    return A * jnp.exp(-p * (distance / r0 - 1))


def drep(distance, A=0.0855, p=10.960, r0=3.615 / jnp.sqrt(2)):
    return -p * A * jnp.exp(-p * (distance / r0 - 1)) / r0


def energy_and_forces(atoms, cutoff=10.0):
    # Construct neighbor list
    i_p, j_p, d_p, D_pc = neighbour_list("ijdD", atoms, cutoff)

    @jax.jit
    def calculate(density_i, i_p, d_p, D_pc):
        energy = F(density_i).sum() + rep(d_p).sum() / 2
        df_pc = -((dF(density_i[i_p]) * df(d_p) + drep(d_p) / 2) * D_pc.T / d_p).T
        return energy, df_pc

    density_i = jnp.bincount(i_p, weights=f(d_p), minlength=len(atoms))
    energy, df_pc = calculate(density_i, i_p, d_p, D_pc)

    fx_i = jnp.bincount(j_p, weights=df_pc[:, 0], minlength=len(atoms)) - \
           jnp.bincount(i_p, weights=df_pc[:, 0], minlength=len(atoms))
    fy_i = jnp.bincount(j_p, weights=df_pc[:, 1], minlength=len(atoms)) - \
           jnp.bincount(i_p, weights=df_pc[:, 1], minlength=len(atoms))
    fz_i = jnp.bincount(j_p, weights=df_pc[:, 2], minlength=len(atoms)) - \
           jnp.bincount(i_p, weights=df_pc[:, 2], minlength=len(atoms))

    return energy, jnp.transpose(jnp.array([fx_i, fy_i, fz_i]))


def numeric_force(atoms, a, i, d=0.001):
    p0 = atoms.get_positions()
    p = p0.copy()
    p[a, i] += d
    atoms.set_positions(p, apply_constraint=False)
    eplus, _ = energy_and_forces(atoms)
    p[a, i] -= 2 * d
    atoms.set_positions(p, apply_constraint=False)
    eminus, _ = energy_and_forces(atoms)
    atoms.set_positions(p0, apply_constraint=False)
    return (eminus - eplus) / (2 * d)


def numeric_forces(atoms, d=0.001):
    return np.array([[numeric_force(atoms, a, i, d)
                      for i in range(3)] for a in range(len(atoms))])


def verlet1(atoms, forces, dt=0.01):
    atoms.set_momenta(atoms.get_momenta() + 0.5 * forces * dt)
    atoms.set_positions(atoms.get_positions() + atoms.get_velocities() * dt)


def verlet2(atoms, forces, dt=0.01):
    atoms.set_momenta(atoms.get_momenta() + 0.5 * forces * dt)


atoms = bulk('Al', 'fcc', a=4.05, cubic=True)
atoms *= (16, 16, 16)  # Create a large supercell
print(len(atoms))
atoms.rattle(0.1)

if False:
    energy, forces = energy_and_forces(atoms)

    numforces = numeric_forces(atoms)

    print(forces)
    print(numforces)

    np.testing.assert_allclose(forces, numforces, rtol=1e-5)

energy, forces = energy_and_forces(atoms)
for i in range(1000):
    verlet1(atoms, forces, dt=0.1)
    energy, forces = energy_and_forces(atoms)
    verlet2(atoms, forces, dt=0.1)

    if i % 10 == 0:
        print(f"Step {i}, Energy: {energy:.3f}")
