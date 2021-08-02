#
# Copyright 2021 Lars Pastewka (U. Freiburg)
#           2020 Wolfram G. Nöhring (U. Freiburg)
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
import numpy as np
import click
from matscipy.calculators.eam import io, average_atom

@click.command()
@click.argument("input_table", type=click.Path(exists=True, readable=True))
@click.argument("output_table", type=click.Path(exists=False, writable=True))
@click.argument("concentrations", nargs=-1)
def average(input_table, output_table, concentrations):
    """Create Average-atom potential for an Embedded Atom Method potential

    Read an EAM potential from INPUT_TABLE, create the Average-atom
    potential for the random alloy with composition specified
    by CONCENTRATIONS and write a new table with both the original
    and the A-atom potential functions to OUTPUT_TABLE.

    CONCENTRATIONS is a whitespace-separated list of the concentration of
    the elements, in the order in which the appear in the input table.

    """
    source, parameters, F, f, rep = io.read_eam(input_table)
    (new_parameters, new_F, new_f, new_rep) = average_atom.average_potential(
        np.array(concentrations, dtype=float), parameters, F, f, rep
    )
    composition = " ".join(
        [str(c * 100.0) + f"% {e}," for c, e in zip(np.array(concentrations, dtype=float), parameters.symbols)]
    )
    composition = composition.rstrip(",")
    source += f", averaged for composition {composition}"
    io.write_eam(
        source,
        new_parameters,
        new_F,
        new_f,
        new_rep,
        output_table,
        kind="eam/alloy",
    )


if __name__ == "__main__":
    average()
