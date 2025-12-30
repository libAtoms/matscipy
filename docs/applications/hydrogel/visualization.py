from pathlib import Path

import numpy as np
import nglview as nv
from ase.geometry import find_mic


def write_pdb_conect(atoms, bonds, filename="network.pdb"):
    filename = Path(filename)

    # Build adjacency for CONECT records
    adj = [[] for _ in range(len(atoms))]
    for i, j in bonds:
        adj[i].append(j)
        adj[j].append(i)

    with filename.open("w") as f:
        f.write("MODEL        1\n")
        for aidx, atom in enumerate(atoms, start=1):  # PDB is 1-based
            x, y, z = atom.position
            sym = (atom.symbol or "X")[:2]
            f.write(
                f"ATOM  {aidx:5d} {sym:<4s} NET A   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {sym:>2s}\n"
            )

        # CONECT lines (split into chunks of 4 partners per line)
        for i0 in range(len(atoms)):
            partners = adj[i0]
            if not partners:
                continue
            i_ser = i0 + 1
            p_ser = [p + 1 for p in partners]
            for k in range(0, len(p_ser), 4):
                chunk = p_ser[k:k+4]
                f.write("CONECT" + f"{i_ser:5d}" + "".join(f"{p:5d}" for p in chunk) + "\n")

        f.write("ENDMDL\nEND\n")

    return str(filename)



def suppress_pbc_crossing_bonds(atoms, bonds, atol=1e-10):
    """
    Return bonds that do NOT cross the periodic boundary, judged by whether
    the direct displacement is already the minimum-image displacement.

    atol: numerical tolerance on the displacement difference.
    """
    cell = atoms.cell.array
    pbc = atoms.pbc
    pos = atoms.get_positions()

    kept = []
    for i, j in bonds:
        d = pos[j] - pos[i]
        d_mic, _shift = find_mic(d, cell, pbc=pbc)
        if np.linalg.norm(d - d_mic) <= atol:
            kept.append((i, j))
    return kept

def view_atoms_with_bonds(atoms, molecules, ):
    # molecules.bonds is a structured array with field 'atoms'
    bonds = [tuple(map(int, pair)) for pair in molecules.bonds["atoms"]]
    # or as an ndarray:
    # bonds = molecules.bonds["atoms"].astype(int)
    pdb_path = write_pdb_conect(atoms, suppress_pbc_crossing_bonds(atoms, bonds), "network.pdb")
    # pdb_path = write_pdb_conect(atoms, bonds, "network.pdb")

    view = nv.show_file(pdb_path)
    return view

def show_atoms_with_bonds(atoms, molecules):
    view = view_atoms_with_bonds(atoms, molecules)
    view.clear_representations()
    view.add_representation("ball+stick", radiusScale=10)
    return view