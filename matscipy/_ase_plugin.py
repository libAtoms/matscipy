"""ASE v4 neighbour-list plugin for matscipy.

Registers matscipy's compiled neighbour list as an ``ase.plugins`` backend, so
it can be selected *explicitly* via :func:`ase.neighborlist.get_neighbor_list`
(or, in due course, a calculator's ``neighbor_list=`` option).  Selection is
never automatic -- this only makes the backend *available* under the name
``"matscipy"``.

This module is the ``ase.plugins`` entry point (it exposes ``__ase_plugins__``).
The plugin registration is guarded so that installing matscipy alongside an ASE
that predates the v4 plugin API registers nothing rather than breaking plugin
discovery.  The heavy import of :mod:`matscipy.neighbours` happens lazily,
inside the adapter, so merely building the plugin collection does not import the
compiled extension.
"""
from __future__ import annotations


def neighbor_list(quantities, atoms, cutoff, *, self_interaction=False):
    """matscipy neighbour list adapted to ASE's ``NeighborListFunction``.

    matscipy's ``neighbour_list`` returns the same ``(i, j, d, D, S)`` flat
    arrays and quantity letters as ``ase.neighborlist.neighbor_list``, but has
    no ``self_interaction`` option (it never returns pure self-pairs).  Per the
    plugin contract, a request it cannot honour is rejected rather than
    silently differing, so ``self_interaction=True`` raises.
    """
    if self_interaction:
        raise NotImplementedError(
            'the matscipy neighbour-list backend does not support '
            'self_interaction=True')
    from matscipy.neighbours import neighbour_list as _matscipy_neighbour_list

    return _matscipy_neighbour_list(quantities, atoms, cutoff)


try:
    from ase._4.plugins.neighborlist import NeighborListPlugin
except ImportError:
    # ASE without the v4 plugin API: register nothing, but do not break the
    # discovery of other plugins.
    __ase_plugins__: set = set()
else:
    __ase_plugins__ = {
        NeighborListPlugin(
            'matscipy',
            long_name='matscipy compiled neighbour list',
            citation='https://github.com/libAtoms/matscipy',
            implementation='matscipy._ase_plugin.neighbor_list',
        ),
    }
