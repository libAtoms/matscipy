/* ======================================================================
   matscipy - Python materials science tools
   https://github.com/pastewka/atomistica

   https://github.com/libAtoms/matscipy

   Copyright (2014) James Kermode, King's College London
                    Lars Pastewka, Karlsruhe Institute of Technology

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 2 of the License, or
   (at your option) any later version.
  
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
  
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
   ====================================================================== */

#ifndef __NEIGHBOURS_H_
#define __NEIGHBOURS_H_

#include <Python.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Neighbour list construction
 */
PyObject *py_neighbour_list(PyObject *self, PyObject *args);
PyObject *py_first_neighbours(PyObject *self, PyObject *args);
PyObject *py_triplet_list(PyObject *self, PyObject *args);
PyObject *py_get_jump_indicies(PyObject *self, PyObject *args);

/*
 * Construct seed array that points to start of rows
 */
void first_neighbours(int n, int nn, npy_int *i_n, npy_int *seed);

#ifdef __cplusplus
}
#endif

#endif
