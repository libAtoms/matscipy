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

#ifndef __TOOLS_H_
#define __TOOLS_H_

#include <Python.h>
#include <numpy/arrayobject.h>

#ifdef __cplusplus
extern "C" {
#else

/*
 * Basics
 */

#define max(x, y)  ( x > y ? x : y )

#endif

/*
 * Some basic linear algebra
 */

void cross_product(double *a, double *b, double *c);
void mat_mul_vec(double *mat, double *vin, double *vout);
double normsq(double *a);

/*
 * Helper functions
 */

void *resize_array(PyObject *py_arr, npy_intp newsize);

#ifdef __cplusplus
}
#endif

#endif
