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

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL MATSCIPY_ARRAY_API
#include <numpy/arrayobject.h>

#include <stddef.h>

#include "matscipymodule.h"

/*
 * Some basic linear algebra
 */

void
cross_product(double *a, double *b, double *c)
{
    c[0] = a[1]*b[2]-a[2]*b[1];
    c[1] = a[2]*b[0]-a[0]*b[2];
    c[2] = a[0]*b[1]-a[1]*b[0];
}

void
mat_mul_vec(double *mat, double *vin, double *vout)
{
    int i, j;
    for (i = 0; i < 3; i++, vout++){
        *vout = 0.0;
        for (j = 0; j < 3; j++, mat++) {
            *vout += (*mat)*vin[j];
        }
    }
}

double
normsq(double *a)
{
    return sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
}

/*
 * Some cell index algebra
 */

int
wrap(int i, int n)
{
    while (i < 0)  i += n;
    while (i >= n)  i -= n;
    return i;
}

void
position_to_cell_index(double *inv_cell, double *ri, int n1, int n2, int n3,
                       int *c1, int *c2, int *c3)
{
    double si[3];
    mat_mul_vec(inv_cell, ri, si);
    *c1 = floor(si[0]*n1);
    *c2 = floor(si[1]*n2);
    *c3 = floor(si[2]*n3);
}

/*
 * Helper functions
 */

int
ensure_neighbor_list_size(int minsize, npy_intp *neighsize, PyObject *py_first,
                          npy_int **first, PyObject *py_secnd,
                          npy_int **secnd)
{
    if (minsize > *neighsize) {
        /* Resize first and secnd arrays */
        *neighsize *= 2;
                                
        PyArray_Dims newshape;
        newshape.ptr = neighsize;
        newshape.len = 1;
        
        PyObject *retval;
        retval = PyArray_Resize((PyArrayObject *)
                                py_first,
                                &newshape, 1,
                                NPY_CORDER);
        if (!retval)  return 1;
        Py_DECREF(retval);
        *first = PyArray_DATA((PyArrayObject *)
                              py_first);
        
        retval = PyArray_Resize((PyArrayObject *)
                                py_secnd,
                                &newshape, 1,
                                NPY_CORDER);
        if (!retval)  return 1;
        Py_DECREF(retval);
        *secnd = PyArray_DATA((PyArrayObject *)
                              py_secnd);
    }

    return 0;
}

/*
 * Neighbour list construction
 */

PyObject *
py_neighbour_list(PyObject *self, PyObject *args)
{
    PyObject *py_cell, *py_inv_cell, *py_r;
    double cutoff;

    if (!PyArg_ParseTuple(args, "OOOd", &py_cell, &py_inv_cell, &py_r,
                          &cutoff))
        return NULL;

    /* Make sure our arrays are contiguous */
    py_cell = PyArray_FROMANY(py_cell, NPY_DOUBLE, 2, 2,
                              NPY_ARRAY_C_CONTIGUOUS);
    if (!py_cell) return NULL;
    py_inv_cell = PyArray_FROMANY(py_inv_cell, NPY_DOUBLE, 2, 2,
                                  NPY_ARRAY_C_CONTIGUOUS);
    if (!py_inv_cell) return NULL;
    py_r = PyArray_FROMANY(py_r, NPY_DOUBLE, 2, 2, NPY_ARRAY_C_CONTIGUOUS);
    if (!py_r) return NULL;

    /* FIXME! Check array shapes. */
    npy_intp nat = PyArray_DIM((PyArrayObject *) py_r, 0);

    /* Get pointers to array data */
    npy_double *cell = PyArray_DATA((PyArrayObject *) py_cell);
    npy_double *cell1 = &cell[0], *cell2 = &cell[3], *cell3 = &cell[6];
    npy_double *inv_cell = PyArray_DATA((PyArrayObject *) py_inv_cell);
    npy_double *r = PyArray_DATA((PyArrayObject *) py_r);

    /* Compute vectors to opposite face */
    double norm1[3], norm2[3], norm3[3];
    cross_product(cell2, cell3, norm1);
    cross_product(cell3, cell1, norm2);
    cross_product(cell1, cell2, norm3);
    double volume = cell1[0]*norm3[0] + cell2[1]*norm3[1] + cell3[2]*norm3[2];
    double len1 = normsq(norm1), len2 = normsq(norm2), len3 = normsq(norm3);
    int i;
    for (i = 0; i < 3; i++) {
        norm1[i] *= volume/(len1*len1);
        norm2[i] *= volume/(len2*len2);
        norm3[i] *= volume/(len3*len3);
    }

    /* Number of cells for cell subdivision */
    int n1 = (int)(normsq(norm1)/cutoff)+1;
    int n2 = (int)(normsq(norm2)/cutoff)+1;
    int n3 = (int)(normsq(norm3)/cutoff)+1;

    /* Sort particles into bins */
    int *seed, *last, *next;
    int ncells = n1*n2*n3;
    seed = (int *) malloc(ncells*sizeof(int));
    last = (int *) malloc(ncells*sizeof(int));
    for (i = 0; i < ncells; i++)  seed[i] = -1;
    next = (int *) malloc(nat*sizeof(int));
    for (i = 0; i < nat; i++) {
        /* Get cell index */
        int c1, c2, c3;
        position_to_cell_index(inv_cell, &r[3*i], n1, n2, n3, &c1, &c2, &c3);

        /* Periodic boundary conditions */
        c1 = wrap(c1, n1);
        c2 = wrap(c2, n2);
        c3 = wrap(c3, n3);

        /* Continuous cell index */
        int ci = c1+n1*(c2+n2*c3);

        /* Put atom into appropriate bin */
        if (seed[ci] < 0) {
            next[i] = -1;
            seed[ci] = i;
            last[ci] = i;
        }
        else {
            next[i] = -1;
            next[last[ci]] = i;
            last[ci] = i;
        }
    }
    free(last);

    /* Neighbour list counter and size */
    npy_intp nneigh = 0; /* Number of neighbours found */
    npy_intp neighsize = nat; /* Initial guess for neighbour list size */

    /* Numpy array objects holding the neighbour list */
    PyObject *py_first = PyArray_ZEROS(1, &neighsize, NPY_INT, 1);
    PyObject *py_secnd = PyArray_ZEROS(1, &neighsize, NPY_INT, 1);
    npy_int *first = PyArray_DATA((PyArrayObject *) py_first);
    npy_int *secnd = PyArray_DATA((PyArrayObject *) py_secnd);

    /* We need the square of the cutoff */
    double cutoff_sq = cutoff*cutoff;

    /* We need the shape of the bin */
    double bin1[3], bin2[3], bin3[3];
    for (i = 0; i < 3; i++) {
        bin1[i] = cell1[i]/n1;
        bin2[i] = cell2[i]/n2;
        bin3[i] = cell3[i]/n3;
    }

    /* Loop over atoms */
    for (i = 0; i < nat; i++) {
        double *ri = &r[3*i];

        int ci1, ci2, ci3;
        position_to_cell_index(inv_cell, ri, n1, n2, n3, &ci1, &ci2, &ci3);

        /* dri is the position relative to the lower left corner of the bin */
        double dri[3];
        dri[0] = ri[0] - ci1*bin1[0] - ci2*bin2[0] - ci3*bin3[0];
        dri[1] = ri[1] - ci1*bin1[1] - ci2*bin2[1] - ci3*bin3[1];
        dri[2] = ri[2] - ci1*bin1[2] - ci2*bin2[2] - ci3*bin3[2];

        /* Apply periodic boundary conditions */
        ci1 = wrap(ci1, n1);
        ci2 = wrap(ci2, n2);
        ci3 = wrap(ci3, n3);

        /* Loop over neighbouring bins */
        int x, y, z;
        for (z = -1; z <= 1; z++) {
            int ncj3 = n2*wrap(ci3 + z, n3);

            double off3[3];
            off3[0] = z*bin3[0];
            off3[1] = z*bin3[1];
            off3[2] = z*bin3[2];

            for (y = -1; y <= 1; y++) {
                int ncj2 = n1*(wrap(ci2 + y, n2) + ncj3);

                double off2[3];
                off2[0] = off3[0] + y*bin2[0];
                off2[1] = off3[1] + y*bin2[1];
                off2[2] = off3[2] + y*bin2[2];

                for (x = -1; x <= 1; x++) {
                    /* Bin index of neighbouring bin */
                    int ncj = wrap(ci1 + x, n1) + ncj2;

                    /* Offset of the neighboring bins */
                    double off[3];
                    off[0] = off2[0] + z*bin3[0];
                    off[1] = off2[1] + z*bin3[1];
                    off[2] = off2[2] + z*bin3[2];

                    /* Loop over all atoms in neighbouring bin */
                    int j = seed[ncj];
                    while (j >= 0) {
                        if (i != j || x != 0 || y != 0 || z != 0) {
                            /* drj is position relative to lower left corner
                               of the bin */
                            double *rj = &r[3*j];

                            int cj1, cj2, cj3;
                            position_to_cell_index(inv_cell, rj, n1, n2, n3,
                                                   &cj1, &cj2, &cj3);

                            double drj[3];
                            drj[0] = rj[0] - cj1*bin1[0] - cj2*bin2[0] -
                                cj3*bin3[0];
                            drj[1] = rj[1] - cj1*bin1[1] - cj2*bin2[1] -
                                cj3*bin3[1];
                            drj[2] = rj[2] - cj1*bin1[2] - cj2*bin2[2] -
                                cj3*bin3[2];

                            /* Compute distance between atoms */
                            double dr[3];
                            dr[0] = dri[0] - drj[0] - off[0];
                            dr[1] = dri[1] - drj[1] - off[1];
                            dr[2] = dri[2] - drj[2] - off[2];
                            double abs_dr_sq = dr[0]*dr[0] + dr[1]*dr[1] + 
                                dr[2]*dr[2];

                            if (abs_dr_sq < cutoff_sq) {

                                ensure_neighbor_list_size(nneigh+1, &neighsize,
                                                          py_first, &first,
                                                          py_secnd, &secnd);

                                first[nneigh] = i;
                                secnd[nneigh] = j;

                                nneigh++;
                            }
                        }

                        j = next[j];
                    }
                }
            }
        }
    }

    /* Release cell subdivision information */
    free(seed);
    free(next);

    /* Resize arrays to actual size of neighbour list */
    PyArray_Dims dims;
    dims.ptr = &nneigh;
    dims.len = 1;

    PyObject *retval;
    retval = PyArray_Resize((PyArrayObject *) py_first, &dims, 1, NPY_CORDER);
    if (!retval)  return NULL;
    Py_DECREF(retval);
    retval = PyArray_Resize((PyArrayObject *) py_secnd, &dims, 1, NPY_CORDER);
    if (!retval)  return NULL;
    Py_DECREF(retval);

    /* Build return tuple */
    return Py_BuildValue("OO", py_first, py_secnd);
}

/*
 * Method declaration
 */

static PyMethodDef module_methods[] = {
    { "neighbour_list", py_neighbour_list, METH_VARARGS,
      "Compute a neighbour list for an atomic configuration." },
    { NULL, NULL, 0, NULL }  /* Sentinel */
};


/*
 * Module initialization
 */

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC
init_matscipy(void)
{
    PyObject* m;

    import_array();

    m = Py_InitModule3("_matscipy", module_methods,
                       "C support functions for matscipy.");
    if (m == NULL)
        return;
}
