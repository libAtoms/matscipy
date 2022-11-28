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
#define PY_ARRAY_UNIQUE_SYMBOL MATSCIPY_ARRAY_API
#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_5_API_VERSION
#include <numpy/arrayobject.h>

#include <limits.h>
#include <float.h> 
#include <stdbool.h>
#include <stddef.h>

#include "tools.h"

/*
 * Some cell index algebra
 */

/* Map i back to the interval [0,n) by shifting by integer multiples of n */
int
bin_wrap(int i, int n) {
    while (i < 0)  i += n;
    while (i >= n)  i -= n;
    return i;
}

/* Map i back to the interval [0,n) by assigning edge value if outside
   interval */
int
bin_trunc(int i, int n)
{
    if (i < 0)  i = 0;
    else if (i >= n)  i = n-1;
    return i;
}

/* Map particle position to a cell index */
void
position_to_cell_index(double *cell_origin, double *inv_cell, double *ri,
                       int n1, int n2, int n3, int *c1, int *c2, int *c3)
{
    int i;
    double dri[3], si[3];
    for (i = 0; i < 3; i++) {
        dri[i] = ri[i] - cell_origin[i];
    }
    mat_mul_vec(inv_cell, dri, si);
    *c1 = floor(si[0]*n1);
    *c2 = floor(si[1]*n2);
    *c3 = floor(si[2]*n3);
}

/*
 * Helper functions
 */

bool
check_bound(int c, int n)
{
    if (c < 0 || c >= n) {
        PyErr_SetString(PyExc_RuntimeError, "Atom out of (non-periodic) "
                        "simulation bounds.");
        return false;
    }
    return true;
}

/*
 * Neighbour list construction
 */

PyObject *
py_neighbour_list(PyObject *self, PyObject *args)
{
    PyObject *py_cell_origin, *py_cell, *py_inv_cell, *py_pbc, *py_r;
    PyObject *py_quantities, *py_cutoffs, *py_types = NULL;
    double cutoff;

    /* Neighbour list */
    int *seed = NULL, *last, *next = NULL;

    /* Optional quantities to be computed */
    PyObject *py_first = NULL, *py_secnd = NULL, *py_distvec = NULL;
    PyObject *py_absdist = NULL, *py_shift = NULL;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTuple(args, "O!OOOOOO|O", &PyUnicode_Type, &py_quantities,
                          &py_cell_origin, &py_cell, &py_inv_cell, &py_pbc,
                          &py_r, &py_cutoffs, &py_types))
#else
    if (!PyArg_ParseTuple(args, "O!OOOOOO|O", &PyString_Type, &py_quantities,
                          &py_cell_origin, &py_cell, &py_inv_cell, &py_pbc,
                          &py_r, &py_cutoffs, &py_types))
#endif
        return NULL;

    /* Make sure our arrays are contiguous */
    py_cell_origin = PyArray_FROMANY(py_cell_origin, NPY_DOUBLE, 1, 1,
                                     NPY_C_CONTIGUOUS);
    if (!py_cell_origin) return NULL;
    py_cell = PyArray_FROMANY(py_cell, NPY_DOUBLE, 2, 2,
                              NPY_C_CONTIGUOUS);
    if (!py_cell) return NULL;
    py_inv_cell = PyArray_FROMANY(py_inv_cell, NPY_DOUBLE, 2, 2,
                                  NPY_C_CONTIGUOUS);
    if (!py_inv_cell) return NULL;
    py_pbc = PyArray_FROMANY(py_pbc, NPY_BOOL, 1, 1, NPY_C_CONTIGUOUS);
    if (!py_pbc) return NULL;
    py_r = PyArray_FROMANY(py_r, NPY_DOUBLE, 2, 2, NPY_C_CONTIGUOUS);
    if (!py_r) return NULL;
    if (py_types) {
        py_types = PyArray_FROMANY(py_types, NPY_INT, 1, 1, NPY_C_CONTIGUOUS);
        if (!py_types) return NULL;
    }

    /* FIXME! Check array shapes. */
    npy_intp nat = PyArray_DIM((PyArrayObject *) py_r, 0);

    npy_intp ncutoffdims = 0;
    npy_intp ncutoffs = 1;
    npy_double *cutoffs = NULL;
    double *cutoffs_sq = NULL;
    if (PyFloat_Check(py_cutoffs)) {
        cutoff = PyFloat_AsDouble(py_cutoffs);
        py_cutoffs = NULL;
    }
    else {
        int i;

        /* This must be an array of cutoffs */
        py_cutoffs = PyArray_FROMANY(py_cutoffs, NPY_DOUBLE, 1, 2,
                                     NPY_C_CONTIGUOUS);
        if (!py_cutoffs) return NULL;
        ncutoffdims = PyArray_NDIM((PyArrayObject *) py_cutoffs);
        ncutoffs = PyArray_DIM((PyArrayObject *) py_cutoffs, 0);
        cutoffs = PyArray_DATA((PyArrayObject *) py_cutoffs);
        cutoff = 0.0;
        if (ncutoffdims == 1) {
            if (ncutoffs != nat) {
                PyErr_SetString(PyExc_TypeError, "One-dimensional cutoff array "
                                "must have length that corresponds to position "
                                "array.");
                goto fail;
            }
            for (i = 0; i < nat; i++) {
                cutoff = max(cutoff, 2*cutoffs[i]);
            }
        }
        else {
            if (PyArray_DIM((PyArrayObject *) py_cutoffs, 1) != ncutoffs) {
                PyErr_SetString(PyExc_TypeError, "Two-dimensional cutoff array "
                                "must be square.");
                goto fail;
            }
            cutoffs_sq = malloc(ncutoffs*ncutoffs*sizeof(double));
            if (!cutoffs_sq) {
                PyErr_SetString(PyExc_RuntimeError, "Failed to allocate "
                                                    "cutoffs_sq array.");
                goto fail;
            }
            for (i = 0; i < ncutoffs*ncutoffs; i++) {
                cutoff = max(cutoff, cutoffs[i]);
                cutoffs_sq[i] = cutoffs[i]*cutoffs[i];
            }
        }
    }

    if (py_types && PyArray_DIM((PyArrayObject *) py_types, 0) != nat) {
       PyErr_SetString(PyExc_TypeError, "Position and type arrays must have "
                                        "identical first dimension.");
       goto fail;
    }

    /* Get pointers to array data */
    npy_double *cell_origin = PyArray_DATA((PyArrayObject *) py_cell_origin);
    npy_double *cell = PyArray_DATA((PyArrayObject *) py_cell);
    npy_double *cell1 = &cell[0], *cell2 = &cell[3], *cell3 = &cell[6];
    npy_double *inv_cell = PyArray_DATA((PyArrayObject *) py_inv_cell);
    npy_bool *pbc = PyArray_DATA((PyArrayObject *) py_pbc);
    npy_double *r = PyArray_DATA((PyArrayObject *) py_r);
    npy_int *types = NULL;
    if (py_types) types = PyArray_DATA((PyArrayObject *) py_types);

    /* Compute vectors to opposite face */
    double norm1[3], norm2[3], norm3[3];
    cross_product(cell2, cell3, norm1);
    cross_product(cell3, cell1, norm2);
    cross_product(cell1, cell2, norm3);
    double volume = fabs(cell3[0]*norm3[0] + cell3[1]*norm3[1] + cell3[2]*norm3[2]);

    if (volume < 1e-12) {
        PyErr_SetString(PyExc_RuntimeError, "Zero cell volume.");
        goto fail;
    }

    double len1 = normsq(norm1), len2 = normsq(norm2), len3 = normsq(norm3);
    int i;
    for (i = 0; i < 3; i++) {
        norm1[i] *= volume/(len1*len1);
        norm2[i] *= volume/(len2*len2);
        norm3[i] *= volume/(len3*len3);
    }

    /* Compute distance of cell faces */
    len1 = volume/len1;
    len2 = volume/len2;
    len3 = volume/len3;

    /* Number of cells for cell subdivision */
    int n1 = max((int) floor(len1/cutoff), 1);
    int n2 = max((int) floor(len2/cutoff), 1);
    int n3 = max((int) floor(len3/cutoff), 1);

    /* Avoid overflow in total number of cells */
    bool warned = false;
    while (((double)n1)*n2*n3 > INT_MAX) {
      if (!warned) {
        PyErr_WarnEx(NULL, "Ratio of simulation cell size to cutoff is very "
                     "large; reducing number of bins for neighbour list "
                     "search, but this may be slow. Are you using a cell with "
                     "lots of vacuum?", 1);
        warned = true;
      }
      n1 /= 2; if (n1 <= 0) n1 = 1;
      n2 /= 2; if (n2 <= 0) n2 = 1;
      n3 /= 2; if (n3 <= 0) n3 = 1;
    }

    assert(n1 > 0);
    assert(n2 > 0);
    assert(n3 > 0);

    /* Find out over how many neighbor cells we need to loop (if the box is
       small */
    int nx = (int) ceil(cutoff*n1/len1);
    int ny = (int) ceil(cutoff*n2/len2);
    int nz = (int) ceil(cutoff*n3/len3);

    /* Sort particles into bins */
    int ncells = n1*n2*n3;
    seed = (int *) malloc(ncells*sizeof(int));
    if (!seed) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate seed array.");
        goto fail;
    }
    last = (int *) malloc(ncells*sizeof(int));
    if (!last) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate last array.");
        goto fail;
    }
    for (i = 0; i < ncells; i++)  seed[i] = -1;
    next = (int *) malloc(nat*sizeof(int));
    if (!next) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate next array.");
        goto fail;
    }
    for (i = 0; i < nat; i++) {
        /* Get cell index */
        int c1, c2, c3;
        position_to_cell_index(cell_origin, inv_cell, &r[3*i], n1, n2, n3,
                               &c1, &c2, &c3);

        /* Periodic/non-periodic boundary conditions */
        if (pbc[0])  c1 = bin_wrap(c1, n1);  else  c1 = bin_trunc(c1, n1);
        if (pbc[1])  c2 = bin_wrap(c2, n2);  else  c2 = bin_trunc(c2, n2);
        if (pbc[2])  c3 = bin_wrap(c3, n3);  else  c3 = bin_trunc(c3, n3);

        /* Continuous cell index */
        int ci = c1+n1*(c2+n2*c3);

        assert(c1 >= 0 && c1 < n1);
        assert(c2 >= 0 && c2 < n2);
        assert(c3 >= 0 && c3 < n3);
        assert(ci >= 0 && ci < ncells);

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

    npy_int *first = NULL, *secnd = NULL, *shift = NULL;
    npy_double *distvec = NULL, *absdist = NULL;

#if PY_MAJOR_VERSION >= 3
    PyObject *py_bquantities = PyUnicode_AsASCIIString(py_quantities);
    if (!py_bquantities) {
        PyErr_SetString(PyExc_TypeError, "Conversion to ASCII string failed.");
        goto fail;
    }
    char *quantities = PyBytes_AS_STRING(py_bquantities);
#else
    char *quantities = PyString_AsString(py_quantities);
    if (!quantities) {
        PyErr_SetString(PyExc_TypeError, "Conversion to string failed.");
        goto fail;
    }
#endif
    i = 0;
    npy_intp dims[2] = { neighsize, 3 };
    while (quantities[i] != '\0') {
        switch (quantities[i]) {
        case 'i':
            py_first = PyArray_ZEROS(1, dims, NPY_INT, 0);
            first = PyArray_DATA((PyArrayObject *) py_first);
            break;
        case 'j':
            py_secnd = PyArray_ZEROS(1, dims, NPY_INT, 0);
            secnd = PyArray_DATA((PyArrayObject *) py_secnd);
            break;
        case 'D':
            py_distvec = PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
            distvec = PyArray_DATA((PyArrayObject *) py_distvec);
            break;
        case 'd':
            py_absdist = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
            absdist = PyArray_DATA((PyArrayObject *) py_absdist);
            break;
        case 'S':
            py_shift = PyArray_ZEROS(2, dims, NPY_INT, 0);
            shift = PyArray_DATA((PyArrayObject *) py_shift);
            break;
        default:
            PyErr_SetString(PyExc_ValueError,
                            "Unsupported quantity specified.");
            goto fail;
        }
        i++;
    }

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

        int ci01, ci02, ci03;
        position_to_cell_index(cell_origin, inv_cell, ri, n1, n2, n3,
                               &ci01, &ci02, &ci03);

        /* Truncate if non-periodic and outside of simulation domain */
        int ci1, ci2, ci3;
        if (!pbc[0])  ci1 = bin_trunc(ci01, n1);  else  ci1 = ci01;
        if (!pbc[1])  ci2 = bin_trunc(ci02, n2);  else  ci2 = ci02;
        if (!pbc[2])  ci3 = bin_trunc(ci03, n3);  else  ci3 = ci03;

        /* dri is the position relative to the lower left corner of the bin */
        double dri[3];
        dri[0] = ri[0] - ci1*bin1[0] - ci2*bin2[0] - ci3*bin3[0];
        dri[1] = ri[1] - ci1*bin1[1] - ci2*bin2[1] - ci3*bin3[1];
        dri[2] = ri[2] - ci1*bin1[2] - ci2*bin2[2] - ci3*bin3[2];

        /* Apply periodic boundary conditions */
        if (pbc[0])  ci1 = bin_wrap(ci01, n1);  else  ci1 = bin_trunc(ci01, n1);
        if (pbc[1])  ci2 = bin_wrap(ci02, n2);  else  ci2 = bin_trunc(ci02, n2);
        if (pbc[2])  ci3 = bin_wrap(ci03, n3);  else  ci3 = bin_trunc(ci03, n3);

        /* Loop over neighbouring bins */
        int x, y, z;
        for (z = -nz; z <= nz; z++) {
            int cj3 = ci3 + z;
            if (pbc[2])  cj3 = bin_wrap(cj3, n3);

            /* Skip to next z value if cell is out of simulation bounds */
            if (cj3 < 0 || cj3 >= n3)  continue;

            cj3 = bin_trunc(cj3, n3);
            int ncj3 = n2*cj3;

            double off3[3];
            off3[0] = z*bin3[0];
            off3[1] = z*bin3[1];
            off3[2] = z*bin3[2];

            for (y = -ny; y <= ny; y++) {
                int cj2 = ci2 + y;
                if (pbc[1])  cj2 = bin_wrap(cj2, n2);

                /* Skip to next y value if cell is out of simulation bounds */
                if (cj2 < 0 || cj2 >= n2)  continue;

                cj2 = bin_trunc(cj2, n2);
                int ncj2 = n1*(cj2 + ncj3);

                double off2[3];
                off2[0] = off3[0] + y*bin2[0];
                off2[1] = off3[1] + y*bin2[1];
                off2[2] = off3[2] + y*bin2[2];

                for (x = -nx; x <= nx; x++) {
                    /* Bin index of neighbouring bin */
                    int cj1 = ci1 + x;
                    if (pbc[0])  cj1 = bin_wrap(cj1, n1);

                    /* Skip to next x value if cell is out of simulation bounds
                     */
                    if (cj1 < 0 || cj1 >= n1)  continue;

                    cj1 = bin_trunc(cj1, n1);
                    int ncj = cj1 + ncj2;

                    assert(ncj == cj1+n1*(cj2+n2*cj3));

                    /* Offset of the neighboring bins */
                    double off[3];
                    off[0] = off2[0] + x*bin1[0];
                    off[1] = off2[1] + x*bin1[1];
                    off[2] = off2[2] + x*bin1[2];

                    /* Loop over all atoms in neighbouring bin */
                    int j = seed[ncj];
                    while (j >= 0) {
                        if (i != j || x != 0 || y != 0 || z != 0) {
                            double *rj = &r[3*j];

                            int cj1, cj2, cj3;
                            position_to_cell_index(cell_origin, inv_cell, rj,
                                                   n1, n2, n3,
                                                   &cj1, &cj2, &cj3);

                            /* Truncate if non-periodic and outside of
                               simulation domain. */
                            if (!pbc[0])  cj1 = bin_trunc(cj1, n1);
                            if (!pbc[1])  cj2 = bin_trunc(cj2, n2);
                            if (!pbc[2])  cj3 = bin_trunc(cj3, n3);

                            /* drj is position relative to lower
                               left corner of the bin */
                            double drj[3];
                            drj[0] = rj[0] - cj1*bin1[0] - cj2*bin2[0] -
                                cj3*bin3[0];
                            drj[1] = rj[1] - cj1*bin1[1] - cj2*bin2[1] -
                                cj3*bin3[1];
                            drj[2] = rj[2] - cj1*bin1[2] - cj2*bin2[2] -
                                cj3*bin3[2];

                            /* Compute distance between atoms */
                            double dr[3];
                            dr[0] = drj[0] - dri[0] + off[0];
                            dr[1] = drj[1] - dri[1] + off[1];
                            dr[2] = drj[2] - dri[2] + off[2];
                            double abs_dr_sq = dr[0]*dr[0] + dr[1]*dr[1] +
                                dr[2]*dr[2];

                            if (abs_dr_sq < cutoff_sq) {
                                bool inside_cutoff = true;
                                if (ncutoffdims == 1) {
                                    double c_sq = cutoffs[i]+cutoffs[j];
                                    c_sq *= c_sq;
                                    inside_cutoff = abs_dr_sq < c_sq;
                                }
                                else if (ncutoffdims == 2 && types) {
                                    if (types[i] < ncutoffs &&
                                        types[j] < ncutoffs){
                                        double c_sq =
                                            cutoffs_sq[types[i]*ncutoffs+
                                                       types[j]];
                                        inside_cutoff = abs_dr_sq < c_sq;
                                    }
                                }

                                if (inside_cutoff) {

                                    if (nneigh >= neighsize) {
                                        neighsize *= 2;

                                        if (py_first &&
                                            !(first = resize_array(py_first,
                                                                   neighsize)))
                                            goto fail;
                                        if (py_secnd &&
                                            !(secnd = resize_array(py_secnd,
                                                                   neighsize)))
                                            goto fail;
                                        if (py_distvec &&
                                            !(distvec = resize_array(
                                                py_distvec, neighsize)))
                                            goto fail;
                                        if (py_absdist &&
                                            !(absdist = resize_array(
                                                py_absdist, neighsize)))
                                            goto fail;
                                        if (py_shift &&
                                            !(shift = resize_array(py_shift,
                                                                   neighsize)))
                                            goto fail;
                                    }

                                    if (py_first)
                                        first[nneigh] = i;
                                    if (py_secnd)
                                        secnd[nneigh] = j;
                                    if (py_distvec) {
                                        distvec[3*nneigh+0] = dr[0];
                                        distvec[3*nneigh+1] = dr[1];
                                        distvec[3*nneigh+2] = dr[2];
                                    }
                                    if (py_absdist)
                                        absdist[nneigh] = sqrt(abs_dr_sq);
                                    if (py_shift) {
                                        shift[3*nneigh+0] = (ci01 - cj1 + x)/n1;
                                        shift[3*nneigh+1] = (ci02 - cj2 + y)/n2;
                                        shift[3*nneigh+2] = (ci03 - cj3 + z)/n3;
                                    }

                                    nneigh++;
                                }

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
    if (cutoffs_sq)  free(cutoffs_sq);

    seed = NULL;
    next = NULL;
    cutoffs_sq = NULL;

    /* Resize arrays to actual size of neighbour list */
    if (py_first && !resize_array(py_first, nneigh))  goto fail;
    if (py_secnd && !resize_array(py_secnd, nneigh))  goto fail;
    if (py_distvec && !resize_array(py_distvec, nneigh))  goto fail;
    if (py_absdist && !resize_array(py_absdist, nneigh))  goto fail;
    if (py_shift && !resize_array(py_shift, nneigh))  goto fail;

    /* Build return tuple */
    PyObject *py_ret = PyTuple_New(strlen(quantities));
    i = 0;
    while (quantities[i] != '\0') {
        switch (quantities[i]) {
        case 'i':
            PyTuple_SetItem(py_ret, i, py_first);
            break;
        case 'j':
            PyTuple_SetItem(py_ret, i, py_secnd);
            break;
        case 'D':
            PyTuple_SetItem(py_ret, i, py_distvec);
            break;
        case 'd':
            PyTuple_SetItem(py_ret, i, py_absdist);
            break;
        case 'S':
            PyTuple_SetItem(py_ret, i, py_shift);
            break;
        default:
            PyErr_SetString(PyExc_ValueError,
                            "Unsupported quantity specified.");
            goto fail;
        }
        i++;
    }
    if (strlen(quantities) == 1) {
        PyObject *py_tuple = py_ret;
        py_ret = PyTuple_GET_ITEM(py_tuple, 0);
        Py_INCREF(py_ret);
        Py_DECREF(py_tuple);
    }

    /* Final cleanup */
#if PY_MAJOR_VERSION >= 3
    Py_DECREF(py_bquantities);
#endif
    Py_XDECREF(py_cutoffs);
    Py_DECREF(py_cell_origin);
    Py_DECREF(py_cell);
    Py_DECREF(py_inv_cell);
    Py_DECREF(py_pbc);
    Py_DECREF(py_r);
    Py_XDECREF(py_types);

    return py_ret;

    fail:
    /* Cleanup. Sorry for the goto. */
#if PY_MAJOR_VERSION >= 3
    Py_XDECREF(py_bquantities);
#endif
    Py_XDECREF(py_cutoffs);
    Py_XDECREF(py_cell_origin);
    Py_XDECREF(py_cell);
    Py_XDECREF(py_inv_cell);
    Py_XDECREF(py_pbc);
    Py_XDECREF(py_r);
    Py_DECREF(py_types);

    if (seed)  free(seed);
    if (next)  free(next);
    if (cutoffs_sq)  free(cutoffs_sq);
    Py_XDECREF(py_first);
    Py_XDECREF(py_secnd);
    Py_XDECREF(py_distvec);
    Py_XDECREF(py_absdist);
    Py_XDECREF(py_shift);
    return NULL;
}

/*
 * Construct seed array that points to start of rows: O(n)
 */
void
first_neighbours(int n, int nn, npy_int *i_n, npy_int *seed)
{
    int k;

    for (k = 0; k < n; k++) {
        seed[k] = -1;
    }

    seed[i_n[0]] = 0;

    for (k = 1; k < nn; k++) {
        if (i_n[k] != i_n[k-1]) {
            int l;
            for (l = i_n[k-1]+1; l <= i_n[k]; l++) {
                seed[l] = k;
            }
        }
    }
    // seed[n] = nn;
    for (k = i_n[nn-1]+1; k <= n; k++) {
        seed[k] = nn;
    }
}


/*
 * Python wrapper for seed array calculation
 */

PyObject *
py_first_neighbours(PyObject *self, PyObject *args)
{
    npy_int n;
    PyObject *py_i;

    if (!PyArg_ParseTuple(args, "iO", &n, &py_i))
        return NULL;

    /* Make sure our arrays are contiguous */
    py_i = PyArray_FROMANY(py_i, NPY_INT, 1, 1, NPY_C_CONTIGUOUS);
    if (!py_i) return NULL;

    /* Neighbour list size */
    npy_intp nn = PyArray_DIM((PyArrayObject *) py_i, 0);

    /* Create seed array of length n */
    npy_intp n1 = n+1;
    PyObject *py_seed = PyArray_ZEROS(1, &n1, NPY_INT, 0);

    /* Construct seed array */
    first_neighbours(n, nn, PyArray_DATA(py_i), PyArray_DATA(py_seed));

    return py_seed;
}

/*
 * Python wrapper for triplet list calculation
 */

PyObject *
py_triplet_list(PyObject *self, PyObject *args)
{
    /* parse python args */
    PyObject *py_fi, *py_absdist = NULL, *py_cutoff = NULL;
    npy_double *absdist = NULL;

    if (!PyArg_ParseTuple(args, "O|OO", &py_fi, &py_absdist, &py_cutoff)) {
        return NULL;
    }


    npy_int *fi = NULL, *ij_t = NULL, *ik_t = NULL, *jk_t = NULL;

    py_fi = PyArray_FROMANY(py_fi, NPY_INT, 1, 1, NPY_C_CONTIGUOUS);
    fi = PyArray_DATA((PyArrayObject *) py_fi);

    if (!fi) return NULL;

    double cutoff = DBL_MAX;

    if (py_cutoff || py_absdist) {
        if (!py_absdist || !py_cutoff) {
    	    PyErr_SetString(PyExc_TypeError, "Cutoff and distances must "
                                                    "be specified together.");
    	    return NULL;
    	}
        py_absdist = PyArray_FROMANY(py_absdist, NPY_DOUBLE,
				         1, 1, NPY_C_CONTIGUOUS);
       	if (!py_absdist) {
            PyErr_SetString(PyExc_TypeError, "Distances must be an "
                                             "array of floats.");
            return NULL;
	    }
    	absdist = PyArray_DATA((PyArrayObject *) py_absdist);
        // absdist_length = (int) PyArray_SIZE(py_absdist);
    	if (PyFloat_Check(py_cutoff)) {
        	cutoff = PyFloat_AsDouble(py_cutoff);
        	py_cutoff = NULL;
   	    }
    	else {
            PyErr_SetString(PyExc_NotImplementedError, "Cutoff must be a single "
                                             "float.");
            return NULL;
    	}
    }

    /* guess initial triplet list size */
    npy_intp dim = (int) PyArray_SIZE(py_fi);
    dim *= 2;

    /* initialize triplet lists */
    PyObject *py_ij_t = PyArray_ZEROS(1, &dim, NPY_INT, 0);
    ij_t = PyArray_DATA((PyArrayObject *) py_ij_t);
    PyObject *py_ik_t = PyArray_ZEROS(1, &dim, NPY_INT, 0);
    ik_t = PyArray_DATA((PyArrayObject *) py_ik_t);

    int init_length = (int) PyArray_SIZE(py_fi);

    /* compute the triplet list */
    int index_trip = 0;
    for (int r = 0; r < (init_length - 1); r++) {
        for (int ij= fi[r]; ij < fi[r+1]; ij++) {
            for (int ik = fi[r]; ik < fi[r+1]; ik++) {
                /* resize array if necessary */
                int length_trip = (int) PyArray_SIZE(py_ij_t);
                if (index_trip >= length_trip) {
                    length_trip *= 2;
                    if (py_ij_t && !(ij_t = resize_array(py_ij_t, length_trip)))
                        goto fail;
                    if (py_ik_t && !(ik_t = resize_array(py_ik_t, length_trip)))
                        goto fail;
                }
        		if ((ij != ik)) {
        		    if (absdist) { 
            		    if ((absdist[ij] >= cutoff) || (absdist[ik] >= cutoff)) {
        			        continue; 
        			    };
        		    }
               	    ij_t[index_trip] = ij;
                    ik_t[index_trip++] = ik;
        	    }
            }
        }
    }

    /* set final array sizes of the triplet lists */
    if (py_ij_t && !(ij_t = resize_array(py_ij_t, index_trip))) goto fail;
    if (py_ik_t && !(ik_t = resize_array(py_ik_t, index_trip))) goto fail;

    npy_intp d1 = (int) PyArray_SIZE(py_ij_t);
    PyObject *py_jk_t = PyArray_ZEROS(1, &d1, NPY_INT, 0);
    jk_t = PyArray_DATA((PyArrayObject *) py_jk_t);
    index_trip++;

    // TODO: ask Lars and/or use an ordered j_n list
    /*for (int t = 0; t < index_trip; t++) {
    	int ij = ij_t[t];
	int ik = ik_t[t];
	continue;
    } */

    /* create return tuple */
    PyObject *py_ret = PyTuple_New(2);
    PyTuple_SetItem(py_ret, 0, py_ij_t);
    PyTuple_SetItem(py_ret, 1, py_ik_t);

    return py_ret;

    fail:
    /* Cleanup */
    if (py_fi)  Py_DECREF(py_fi);
    if (py_cutoff)  Py_DECREF(py_cutoff);
    if (py_ij_t)  Py_DECREF(py_ij_t);
    if (py_ik_t)  Py_DECREF(py_ik_t);
    return NULL;
}

/*
 *  construct array that points to the index jumps of a continous, sorted array
 *  starting with 0 at index 0
 */

PyObject*
py_get_jump_indicies(PyObject *self, PyObject *args)
{
    PyObject *py_sorted;

    if (!PyArg_ParseTuple(args, "O", &py_sorted))
        return NULL;

    /* Make sure our arrays are contiguous */
    py_sorted = PyArray_FROMANY(py_sorted, NPY_INT, 1, 1, NPY_C_CONTIGUOUS);
    if (!py_sorted) return NULL;

    /* sorted imput array size */
    int nn = (int) PyArray_SIZE(py_sorted);

    /* calculate number of jumps */
    npy_int *sorted = PyArray_DATA((PyArrayObject *) py_sorted);
    int n = 0;
    for (int i = 0; i < nn-1; i++) {
            if (sorted[i] != sorted[i+1]) {
            n++;
        }
    }
    n++;

    /* Create seed array of length n */
    npy_intp n1 = n+1;
    PyObject *py_seed = PyArray_ZEROS(1, &n1, NPY_INT, 0);

    /* Construct seed array */
    first_neighbours(n, nn, sorted, PyArray_DATA(py_seed));

    return py_seed;
}
