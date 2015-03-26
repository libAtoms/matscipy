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
#include <numpy/arrayobject.h>

#include <algorithm>

#include "neighbours.h"
#include "tools.h"

const int MAX_WALKER = 4096;

/*
 * Tolerance for determining if a ring has closed (i.e. if the sum of all bond
 * vectors is zero).
 */
const double TOL = 0.0001;


/*
 * Look for the shortest distance starting at atom *root*,
 */
bool
find_shortest_distances(int *seed, int *neighbours, int root, int *dist)
{
    int n_walker, walker[MAX_WALKER];

    n_walker = 1;
    walker[0] = root;

    while (n_walker > 0) {
        int n_new_walker = 0;
        int new_walker[MAX_WALKER];

        for (int k = 0; k < n_walker; k++) {
            int i = walker[k];

            for (int ni = seed[i]; ni < seed[i+1]; ni++) {
                int j = neighbours[ni];
                if (dist[j] == 0) {
                    n_new_walker++;
                    if (n_new_walker > MAX_WALKER) {
                        PyErr_SetString(PyExc_RuntimeError,
                                        "MAX_WALKER exceeded");
                        return false;
                    }

                    new_walker[n_new_walker-1] = j;

                    dist[j] = dist[i]+1;
                }
            }
        }
 
        n_walker = n_new_walker;
        std::copy(new_walker, new_walker+n_new_walker, walker);
    }

    dist[root] = 0;

    return true;
}


/*
 * Look for the shortest distance starting at atom *root*,
 * look only for elements *f*
 */
bool
distance_map(int nat, int *seed, int *neighbours, int *dist, int *diameter)
{
    if (diameter) *diameter = 0;
    std::fill(dist, dist+nat*nat, 0);

    for (int i = 0; i < nat; i++) {
        if (!find_shortest_distances(seed, neighbours, i, &dist[i*nat]))
            return false;
        if (diameter) {
            *diameter = std::max(*diameter,
                                 *std::max_element(&dist[i*nat],
                                                   &dist[i*nat]+nat));
        }
    }

    return true;
}


/*
 * Python wrapper
 */
extern "C" PyObject *
py_distance_map(PyObject *self, PyObject *args)
{
    PyObject *py_i, *py_j;

    if (!PyArg_ParseTuple(args, "OO", &py_i, &py_j))
        return NULL;

    /* Make sure our arrays are contiguous */
    py_i = PyArray_FROMANY(py_i, NPY_INT, 1, 1, NPY_C_CONTIGUOUS);
    if (!py_i) return NULL;
    py_j = PyArray_FROMANY(py_j, NPY_INT, 1, 1, NPY_C_CONTIGUOUS);
    if (!py_j) return NULL;

    /* Check array shapes. */
    npy_intp nneigh = PyArray_DIM((PyArrayObject *) py_i, 0);
    if (PyArray_DIM((PyArrayObject *) py_j, 0) != nneigh) {
        PyErr_SetString(PyExc_ValueError, "Arrays must have same length.");
        return NULL;
    }

    /* Get total number of atoms */
    int *i = (int *) PyArray_DATA(py_i);
    int nat = *std::max_element(i, i+nneigh)+1;

    printf("nat = %i\n", nat);

    /* Construct seed array */
    int seed[nat+1];
    seed_array(nat, nneigh, i, seed);

    npy_intp dims[2];
    dims[0] = nat;
    dims[1] = nat;
    PyObject *py_dist = PyArray_ZEROS(2, dims, NPY_INT, 0);

    if (!distance_map(nat, seed, (int *) PyArray_DATA(py_j),
                      (int *) PyArray_DATA(py_dist), NULL)) {
        Py_DECREF(py_dist);
        return NULL;
    }

    return py_dist;
}

#if 0

/*
 * Look for the "shortest path" ring starting at atom *at*,
 * look only for elements *f*
 */
void
find_sp_rings(p, nl, cutoff, dist, max_ring_len, stat, stat_per_at, stat_per_slice, mask, plane, weight, error)
{
    /*
    implicit none

    type(particles_t),           intent(in)    :: p
    type(neighbors_t),           intent(in)    :: nl
    real(DP),                    intent(in)    :: cutoff
    integer,                     intent(in)    :: dist(p%nat, p%nat)
    integer,                     intent(in)    :: max_ring_len
    integer,           optional, intent(out)   :: stat(max_ring_len)
    logical,           optional, intent(in)    :: mask(p%nat)
    integer,           optional, intent(out)   :: error
    */



    ! ---

    double d[3], abs_dr_sq, cutoff_sq;

    bool done[neighbors_size];

    int n_walker;
    int walker[MAX_WALKER], last[MAX_WALKER];

    int ring_len[MAX_WALKER];
    int rings[max_ring_len, MAX_WALKER];
    double dr[3, max_ring_len, MAX_WALKER];

    int n_new_walker;
    int new_walker[MAX_WALKER], new_last[MAX_WALKER];

    int new_ring_len[MAX_WALKER];
    int new_rings[max_ring_len, MAX_WALKER];
    double new_dr[3, max_ring_len, MAX_WALKER];

    cutoff_sq  = cutoff*cutoff;

    for (i = 0; i < max_ring_len; i++) stat[i] = 0;
    for (i = 0; i < neighbours_size; i++) done[i] = false;

    for (a = 0; a < nat; a++) {
        if (!mask || mask[a]) {
            int na;
            for (na = seed[a]; na <= last[a]; na++) {
                int b = neighbors[na];

                if (a < b && (!mask || mask[b])) {
                    int ni;
                    done[na] = true;
                    for (ni = seed[b]; ni <= last[b]; ni++) {
                        if (neighbors[ni] == a) {
                            done[ni] = true;
                        }
                    }
                }

                n_walker = 1;
                walker[1] = b;
                last[1] = a;

                ring_len = 1;
                rings[1, 1] = b;
                dr[:, 1, 1] = GET_DRJ(p, nl, a, b, na)

                while (n_walker > 0) {
                    int k;

                    n_new_walker = 0;

                    for (k = 0; k < n_walker; k++) {
                        i = walker[k];

                        if (i > 0) {
                            int ni;
                            for (ni = seed[i]; ni <= last[i]; ni++) {
                                if (!done[ni]) {
                                    DISTJ_SQ(p, nl, i, ni, j, d, abs_dr_sq)

                                    if (abs_dr_sq < cutoff_sq && (!mask || mask[j]) && j != last[k]) {
                                        if (dist[j, a] == dist[i, a]+1) {
                                            if (ring_len[k] < (max_ring_len-1)/2) {
                                                int n;

                                                n_new_walker = n_new_walker+1;
                                                if (n_new_walker > MAX_WALKER) {
                                                  /* RAISE ERROR or increase buffer size */
                                                }

                                                new_walker[n_new_walker] = j;
                                                new_last[n_new_walker] = i;

                                                for (n = 0; n < ring_len[k]; n++) {
                                                    new_rings(n, n_new_walker) = rings(n, k);
                                                }
                                                new_ring_len[n_new_walker] = ring_len[k]+1;
                                                if (new_ring_len[n_new_walker] > max_ring_len) {
                                                    /* RAISE ERROR */
                                                }
                                                new_rings[new_ring_len[n_new_walker], n_new_walker] = j;

                                                new_dr[:, 1:ring_len[k], n_new_walker] = dr[:, 1:ring_len[k], n_new_walker];
                                                new_dr[:, new_ring_len[n_new_walker], n_new_walker] = dr[:, ring_len[k], k] + d;
                                            }
                                        }
                                        else {
                                            if (dist[j, a] == dist[i, a] .or. dist[j, a] == dist[i, a]-1) {
                                                int n;

                                                /* Reverse search direction */
                                                n_new_walker = n_new_walker+1;
                                                if (n_new_walker > MAX_WALKER) {
                                                  /* RAISE ERROR or increase buffer size */
                                                }

                                                new_walker[n_new_walker] = -j;
                                                new_last[n_new_walker] = i;

                                                for (n = 0; n < ring_len[k]; n++) {
                                                    new_rings[n, n_new_walker] = rings[n, k];
                                                }
                                                new_ring_len[n_new_walker] = ring_len[k]+1;
                                                if (new_ring_len[n_new_walker] > max_ring_len) {
                                                    /* RAISE ERROR */
                                                }
                                                new_rings[new_ring_len[n_new_walker], n_new_walker] = j;

                                                new_dr[:, 1:ring_len[k], n_new_walker] = dr[:, 1:ring_len[k], k];
                                                new_dr[:, new_ring_len(n_new_walker), n_new_walker] = dr[:, ring_len[k], k) + d;
                                            }
                                            else {
                                                stop "Something is wrong with the distance map."
                                            }
                                        }

                                    }

                                }
                            }
                        }
                        else {
                            int ni;
                            i = -i

                            for (ni = seed[i]; ni <= last[i]; ni++) {
                                if (!done[ni]) {
                                    DISTJ_SQ(p, nl, i, ni, j, d, abs_dr_sq)
                                    if (abs_dr_sq < cutoff_sq && (!mask || mask[j]) && j /= last[k]) {
                                        if (j == a) {
                                            d = d + dr[:, ring_len[k], k];
                                            if (dot_product(d, d) < TOL) {
                                                int m, n;

                                                /* Now we need to check whether this ring is SP */
                                                bool is_sp = true;

                                                ring_len[k] = ring_len[k]+1;
                                                if (ring_len[k] > max_ring_len) {
                                                    /* RAISE ERROR */
                                                }
                                                rings[ring_len[k], k] = a;

                                                for (m = 0; m < ring_len[k]; m++) {
                                                    for (n = m+2; n < ring_len[k]; n++) {
                                                        int dn = n-m;
                                                        if (dn > ring_len[k]/2) dn = ring_len[k]-dn;
                                                        if (dist[rings[n, k], rings[m, k]] != dn) {
                                                            is_sp = false;
                                                        }
                                                    }
                                                }

                                                if (is_sp) {
                                                    stat(ring_len[k]) = stat(ring_len[k])+1;
                                                }
                                            }
                                        else {
                                            if (dist(j, a) == dist(i, a)-1) {
                                                if (all(rings(1:ring_len[k], k) /= j)) {
                                                    n_new_walker = n_new_walker+1;
                                                    if (n_new_walker > MAX_WALKER) {
                                                        /* RAISE ERROR */
                                                    }

                                                    new_walker[n_new_walker] = -j;
                                                    new_last[n_new_walker] = i;

                                                    new_rings[1:ring_len[k], n_new_walker] = rings[1:ring_len[k], k];
                                                    new_ring_len[n_new_walker] = ring_len[k]+1;
                                                    if (new_ring_len[n_new_walker] > max_ring_len) {
                                                        /* RAISE ERROR */
                                                    }
                                                    new_rings[new_ring_len(n_new_walker), n_new_walker] = j;

                                                    new_dr[:, 1:ring_len[k], n_new_walker]               = dr[:, 1:ring_len[k], k];
                                                    new_dr[:, new_ring_len(n_new_walker), n_new_walker]  = dr[:, ring_len[k], k) + d;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    n_walker = n_new_walker;
                    std::copy(new_walker, new_walker+n_walker, walker);
                    std::copy(new_last, new_last+n_walker, last);

                    std::copy(new_ring_len, new_ring_len+n_walker, ring_len);
                    std::copy(new_rings, new_rings+max_ring_len*n_walker, rings);

                    dr(:, :, 1:n_walker)  = new_dr(:, :, 1:n_walker);
                }
            }
        }
    }
}


/*
 * Python wrapper
 */
extern "C" void
py_find_sp_rings(PyObject *self, PyObject *args)
{
    PyObject *py_i, *py_j, *py_r;

    if (!PyArg_ParseTuple(args, "OOO", &py_i, &py_j, &py_r))
        return NULL;

    /* Make sure our arrays are contiguous */
    py_i = PyArray_FROMANY(py_i, NPY_INT, 1, 1, NPY_C_CONTIGUOUS);
    if (!py_i) return NULL;
    py_j = PyArray_FROMANY(py_j, NPY_INT, 1, 1, NPY_C_CONTIGUOUS);
    if (!py_j) return NULL;
    py_r = PyArray_FROMANY(py_r, NPY_DOUBLE, 2, 2, NPY_C_CONTIGUOUS);
    if (!py_r) return NULL;

    /* Check array shapes. */
    npy_intp nneigh = PyArray_DIM((PyArrayObject *) py_i, 0);
    if (PyArray_DIM((PyArrayObject *) py_j, 0) != nneigh) {
        PyErr_SetString(PyExc_ValueError, "Array must have same length.");
        return NULL;
    }
    if (PyArray_DIM((PyArrayObject *) py_r, 0) != nneigh) {
        PyErr_SetString(PyExc_ValueError, "Array must have same length.");
        return NULL;
    }
    if (PyArray_DIM((PyArrayObject *) py_r, 1) != 3) {
        PyErr_SetString(PyExc_ValueError, "Distance array must have second "
                                          "dimension of length 3.");
        return NULL;
    }

    /* Get total number of atoms */
    int nat = *std::max_element(py_i, py_i+nneigh)+1;

    printf("nat = %i\n", nat);

    /* Construct seed array */
    int seed[nat+1];
    seed_array(nat, nneigh, py_i, seed);

    npy_intp max_ring_len = 32;
    PyObject *py_ringstat = PyArray_ZERO(1, &max_ring_len, NPY_INT, 0);

    return py_ringstat;
}

#endif