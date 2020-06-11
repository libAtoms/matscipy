# ======================================================================
# matscipy - Python materials science tools
# https://github.com/libAtoms/matscipy
#
# Copyright (2014) James Kermode, King's College London
#                  Lars Pastewka, Karlsruhe Institute of Technology
#                  Adrien Gola, Karlsruhe Institute of Technology
#                  Wolfram Nöhring, University of Freiburg
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# ======================================================================
"""Read and write tabulated EAM potentials"""

from __future__ import division, print_function

from collections import namedtuple

import numpy as np
try:
    from scipy import interpolate
except:
    print('Warning: No scipy')
    interpolate = False

import os

###


# Todo: replace by data class (requires Python > 3.7)
class EAMParameters(
    namedtuple(
        "EAMParameters",
        "symbols atomic_numbers "
        "atomic_masses lattice_constants crystal_structures "
        "number_of_density_grid_points "
        "number_of_distance_grid_points "
        "density_grid_spacing distance_grid_spacing "
        "cutoff",
    )
):
    """Embedded Atom Method potential parameters

    :param array_like symbols: Symbols of the elements coverered by
        this potential (only for eam/alloy and 
        eam/fs, EMPTY for eam
    :param array_like atomic_numbers: Atomic numbers of the elements 
        covered by this potential
    :param array_like atomic_masses: Atomic masses of the elements 
        covered by this potential
    :param array_like lattice_constants: Lattice constant of a pure crystal 
        with crystal structure as specified in crystal_structures
    :param array_like crystal_structures: Crystal structure of the pure metal.
    :param int number_of_density_grid_points: Number of grid points 
        of the embedding energy functional
    :param int number_of_distance_grid_points: Number of grid points of 
        the electron density function and the pair potential
    :param float density_grid_spacing: Grid spacing in electron density space
    :param float distance_grid_spacing: Grid spacing in pair distance space
    :param float cutoff: Cutoff distance of the potential
    """

    __slots__ = ()

###

def _strip_comments_from_line(string, marker="#"):
    """Strip comments from lines but retain newlines

    Parameters
    ----------
    string : str
        string which may contain comments
    marker : str
        marker which indicates the start of a comment

    Returns
    -------
    stripped_string : str
        string without commments; if the string terminated
        with a newline, then the newline is retained
    """
    start = string.find(marker)
    if start != -1:
        stripped_string = string[:start]
        if string.endswith("\n"):
            stripped_string += "\n"
    else:
        stripped_string = string
    return stripped_string


def read_eam(eam_file, kind="eam/alloy"):
    """Read a tabulated EAM potential
    
    There are differnt flavors of EAM, with different storage
    formats. This function supports a subset of the formats supported
    by Lammps (http://lammps.sandia.gov/doc/pair_eam.html),
    * eam (DYNAMO funcfl format)
    * eam/alloy (DYNAMO setfl format)
    * eam/fs (DYNAMO setfl format)
    
    Parameters
    ----------
    eam_file : string
        eam alloy file name 
    kind : {'eam', 'eam/alloy', 'eam/fs'}
        kind of EAM file to read

    Returns
    -------
    source : string
        Source informations or comment line for the file header
    parameters : EAMParameters
        EAM potential parameters
    F : array_like
        contain the tabulated values of the embedded functions
        shape = (nb elements, nb of data points)
    f : array_like
        contain the tabulated values of the density functions
        shape = (nb elements, nb of data points)
    rep : array_like
        contain the tabulated values of pair potential
        shape = (nb elements,nb elements, nb of data points)
    """
    supported_kinds = ["eam", "eam/alloy", "eam/fs"]
    if kind not in supported_kinds: 
        raise ValueError(f"EAM kind {kind} not supported")
    with open(eam_file, 'r') as file:
        eam = file.readlines()

    if kind == "eam":
        with open(eam_file, 'r') as file:
            # ignore comment characters on first line but strip them from subsequent lines
            lines = [file.readline()]
            lines.extend(_strip_comments_from_line(line) for line in file.readlines())
        # reading first comment line as source for eam potential data
        source = lines[0].strip()

        words = lines[1].strip().split()
        if len(words) != 4:
            raise ValueError(
                "expected four values on second line of EAM setfl file: "
                "atomic number, mass, lattice constant, lattice"
            )
        atomic_numbers = int(words[0])
        atomic_masses = float(words[1])
        lattice_parameters = float(words[2])
        crystal_structures = words[3]

        words = lines[2].strip().split()
        if len(words) != 5:
            raise ValueError(
                "expected five values on third line of EAM setfl file: "
                "Nrho, drho, Nr, dr, cutoff"
            )
        Nrho = int(words[0])       # Nrho (number of values for the embedding function F(rho))
        drho = float(words[1])     # spacing in density space
        Nr = int(words[2])         # Nr (number of values for the effective charge function Z(r) and density function rho(r))
        dr = float(words[3])       # spacing in distance space
        cutoff = float(words[4])
        parameters = EAMParameters(
            np.zeros(1), atomic_numbers, atomic_masses, lattice_parameters, 
            crystal_structures, Nrho, Nr, drho, dr, cutoff
        )

        # Strip empty lines 
        remaining_lines = [line for line in lines[3:] if len(line.strip()) > 0]
        remaining_words = []
        for line in remaining_lines:
            words = line.split()
            remaining_words.extend(words)
        expected_length = Nrho + 2 * Nr
        true_length = len(remaining_words)
        if true_length != expected_length:
            raise ValueError(f"expected {expected_length} tabulated values, but there are {true_length}")
        data = np.array(remaining_words, dtype=float)
        F = data[0:Nrho]
        f = data[Nrho:Nrho+Nr]
        rep = data[Nrho+Nr:2*Nr+Nrho]
        return source, parameters, F, f, rep

    if kind in ["eam/alloy", "eam/fs"]:
        """eam/alloy and eam/fs have almost the same structure, except for the electron density section"""
        with open(eam_file, 'r') as file:
            # ignore comment characters on first line but strip them from subsequent lines
            lines = [file.readline() for _ in range(3)]
            lines.extend(_strip_comments_from_line(line) for line in file.readlines())
        # reading 3 first comment lines as source for eam potential data
        source = "".join(line.strip() for line in lines[:3])

        words = lines[3].strip().split()
        alleged_num_elements = int(words[0])
        elements = words[1:]
        true_num_elements = len(elements)
        if alleged_num_elements != true_num_elements:
            raise ValueError(
                f"Header claims there are tables for {alleged_num_elements} elements, "
                f"but actual element list has {true_num_elements} elements: {' '.join(elements)}"
            )

        words = lines[4].strip().split()
        Nrho = int(words[0])     # Nrho (number of values for the embedding function F(rho))
        drho = float(words[1])   # spacing in density space
        Nr = int(words[2])       # Nr (number of values for the effective charge function Z(r) and density function rho(r))
        dr = float(words[3])     # spacing in distance space
        cutoff = float(words[4])

        # Strip empty lines and check that the table contains the expected number of values
        remaining_lines = [line for line in lines[5:] if len(line.strip()) > 0]
        remaining_words = []
        for line in remaining_lines:
            words = line.split()
            remaining_words.extend(words)
        if kind == "eam/fs":
            expected_num_density_functions_per_element = true_num_elements
        else:
            expected_num_density_functions_per_element = 1
        expected_num_words_per_element = (
            4 + 
            Nrho + 
            expected_num_density_functions_per_element * Nr 
        )
        expected_num_pair_functions = np.sum(np.arange(1, true_num_elements+1)).astype(int)
        expected_length = true_num_elements * expected_num_words_per_element + expected_num_pair_functions * Nr
        true_length = len(remaining_words)
        if true_length != expected_length:
            raise ValueError(f"expected {expected_length} tabulated values, but there are {true_length}")

        atomic_numbers = np.zeros(true_num_elements, dtype=int)
        atomic_masses = np.zeros(true_num_elements)
        lattice_parameters = np.zeros(true_num_elements)
        crystal_structures = np.empty(true_num_elements).astype(np.str) # fixme: be careful with string length
        F = np.zeros((true_num_elements, Nrho))
        for i in range(true_num_elements):
            offset = i * expected_num_words_per_element
            atomic_numbers[i] = int(remaining_words[offset])
            atomic_masses[i] = float(remaining_words[offset+1])
            lattice_parameters[i] = float(remaining_words[offset+2])
            crystal_structures[i] = remaining_words[offset+3]
            F[i, :] = np.array(remaining_words[offset+4:offset+4+Nrho], dtype=float)

        # Read data for individual elemements
        if kind == "eam/alloy":
            f = np.zeros((true_num_elements, Nr))
            for i in range(true_num_elements):
                offset = i * expected_num_words_per_element + 4 + Nrho
                f[i, :] = np.array(remaining_words[offset:offset+Nr], dtype=float)
        if kind == "eam/fs":
            f = np.zeros((true_num_elements, true_num_elements, Nr))
            for i in range(true_num_elements):
                offset = i * expected_num_words_per_element + 4 + Nrho
                for j in range(true_num_elements):
                    f[i, j, :] = np.array(remaining_words[offset+j*Nr:offset+(j+1)*Nr], dtype=float)

        # Read pair data
        rep = np.zeros((true_num_elements, true_num_elements, Nr))
        rows, cols = np.tril_indices(true_num_elements)
        for pair_number, (i, j) in enumerate(zip(rows, cols)):
            offset = true_num_elements * expected_num_words_per_element + pair_number * Nr
            rep[i, j, :] = np.array(remaining_words[offset:offset+Nr], dtype=float)
            rep[j, i, :] = rep[i, j, :]

        parameters = EAMParameters(
            elements, atomic_numbers, atomic_masses, 
            lattice_parameters, crystal_structures, 
            Nrho, Nr, drho, dr, cutoff
        )
        return source, parameters, F, f, rep
            

def mix_eam(files,kind,method,f=[],rep_ab=[],alphas=[],betas=[]):
    """
    mix eam alloy files data set and compute the interspecies pair potential part using the 
    mean geometric value from each pure species 
    
    Parameters
    ----------
    files : array of strings
            Contain all the files to merge and mix
    kind : string
            kinf of eam. Supported eam/alloy, eam/fs
    method : string, {geometric, arithmetic, weighted, fitted}
        Method used to mix the pair interaction terms. The geometric,
        arithmetic, and weighted arithmetic average are available. The weighted
        arithmetic method is using the electron density function values of atom
        :code:`a` and :code:`b` to ponderate the pair potential between species
        :code:`a` and :math:`b`, :code:`rep_ab = 0.5(fb/fa * rep_a + fa/fb *
        rep_b)`, see [1]. The fitted method is to be used if :code:`rep_ab`
        has been previously fitted and is parse as :math:`rep_ab` karg.
    f : np.array 
        fitted density term (for FS eam style)
    rep_ab : np.array 
        fitted rep_ab term
    alphas : array
        fitted alpha values for the fine tuned mixing. 
        :code:`rep_ab = alpha_a*rep_a+alpha_b*rep_b`
    betas : array 
        fitted values for the fine tuned mixing. 
        :code:`f_ab = beta_00*rep_a+beta_01*rep_b`
        :code:`f_ba = beta_10*rep_a+beta_11*rep_b`

    Returns
    -------
    sources : string
        Source informations or comment line for the file header
    parameters_mix: EAMParameters
        EAM potential parameters
    F_ : array_like
        contain the tabulated values of the embedded functions
        shape = (nb elements, nb elements, nb of data points)
    f_ : array_like
        contain the tabulated values of the density functions
        shape = (nb elements, nb elements, nb of data points)
    rep_ : array_like
        contain the tabulated values of pair potential
        shape = (nb elements, nb elements, nb of data points)

    References
    ----------

    1. X. W. Zhou, R. A. Johnson, and H. N. G. Wadley, Phys. Rev. B, 69, 144113 (2004)
    """

    nb_at = 0
    # Counting elements and repartition and select smallest tabulated set Nrho*drho // Nr*dr
    Nrho,drho,Nr,dr,cutoff = np.empty((len(files))),np.empty((len(files))),np.empty((len(files))),np.empty((len(files))),np.empty((len(files)))
    sources = ""
    if kind == "eam/alloy":
        for i,f_eam in enumerate(files):
            source,parameters, F,f,rep = read_eam(f_eam,kind="eam/alloy")
            sources+= source
            source += " "
            nb_at+=len(parameters[0])
            Nrho[i] = parameters[5]
            drho[i] = parameters[7]
            cutoff[i] = parameters[9]
            Nr[i] = parameters[6]
            dr[i] = parameters[8]
        # --- #
        max_cutoff = cutoff.argmax()
        max_prod = (Nrho*drho).argmax()
        max_prod_r = (Nr*dr).argmax()
        atomic_numbers,atomic_masses,lattice_parameters,crystal_structures,elements = np.empty(0),np.empty(0),np.empty(0),np.empty(0).astype(np.str),np.empty(0).astype(np.str)
        Nr_ = Nr[max_prod_r]
        dr_ = ((Nr*dr).max())/Nr_
        Nrho_ = Nrho[max_prod]
        drho_ = ((Nrho*drho).max())/Nrho_
        
        if Nr_ > 2000:
          Nr_ = 2000   # reduce
          dr_ = ((Nr*dr).max())/Nr_
        if Nrho_ > 2000:
          Nrho_ = 2000 # reduce
          drho_ = ((Nrho*drho).max())/Nrho_
        F_,f_,rep_ = np.empty((nb_at,Nrho_)),np.empty((nb_at,Nr_)),np.empty((nb_at,nb_at,Nr_))
        at = 0
        for i,f_eam in enumerate(files):
            source,parameters, F,f,rep = read_eam(f_eam,kind="eam/alloy")
            elements = np.append(elements,parameters[0])
            atomic_numbers = np.append(atomic_numbers,parameters[1])
            atomic_masses = np.append(atomic_masses,parameters[2])
            lattice_parameters = np.append(lattice_parameters,parameters[3])
            crystal_structures = np.append(crystal_structures,parameters[4])
            for j in range(len(parameters[0])):
                F_[at,:] = interpolate.InterpolatedUnivariateSpline(np.linspace(0,Nrho[i]*drho[i],Nrho[i]),F[j,:])(np.linspace(0,Nrho_*drho_,Nrho_))
                f_[at,:] = interpolate.InterpolatedUnivariateSpline(np.linspace(0,Nr[i]*dr[i],Nr[i]),f[j,:])(np.linspace(0,Nr_*dr_,Nr_))
                rep_[at,at,:] = interpolate.InterpolatedUnivariateSpline(np.linspace(0,Nr[i]*dr[i],Nr[i]),rep[j,j,:])(np.linspace(0,Nr_*dr_,Nr_))
                at+=1
        # mixing repulsive part
        old_err_state = np.seterr(divide='raise')
        ignored_states = np.seterr(**old_err_state)
        for i in range(nb_at):
            for j in range(nb_at):
                if j < i :
                    if method == "geometric":
                        rep_[i,j,:] = (rep_[i,i,:]*rep_[j,j,:])**0.5
                    if method == "arithmetic":
                        if alphas:
                            rep_[i,j,:] = alphas[i]*rep_[i,i,:]+alphas[j]*rep_[j,j,:]
                        else:
                            rep_[i,j,:] = 0.5*(rep_[i,i,:]+rep_[j,j,:])
                    if method == "weighted":
                        rep_[i,j,:] = 0.5*(np.divide(f_[j,:],f_[i,:])*rep_[i,i,:]+np.divide(f_[i,:],f_[j,:])*rep_[j,j,:])
                    if method == "fitted":
                        rep_ab[np.isnan(rep_ab)] = 0
                        rep_ab[np.isinf(rep_ab)] = 0
                        rep_[i,j,:] = interpolate.InterpolatedUnivariateSpline(np.linspace(0,max(Nr*dr),rep_ab.shape[0]),rep_ab)(np.linspace(0,Nr_*dr_,Nr_))
                    rep_[i,j,:][np.isnan(rep_[i,j,:])] = 0
                    rep_[i,j,:][np.isinf(rep_[i,j,:])] = 0
    elif kind == "eam/fs":
        for i,f_eam in enumerate(files):
            source,parameters, F,f,rep = read_eam(f_eam,kind="eam/alloy")
            sources+= source
            source += " "
            nb_at+=len(parameters[0])
            Nrho[i] = parameters[5]
            drho[i] = parameters[7]
            cutoff[i] = parameters[9]
            Nr[i] = parameters[6]
            dr[i] = parameters[8]
        # --- #
        max_cutoff = cutoff.argmax()
        max_prod = (Nrho*drho).argmax()
        max_prod_r = (Nr*dr).argmax()
        atomic_numbers,atomic_masses,lattice_parameters,crystal_structures,elements = np.empty(0),np.empty(0),np.empty(0),np.empty(0).astype(np.str),np.empty(0).astype(np.str)
        Nr_ = Nr[max_prod_r]
        dr_ = ((Nr*dr).max())/Nr_
        Nrho_ = Nrho[max_prod]
        drho_ = ((Nrho*drho).max())/Nrho_
        
        if Nr_ > 2000:
          Nr_ = 2000   # reduce
          dr_ = ((Nr*dr).max())/Nr_
        if Nrho_ > 2000:
          Nrho_ = 2000 # reduce
          drho_ = ((Nrho*drho).max())/Nrho_
        F_,f_,rep_ = np.empty((nb_at,Nrho_)),np.empty((nb_at,nb_at,Nr_)),np.empty((nb_at,nb_at,Nr_))
        at = 0
        for i,f_eam in enumerate(files):
            source,parameters, F,f,rep = read_eam(f_eam,kind="eam/alloy")
            elements = np.append(elements,parameters[0])
            atomic_numbers = np.append(atomic_numbers,parameters[1])
            atomic_masses = np.append(atomic_masses,parameters[2])
            lattice_parameters = np.append(lattice_parameters,parameters[3])
            crystal_structures = np.append(crystal_structures,parameters[4])
            for j in range(len(parameters[0])):
                F_[at,:] = interpolate.InterpolatedUnivariateSpline(np.linspace(0,Nrho[i]*drho[i],Nrho[i]),F[j,:])(np.linspace(0,Nrho_*drho_,Nrho_))
                f_[at,at,:] = interpolate.InterpolatedUnivariateSpline(np.linspace(0,Nr[i]*dr[i],Nr[i]),f[j,:])(np.linspace(0,Nr_*dr_,Nr_))
                rep_[at,at,:] = interpolate.InterpolatedUnivariateSpline(np.linspace(0,Nr[i]*dr[i],Nr[i]),rep[j,j,:])(np.linspace(0,Nr_*dr_,Nr_))
                at+=1
        # mixing density part
        old_err_state = np.seterr(divide='raise')
        ignored_states = np.seterr(**old_err_state)
        for i in range(nb_at):
            for j in range(nb_at):
                if i!=j:
                    if method == "geometric":
                        f_[i,j,:] = (f_[i,i,:]*f_[j,j,:])**0.5
                    if method == "arithmetic":
                        if betas.any():
                            f_[i,j,:] = betas[i,i]*f_[i,i,:]+betas[i,j]*f_[j,j,:]
                        else:
                            f_[i,j,:] = 0.5*(f_[i,i,:]+f_[j,j,:])
                    if method == "fitted":
                        f_ab[np.isnan(f_ab)] = 0
                        f_ab[np.isinf(f_ab)] = 0
                        f_[i,j,:] = interpolate.InterpolatedUnivariateSpline(np.linspace(0,max(Nr*dr),rep_ab.shape[0]),rep_ab)(np.linspace(0,Nr_*dr_,Nr_))
                    f_[i,j,:][np.isnan(f_[i,j,:])] = 0
                    f_[i,j,:][np.isinf(f_[i,j,:])] = 0
        # mixing repulsive part
        for i in range(nb_at):
            for j in range(nb_at):
                if j < i :
                    if method == "geometric":
                        rep_[i,j,:] = (rep_[i,i,:]*rep_[j,j,:])**0.5
                    if method == "arithmetic":
                        if alphas:
                            rep_[i,j,:] = alphas[i]*rep_[i,i,:]+alphas[j]*rep_[j,j,:]
                        else:
                            rep_[i,j,:] = 0.5*(rep_[i,i,:]+rep_[j,j,:])
                    if method == "fitted":
                        rep_ab[np.isnan(rep_ab)] = 0
                        rep_ab[np.isinf(rep_ab)] = 0
                        rep_[i,j,:] = interpolate.InterpolatedUnivariateSpline(np.linspace(0,max(Nr*dr),rep_ab.shape[0]),rep_ab)(np.linspace(0,Nr_*dr_,Nr_))
                    rep_[i,j,:][np.isnan(rep_[i,j,:])] = 0
                    rep_[i,j,:][np.isinf(rep_[i,j,:])] = 0
    else:
        raise ValueError(f"EAM kind {kind} is not supported")
                
    parameters_mix = EAMParameters(elements, atomic_numbers, atomic_masses,lattice_parameters,crystal_structures, Nrho_,Nr_, drho_, dr_, cutoff[max_cutoff])
    return sources, parameters_mix, F_, f_, rep_

      
def write_eam(source, parameters, F, f, rep, out_file, kind="eam"):
    """Write an eam lammps format file 

    There are differnt flavors of EAM, with different storage
    formats. This function supports a subset of the formats supported
    by Lammps (http://lammps.sandia.gov/doc/pair_eam.html),
    * eam (DYNAMO funcfl format)
    * eam/alloy (DYNAMO setfl format)
    * eam/fs (DYNAMO setfl format)
    
    Parameters
    ----------
    source : string
          Source information or comment line for the file header
    parameters_mix: EAMParameters
        EAM potential parameters
    F : array_like
        contain the tabulated values of the embedded functions
        shape = (nb of data points)
    f : array_like
        contain the tabulated values of the density functions
        shape = (nb of data points)
    rep : array_like
        contain the tabulated values of pair potential
        shape = (nb of data points)
    out_file : string
              output file name for the eam alloy potential file
    kind : {'eam', 'eam/alloy', 'eam/fs'}
        kind of EAM file to read

    Returns
    -------
    None
    """
  
    elements, atomic_numbers, atomic_masses, lattice_parameters, crystal_structures = parameters[0:5]
    Nrho, Nr, drho, dr, cutoff = parameters[5:10]
    
    if kind == "eam":
        # parameters unpacked
        atline = f"{int(atomic_numbers)} {float(atomic_masses)}  {float(lattice_parameters)} {str(crystal_structures)}"
        parameterline = f'{int(Nrho)}\t{float(drho):.16e}\t{int(Nr)}\t{float(dr):.16e}\t{float(cutoff):.10e}'
        potheader = f"# EAM potential from : # {source} \n {atline} \n {parameterline}"
        # --- Writing new EAM alloy pot file --- #
        # write header and file parameters
        potfile = open(out_file,'wb')
        # write F and f tables
        np.savetxt(potfile, F, fmt='%.16e', header=potheader, comments='')
        np.savetxt(potfile, f, fmt='%.16e')
        # write pair interactions tables
        np.savetxt(potfile, rep, fmt='%.16e')
        potfile.close()  
    elif kind == "eam/alloy":
        num_elements = len(elements)
        # parameters unpacked
        potheader = f"# Mixed EAM alloy potential from :\n# {source} \n# \n"
        # --- Writing new EAM alloy pot file --- #
        potfile = open(out_file,'wb')
        # write header and file parameters
        np.savetxt(
            potfile, elements, fmt="%s", newline=' ', 
            header=potheader+str(num_elements), 
            footer=f'\n{Nrho}\t{drho:e}\t{Nr}\t{dr:e}\t{cutoff:e}\n', 
            comments=''
        )
        # write F and f tables
        for i in range(num_elements):
            np.savetxt(
                potfile, np.append(F[i,:], f[i,:]), fmt="%.16e", 
                header=f'{atomic_numbers[i]:d}\t{atomic_masses[i]}\t{lattice_parameters[i]}\t{crystal_structures[i]}',
                comments=''
            )
        # write pair interactions tables
        [[np.savetxt(potfile,rep[i,j,:],fmt="%.16e") for j in range(rep.shape[0]) if j <= i] for i in range(rep.shape[0])]
        potfile.close() 
    elif kind == "eam/fs":
        num_elements = len(elements)
        # parameters unpacked
        potheader = f"# Mixed EAM fs potential from :\n# {source} \n# \n"
        # --- Writing new EAM alloy pot file --- #
        potfile = open(out_file,'wb')
        # write header and file parameters
        np.savetxt(
            potfile, elements, fmt="%s", newline=' ', 
            header=potheader+str(num_elements), 
            footer=f'\n{Nrho}\t{drho:e}\t{Nr}\t{dr:e}\t{cutoff:e}\n', 
            comments=''
        )
        # write F and f tables
        for i in range(num_elements):
            np.savetxt(
                potfile, np.append(F[i,:], f[i,:,:].flatten()), fmt="%.16e", 
                header=f'{atomic_numbers[i]:d}\t{atomic_masses[i]}\t{lattice_parameters[i]}\t{crystal_structures[i]}',
                comments=''
            )
        # write pair interactions tables
        [[np.savetxt(potfile, rep[i,j,:], fmt="%.16e") for j in range(rep.shape[0]) if j <= i] for i in range(rep.shape[0])]
        potfile.close() 
    else:
        raise ValueError(f"EAM kind {kind} is not supported")
