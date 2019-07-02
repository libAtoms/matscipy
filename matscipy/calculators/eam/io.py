# ======================================================================
# matscipy - Python materials science tools
# https://github.com/libAtoms/matscipy
#
# Copyright (2014) James Kermode, King's College London
#                  Lars Pastewka, Karlsruhe Institute of Technology
#                  Adrien Gola, Karlsruhe Institute of Technology
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

EAMParameters = namedtuple('EAMParameters', 'symbols atomic_numbers '
                           'atomic_masses lattice_constants crystal_structures '
                           'number_of_density_grid_points '
                           'number_of_distance_grid_points '
                           'density_grid_spacing distance_grid_spacing '
                           'cutoff')

###

def read_eam(eam_file,kind="eam/alloy"):
    """
    Read an eam alloy lammps format file and return the tabulated data and parameters
    http://lammps.sandia.gov/doc/pair_eam.html
    
    Parameters
    ----------
      eam_file : string
                      eam alloy file name 
      kind : string
             kind of EAM file to read (supported eam,eam/alloy,eam/fs)
    Returns
    -------
      source : string
          Source informations or comment line for the file header
      parameters : list of tuples
                [0] - array of str - atoms (ONLY FOR eam/alloy and eam/fs, EMPTY for eam)
                [1] - array of int - atomic numbers
                [2] - array of float -atomic masses
                [3] - array of float - equilibrium lattice parameter
                [4] - array of str - crystal structure
                [5] - int - number of data point for embedded function
                [6] - int - number of data point for density and pair functions
                [7] - float - step size for the embedded function
                [8] - float - step size for the density and pair functions
                [9] - float - cutoff of the potentials
      F : array_like
          contain the tabulated values of the embedded functions
          shape = (nb atoms, nb of data points)
      f : array_like
          contain the tabulated values of the density functions
          shape = (nb atoms, nb of data points)
      rep : array_like
          contain the tabulated values of pair potential
          shape = (nb atoms,nb atoms, nb of data points)
    """
    with open(eam_file, 'r') as file:
        eam = file.readlines()
    if kind=="eam":
        # reading first comment line as source for eam potential data
        source = eam[0].strip()
        # -- Parameters -- #
        atnumber = int(eam[1].split()[0])
        atmass = float(eam[1].split()[1])
        crystallatt = float(eam[1].split()[2])
        crystal = eam[1].split()[3]
        Nrho = int(eam[2].split()[0])       # Nrho (number of values for the embedding function F(rho))
        Nr = int(eam[2].split()[2])         # Nr (number of values for the effective charge function Z(r) and density function rho(r))
        drho = float(eam[2].split()[1])     # spacing in density space
        dr = float(eam[2].split()[3]) # spacing in distance space
        cutoff = float(eam[2].split()[4])
        parameters = EAMParameters(np.zeros(1),atnumber, atmass,crystallatt,crystal, Nrho,Nr, drho, dr, cutoff)
        # -- Tabulated data -- #
        data = np.loadtxt(eam_file, dtype="float", skiprows = 3).flatten()
        F = data[0:Nrho]
        f = data[Nrho:Nrho+Nr]
        rep = data[Nrho+Nr:2*Nr+Nrho]
        return source,parameters, F,f,rep
    elif kind=="eam/alloy":
        # reading 3 first comment lines as source for eam potential data
        source = eam[0].strip()+eam[1].strip()+eam[2].strip()
        # -- Parameters -- #
        atoms = eam[3].strip().split()[1:]
        nb_atoms = len(atoms)
        Nrho = int(eam[4].split()[0])       # Nrho (number of values for the embedding function F(rho))
        Nr = int(eam[4].split()[2])         # Nr (number of values for the effective charge function Z(r) and density function rho(r))
        drho = float(eam[4].split()[1])     # spacing in density space
        dr = float(eam[4].split()[3]) # spacing in distance space
        cutoff = float(eam[4].split()[4])
        atnumber,atmass,crystallatt,crystal = np.empty(nb_atoms,dtype=int),np.empty(nb_atoms),np.empty(nb_atoms),np.empty(nb_atoms).astype(np.str)
        for i in range(nb_atoms):
            # Fixme: The following lines assume that data occurs in blocks of
            # homogeneous width. This can break.
            l = len(eam[6].strip().split())
            row = int(5+i*((Nr+Nrho)/l+1))
            atnumber[i] = int(eam[row].split()[0])
            atmass[i] = float(eam[row].split()[1])
            crystallatt[i] = float(eam[row].split()[2])
            crystal[i] = str(eam[row].split()[3])
        parameters = EAMParameters(atoms,atnumber,atmass,crystallatt,crystal,Nrho,Nr,drho,dr,cutoff)
        # -- Tabulated data -- #
        F,f,rep,data = np.empty((nb_atoms,Nrho)),np.empty((nb_atoms,Nr)),np.empty((nb_atoms,nb_atoms,Nr)),np.empty(())
        eam = open(eam_file,'r')
        [eam.readline() for i in range(5)]
        for i in range(nb_atoms):
            eam.readline()
            data = np.append(data,np.fromfile(eam,count=Nrho+Nr, sep=' '))
        data = np.append(data,np.fromfile(eam,count=-1, sep=' '))
        data = data[1:]
        for i in range(nb_atoms):
            F[i,:] = data[i*(Nrho+Nr):Nrho+i*(Nrho+Nr)]
            f[i,:] = data[Nrho+i*(Nrho+Nr):Nrho+Nr+i*(Nrho+Nr)]
        interaction = 0
        for i in range(nb_atoms):
            for j in range(nb_atoms):
                if j < i :
                    rep[i,j,:] = data[nb_atoms*(Nrho+Nr)+interaction*Nr:nb_atoms*(Nrho+Nr)+interaction*Nr+Nr]
                    rep[j,i,:] = data[nb_atoms*(Nrho+Nr)+interaction*Nr:nb_atoms*(Nrho+Nr)+interaction*Nr+Nr]
                    interaction+=1
            rep[i,i,:] = data[nb_atoms*(Nrho+Nr)+interaction*Nr:nb_atoms*(Nrho+Nr)+interaction*Nr+Nr]
            interaction+=1
        return source,parameters, F,f,rep
    elif kind=="eam/fs":
        # reading 3 first comment lines as source for eam potential data
        source = eam[0].strip()+eam[1].strip()+eam[2].strip()
        # -- Parameters -- #
        atoms = eam[3].strip().split()[1:]
        nb_atoms = len(atoms)
        Nrho = int(eam[4].split()[0])       # Nrho (number of values for the embedding function F(rho))
        Nr = int(eam[4].split()[2])         # Nr (number of values for the effective charge function Z(r) and density function rho(r))
        drho = float(eam[4].split()[1])     # spacing in density space
        dr = float(eam[4].split()[3]) # spacing in distance space
        cutoff = float(eam[4].split()[4])
        atnumber,atmass,crystallatt,crystal = np.empty(nb_atoms,dtype=int),np.empty(nb_atoms),np.empty(nb_atoms),np.empty(nb_atoms).astype(np.str)
        for i in range(nb_atoms):
            # Fixme: The following lines assume that data occurs in blocks of
            # homogeneous width. This can break.
            l = len(eam[6].strip().split())
            row = int(5+i*((Nr*nb_atoms+Nrho)/l+1))
            atnumber[i] = int(eam[row].split()[0])
            atmass[i] = float(eam[row].split()[1])
            crystallatt[i] = float(eam[row].split()[2])
            crystal[i] = str(eam[row].split()[3])
        parameters = EAMParameters(atoms,atnumber,atmass,crystallatt,crystal,Nrho,Nr,drho,dr,cutoff)
        # -- Tabulated data -- #
        F,f,rep,data = np.empty((nb_atoms,Nrho)),np.empty((nb_atoms,nb_atoms,Nr)),np.empty((nb_atoms,nb_atoms,Nr)),np.empty(())
        eam = open(eam_file,'r')
        [eam.readline() for i in range(5)]
        for i in range(nb_atoms):
            eam.readline()
            data = np.append(data,np.fromfile(eam,count=Nrho+Nr*nb_atoms, sep=' '))
        data = np.append(data,np.fromfile(eam,count=-1, sep=' '))
        data = data[1:]
        for i in range(nb_atoms):
            F[i,:] = data[i*(Nrho+Nr*nb_atoms):Nrho+i*(Nrho+Nr*nb_atoms)]
            #f[i,:] = data[Nrho+i*(Nrho+Nr):Nrho+Nr+i*(Nrho+Nr)]
        interaction = 0
        for i in range(nb_atoms):
            for j in range(nb_atoms):
                f[i,j,:] = data[Nrho+j*Nr+i*(Nrho+Nr*nb_atoms):Nr+Nrho+j*Nr+i*(Nrho+Nr*nb_atoms)]
                if j < i :
                    rep[i,j,:] = data[nb_atoms*(Nrho+Nr*nb_atoms)+interaction*Nr:nb_atoms*(Nrho+Nr*nb_atoms)+interaction*Nr+Nr]
                    rep[j,i,:] = data[nb_atoms*(Nrho+Nr*nb_atoms)+interaction*Nr:nb_atoms*(Nrho+Nr*nb_atoms)+interaction*Nr+Nr]
                    interaction+=1
            rep[i,i,:] = data[nb_atoms*(Nrho+Nr*nb_atoms)+interaction*Nr:nb_atoms*(Nrho+Nr*nb_atoms)+interaction*Nr+Nr]
            interaction+=1
        eam.close()
        return source,parameters, F,f,rep
    else:
        print('Non supported eam file type')
        raise ValueError
            

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
      method : string, (geometric,arithmetic,weighted,fitted)
              Method used to mix the pair interaction terms
              Available : Geometric average, arithmetic average, weighted arithmetic average
                  The Weighted arithmetic method is using the electron density function values of atom a and b to 
                  ponderate the pair potential between species a and b, 
                  rep_ab = 0.5*(fb/fa * rep_a + fa/fb * rep_b)
                  ref : X. W. Zhou, R. A. Johnson, and H. N. G. Wadley, Phys. Rev. B, 69, 144113 (2004)
                  Fitted method is to be used if the rep_ab has been previously fitted and is parse as rep_ab karg
      f : np.array of the fitted density term (for FS eam style)
      rep_ab : np.array of the fitted rep_ab term
      alphas : array of fitted alpha values for the fine tuned mixing. rep_ab = alpha_a*rep_a+alpha_b*rep_b
      betas : array of fitted values for the fine tuned mixing. f_ab = beta_00*rep_a+beta_01*rep_b
                                                                f_ba = beta_10*rep_a+beta_11*rep_b
    Returns
    -------
      sources : string
              Source informations or comment line for the file header
      parameters_mix : list of tuples
                [0] - array of str - atoms 
                [1] - array of int - atomic numbers
                [2] - array of float -atomic masses
                [3] - array of float - equilibrium lattice parameter
                [4] - array of str - crystal structure
                [5] - int - number of data point for embedded function
                [6] - int - number of data point for density and pair functions
                [7] - float - step size for the embedded function
                [8] - float - step size for the density and pair functions
                [9] - float - cutoff of the potentials
      F_ : array_like
          contain the tabulated values of the embedded functions
          shape = (nb atoms,nb atoms, nb of data points)
      f_ : array_like
          contain the tabulated values of the density functions
          shape = (nb atoms,nb atoms, nb of data points)
      rep_ : array_like
          contain the tabulated values of pair potential
          shape = (nb atoms,nb atoms, nb of data points)
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
        atnumber,atmass,crystallatt,crystal,atoms = np.empty(0),np.empty(0),np.empty(0),np.empty(0).astype(np.str),np.empty(0).astype(np.str)
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
            atoms = np.append(atoms,parameters[0])
            atnumber = np.append(atnumber,parameters[1])
            atmass = np.append(atmass,parameters[2])
            crystallatt = np.append(crystallatt,parameters[3])
            crystal = np.append(crystal,parameters[4])
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
        atnumber,atmass,crystallatt,crystal,atoms = np.empty(0),np.empty(0),np.empty(0),np.empty(0).astype(np.str),np.empty(0).astype(np.str)
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
            atoms = np.append(atoms,parameters[0])
            atnumber = np.append(atnumber,parameters[1])
            atmass = np.append(atmass,parameters[2])
            crystallatt = np.append(crystallatt,parameters[3])
            crystal = np.append(crystal,parameters[4])
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
        print('Non supported eam file type')
        raise ValueError
                
    parameters_mix = EAMParameters(atoms, atnumber, atmass,crystallatt,crystal, Nrho_,Nr_, drho_, dr_, cutoff[max_cutoff])
    return sources, parameters_mix, F_, f_, rep_
      
def write_eam(source, parameters, F, f, rep,out_file,kind="eam"):
    """
    Write an eam lammps format file 
    http://lammps.sandia.gov/doc/pair_eam.html
    
    Parameters
    ----------
    source : string
          Source information or comment line for the file header
    parameters : list of tuples
                [0] - array of str - atoms (ONLY FOR eam/alloy and eam/fs, EMPTY for eam)
                [1] - array of int - atomic numbers
                [2] - array of float -atomic masses
                [3] - array of float - equilibrium lattice parameter
                [4] - array of str - crystal structure
                [5] - int - number of data point for embedded function
                [6] - int - number of data point for density and pair functions
                [7] - float - step size for the embedded function
                [8] - float - step size for the density and pair functions
                [9] - float - cutoff of the potentials
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
    Returns
    -------
      
    """
  
    atoms, atnumber, atmass, crystallatt, crystal = parameters[0:5]
    Nrho, Nr, drho, dr, cutoff = parameters[5:10]
    
    if kind == "eam":
        # parameters unpacked
        atline = "%i %f %f %s"%(int(atnumber),float(atmass),float(crystallatt),str(crystal))
        parameterline = '%i\t%.16e\t%i\t%.16e\t%.10e'%(int(Nrho),float(drho),int(Nr),float(dr),float(cutoff))
        potheader = "# EAM potential from : # %s \n %s \n %s"%(source,atline,parameterline)
        # --- Writing new EAM alloy pot file --- #
        # write header and file parameters
        potfile = open(out_file,'wb')
        # write F and f tables
        np.savetxt(potfile, F, fmt='%.16e',header=potheader,comments='')
        np.savetxt(potfile, f, fmt='%.16e')
        # write pair interactions tables
        np.savetxt(potfile, rep, fmt='%.16e')
        potfile.close()  
    elif kind == "eam/alloy":
        nb_atoms = len(atoms)
        # parameters unpacked
        potheader = "# Mixed EAM alloy potential from :\n# %s \n# \n"%(source)
        # --- Writing new EAM alloy pot file --- #
        potfile = open(out_file,'wb')
        # write header and file parameters
        np.savetxt(potfile,atoms,fmt="%s",newline=' ', header=potheader+str(nb_atoms),footer='\n%i\t%e\t%i\t%e\t%e\n'%(Nrho,drho,Nr,dr,cutoff), comments='')
        # write F and f tables
        [np.savetxt(potfile,np.append(F[i,:],f[i,:]),fmt="%.16e",header='%i\t%f\t%f\t%s'%(atnumber[i],atmass[i],crystallatt[i],crystal[i]),comments='') for i in range(nb_atoms)]
        # write pair interactions tables
        [[np.savetxt(potfile,rep[i,j,:],fmt="%.16e") for j in range(rep.shape[0]) if j <= i] for i in range(rep.shape[0])]
        potfile.close() 
    elif kind == "eam/fs":
        nb_atoms = len(atoms)
        # parameters unpacked
        potheader = "# Mixed EAM fs potential from :\n# %s \n# \n"%(source)
        # --- Writing new EAM alloy pot file --- #
        potfile = open(out_file,'wb')
        # write header and file parameters
        np.savetxt(potfile,atoms,fmt="%s",newline=' ', header=potheader+str(nb_atoms),footer='\n%i\t%e\t%i\t%e\t%e\n'%(Nrho,drho,Nr,dr,cutoff), comments='')
        # write F and f tables
        [np.savetxt(potfile,np.append(F[i,:],f[i,:,:].flatten()),fmt="%.16e",header='%i\t%f\t%f\t%s'%(atnumber[i],atmass[i],crystallatt[i],crystal[i]),comments='') for i in range(nb_atoms)]
        # write pair interactions tables
        [[np.savetxt(potfile,rep[i,j,:],fmt="%.16e") for j in range(rep.shape[0]) if j <= i] for i in range(rep.shape[0])]
        potfile.close() 
    else:
        print('Non supported eam file type')
        raise ValueError
