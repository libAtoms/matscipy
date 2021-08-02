#
# Copyright 2021 Lars Pastewka (U. Freiburg)
#           2019-2021 Johannes Hoermann (U. Freiburg)
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
import ase.io
import io
import matscipytest
import numpy as np
import os, os.path
import shutil
import stat
import subprocess
import tempfile
import unittest

from distutils.version import LooseVersion

class c2dCliTest(matscipytest.MatSciPyTestCase):
    """Tests c2d and pnp command line interfaces"""

    def setUp(self):
        """Reads reference data files"""
        self.test_path = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.test_path, 'electrochemistry_data')

        # Provides 0.1 mM NaCl solution at 0.05 V across 100 nm cell reference
        # data from binary npz file
        self.ref_npz =  np.load( os.path.join(self.data_path,'NaCl.npz') )

        # Provides 0.1 mM NaCl solution at 0.05 V across 100 nm cell reference
        # data from plain text file
        self.ref_txt = np.loadtxt(os.path.join(self.data_path,'NaCl.txt'),
            unpack=True)

        # Provides 0.1 mM NaCl (15 Na, 15 Cl) solution at 0.05 V across
        # [50,50,100] nm box reference sample from xyz file
        self.ref_xyz = ase.io.read(os.path.join(self.data_path,'NaCl.xyz'))

        # Provides 0.1 mM NaCl (15 Na, 15 Cl) solution at 0.05 V across
        # [50,50,100] nm box reference sample from LAMMPS data file
        self.ref_lammps = ase.io.read(
            os.path.join(self.data_path,'NaCl.lammps'),format='lammps-data')

        # command line interface scripts are expected to reside within
        # ../matscipy/cli/electrochemistry relative to this test directory
        self.pnp_cli = os.path.join(
            self.test_path,os.path.pardir,'matscipy','cli','electrochemistry','pnp.py')
        self.c2d_cli = os.path.join(
            self.test_path,os.path.pardir,'matscipy','cli','electrochemistry','c2d.py')

        self.assertTrue(os.path.exists(self.pnp_cli))
        self.assertTrue(os.path.exists(self.c2d_cli))

        self.bin_path = tempfile.TemporaryDirectory()

        # mock callable executables on path:
        pnp_exe = os.path.join(self.bin_path.name,'pnp')
        c2d_exe = os.path.join(self.bin_path.name,'c2d')
        shutil.copyfile(self.pnp_cli,pnp_exe)
        shutil.copyfile(self.c2d_cli,c2d_exe)
        st = os.stat(pnp_exe)
        os.chmod(pnp_exe, st.st_mode | stat.S_IEXEC)
        st = os.stat(c2d_exe)
        os.chmod(c2d_exe, st.st_mode | stat.S_IEXEC)
        self.myenv = os.environ.copy()
        self.myenv["PATH"] = os.pathsep.join((
            self.bin_path.name, self.myenv["PATH"]))

    def tearDown(self):
        self.bin_path.cleanup()

    def test_c2d_input_format_npz_output_format_xyz(self):
        """c2d NaCl.npz NaCl.xyz"""
        print("  RUN test_c2d_input_format_npz_output_format_xyz")
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                [ 'c2d', os.path.join(self.data_path,'NaCl.npz'), 'NaCl.xyz' ],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=tmpdir, env=self.myenv)
            xyz = ase.io.read(os.path.join(tmpdir,'NaCl.xyz'), format='extxyz')
            self.assertEqual(len(xyz),len(self.ref_xyz))
            self.assertTrue( ( xyz.symbols == self.ref_xyz.symbols ).all() )
            self.assertTrue( ( xyz.get_initial_charges() ==
                self.ref_xyz.get_initial_charges() ).all() )
            self.assertTrue( ( xyz.cell == self.ref_xyz.cell ).all() )

    def test_c2d_input_format_txt_output_format_xyz(self):
        """c2d NaCl.txt NaCl.xyz"""
        print("  RUN test_c2d_input_format_txt_output_format_xyz")
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                [ 'c2d', os.path.join(self.data_path,'NaCl.txt'), 'NaCl.xyz' ],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=tmpdir, env=self.myenv)
            xyz = ase.io.read(os.path.join(tmpdir,'NaCl.xyz'), format='extxyz')
            self.assertEqual(len(xyz), len(self.ref_xyz))
            self.assertTrue( ( xyz.symbols == self.ref_xyz.symbols ).all() )
            self.assertTrue( ( xyz.get_initial_charges()
                == self.ref_xyz.get_initial_charges() ).all() )
            self.assertTrue( ( xyz.cell == self.ref_xyz.cell ).all() )

    @unittest.skipUnless(LooseVersion(ase.__version__) > LooseVersion('3.19.0'),
        """ LAMMPS data file won't work for ASE version up until 3.18.1,
            LAMMPS data file input broken in ASE 3.19.0, skipped""")
    def test_c2d_input_format_npz_output_format_lammps(self):
        """c2d NaCl.npz NaCl.lammps"""
        print("  RUN test_c2d_input_format_npz_output_format_lammps")
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                [ 'c2d', os.path.join(self.data_path,'NaCl.npz'), 'NaCl.lammps' ],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=tmpdir, env=self.myenv)
            lammps = ase.io.read(
                os.path.join(tmpdir,'NaCl.lammps'),format='lammps-data')
            self.assertEqual(len(lammps), len(self.ref_lammps))
            self.assertTrue( ( lammps.symbols
                ==  self.ref_lammps.symbols ).all() )
            self.assertTrue( ( lammps.get_initial_charges()
                ==  self.ref_lammps.get_initial_charges() ).all() )
            self.assertTrue( ( lammps.cell ==  self.ref_lammps.cell ).all() )

    def test_pnp_output_format_npz(self):
        """pnp -c 0.1 0.1 -u 0.05 -l 1.0e-7 -bc cell NaCl.npz"""
        print("  RUN test_pnp_output_format_npz")
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                ['pnp', '-c', '0.1', '0.1', '-z', '1', '-1',
                    '-u', '0.05', '-l', '1.0e-7', '-bc', 'cell', 'out.npz'],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=tmpdir, env=self.myenv)

            test_npz = np.load(os.path.join(tmpdir,'out.npz'))

            # depending on which solver is used for the tests, absolute values might deviate more than the default tolerance
            self.assertArrayAlmostEqual(test_npz['x'], self.ref_npz['x'], tol=1e-7)
            self.assertArrayAlmostEqual(test_npz['u'], self.ref_npz['u'], tol=1e-5)
            self.assertArrayAlmostEqual(test_npz['c'], self.ref_npz['c'], tol=1e-3)

    def test_pnp_output_format_txt(self):
        """pnp -c 0.1 0.1 -u 0.05 -l 1.0e-7 -bc cell NaCl.txt"""
        print("  RUN test_pnp_output_format_txt")
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                ['pnp', '-c', '0.1', '0.1', '-z', '1', '-1',
                    '-u', '0.05', '-l', '1.0e-7', '-bc', 'cell', 'out.txt'],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=tmpdir, env=self.myenv)

            test_txt = np.loadtxt(os.path.join(tmpdir,'out.txt'), unpack=True)

            # depending on which solver is used for the tests, absolute values might deviate more than the default tolerance
            self.assertArrayAlmostEqual(test_txt[0,:], self.ref_txt[0,:], tol=1e-5) # x
            self.assertArrayAlmostEqual(test_txt[1,:],self.ref_txt[1,:], tol=1e-5) # u
            self.assertArrayAlmostEqual(test_txt[2:,:],self.ref_txt[2:,:], tol=1e-3) # c

    def test_pnp_c2d_pipeline_mode(self):
        """pnp -c 0.1 0.1 -u 0.05 -l 1.0e-7 -bc cell | c2d > NaCl.xyz"""
        print("  RUN test_pnp_c2d_pipeline_mode")
        with tempfile.TemporaryDirectory() as tmpdir:
            pnp = subprocess.Popen(
                ['pnp', '-c', '0.1', '0.1', '-z', '1', '-1',
                    '-u', '0.05', '-l', '1.0e-7', '-bc', 'cell'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=tmpdir, encoding='utf-8', env=self.myenv)

            c2d = subprocess.Popen([ 'c2d' ],
                stdin=pnp.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=tmpdir, encoding='utf-8', env=self.myenv)
            pnp.stdout.close()  # Allow pnp to receive a SIGPIPE if p2 exits.

            self.assertEqual(c2d.wait(),0)
            self.assertEqual(pnp.wait(),0)

            c2d_output = c2d.communicate()[0]
            with io.StringIO(c2d_output) as xyz_instream:
                xyz = ase.io.read(xyz_instream, format='extxyz')
            self.assertEqual(len(xyz),len(self.ref_xyz))
            self.assertTrue( ( xyz.symbols == self.ref_xyz.symbols ).all() )
            self.assertTrue( ( xyz.get_initial_charges()
                == self.ref_xyz.get_initial_charges() ).all() )
            self.assertTrue( ( xyz.cell == self.ref_xyz.cell ).all() )
