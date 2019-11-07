import ase.io
import io
import numpy as np
import os.path
import pytest
import subprocess

rel_err_norm_tol = 1.0e-10 # relative tolerance when comparing test data against reference

@pytest.fixture
def ref_npz(shared_datadir):
    """Provides 0.1 mM NaCl solution at 0.05 V across 100 nm cell reference data from binary npz file"""
    return np.load( os.path.join(shared_datadir,'NaCl.npz') )

@pytest.fixture
def ref_txt(shared_datadir):
    """Provides 0.1 mM NaCl solution at 0.05 V across 100 nm cell reference data from plain text file"""
    return np.loadtxt(os.path.join(shared_datadir,'NaCl.txt'), unpack=True)

@pytest.fixture
def ref_xyz(shared_datadir):
    """Provides 0.1 mM NaCl (15 Na, 15 Cl) solution at 0.05 V across
    [50,50,100] nm box reference sample from xyz file"""
    return ase.io.read(os.path.join(shared_datadir,'NaCl.xyz'))

@pytest.fixture
def ref_lammps(shared_datadir):
    """Provides 0.1 mM NaCl (15 Na, 15 Cl) solution at 0.05 V across
    [50,50,100] nm box reference sample from LAMMPS data file"""
    return ase.io.read(os.path.join(shared_datadir,'NaCl.lammps'),format='lammps-data')

def test_c2d_input_format_npz_output_format_xyz(tmpdir, shared_datadir, ref_xyz):
    """c2d NaCl.npz NaCl.xyz"""
    print("  RUN test_c2d_input_format_npz_output_format_xyz")
    subprocess.run(
        [ 'c2d', os.path.join(shared_datadir,'NaCl.npz'), 'NaCl.xyz' ],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=tmpdir)
    xyz = ase.io.read(os.path.join(tmpdir,'NaCl.xyz'))
    assert len(xyz) == len(ref_xyz)
    assert np.all(xyz.symbols == ref_xyz.symbols)
    assert np.all(xyz.get_initial_charges() == ref_xyz.get_initial_charges())
    assert np.all(xyz.cell == ref_xyz.cell)

def test_c2d_input_format_txt_output_format_xyz(tmpdir, shared_datadir, ref_xyz):
    """c2d NaCl.txt NaCl.xyz"""
    print("  RUN test_c2d_input_format_txt_output_format_xyz")
    subprocess.run(
        [ 'c2d', os.path.join(shared_datadir,'NaCl.txt'), 'NaCl.xyz' ],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=tmpdir)
    xyz = ase.io.read(os.path.join(tmpdir,'NaCl.xyz'))
    assert len(xyz) == len(ref_xyz)
    assert np.all(xyz.symbols == ref_xyz.symbols)
    assert np.all(xyz.get_initial_charges() == ref_xyz.get_initial_charges())
    assert np.all(xyz.cell == ref_xyz.cell)

def test_c2d_input_format_npz_output_format_lammps(tmpdir, shared_datadir, ref_lammps):
    """c2d NaCl.npz NaCl.lammps"""
    print("  RUN test_c2d_input_format_npz_output_format_lammps")
    subprocess.run(
        [ 'c2d', os.path.join(shared_datadir,'NaCl.npz'), 'NaCl.lammps' ],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=tmpdir)
    lammps = ase.io.read(os.path.join(tmpdir,'NaCl.lammps'),format='lammps-data')
    assert len(lammps) == len(ref_lammps)
    assert np.all(lammps.symbols == ref_lammps.symbols)
    assert np.all(lammps.get_initial_charges() == ref_lammps.get_initial_charges())
    assert np.all(lammps.cell == ref_lammps.cell)

def test_pnp_output_format_npz(tmpdir, ref_npz):
    """pnp -c 0.1 0.1 -u 0.05 -l 1.0e-7 -bc cell NaCl.npz"""
    print("  RUN test_pnp_output_format_npz")

    subprocess.run(
        ['pnp', '-c', '0.1', '0.1', '-z', '1', '-1',
            '-u', '0.05', '-l', '1.0e-7', '-bc', 'cell', 'out.npz'],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=tmpdir)

    test_npz = np.load(os.path.join(tmpdir,'out.npz'))

    ref_x = test_npz['x']
    ref_u = test_npz['u']
    ref_c = test_npz['c']

    dx = test_npz['x'] - ref_x
    du = test_npz['u'] - ref_u
    dc = test_npz['c'] - ref_c

    rel_err_norm_x = np.linalg.norm(dx)/np.linalg.norm(ref_x)
    rel_err_norm_u = np.linalg.norm(du)/np.linalg.norm(ref_u)
    rel_err_norm_c = np.linalg.norm(dc)/np.linalg.norm(ref_c)

    assert rel_err_norm_x < rel_err_norm_tol
    assert rel_err_norm_u < rel_err_norm_tol
    assert rel_err_norm_c < rel_err_norm_tol

def test_pnp_output_format_txt(tmpdir, ref_txt):
    """pnp -c 0.1 0.1 -u 0.05 -l 1.0e-7 -bc cell NaCl.txt"""
    print("  RUN test_pnp_output_format_txt")

    subprocess.run(
        ['pnp', '-c', '0.1', '0.1', '-z', '1', '-1',
            '-u', '0.05', '-l', '1.0e-7', '-bc', 'cell', 'out.txt'],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=tmpdir)

    test_txt = np.loadtxt(os.path.join(tmpdir,'out.txt'), unpack=True)

    ref_x = ref_txt[0,:]
    ref_u = ref_txt[1,:]
    ref_c = test_txt[2:,:]

    dx = test_txt[0,:] - ref_x
    du = test_txt[1,:] - ref_u
    dc = test_txt[2:,:] - ref_c

    rel_err_norm_x = np.linalg.norm(dx)/np.linalg.norm(ref_x)
    rel_err_norm_u = np.linalg.norm(du)/np.linalg.norm(ref_u)
    rel_err_norm_c = np.linalg.norm(dc)/np.linalg.norm(ref_c)

    assert rel_err_norm_x < rel_err_norm_tol
    assert rel_err_norm_u < rel_err_norm_tol
    assert rel_err_norm_c < rel_err_norm_tol

def test_pnp_c2d_pipeline_mode(tmpdir, ref_xyz):
    """pnp -c 0.1 0.1 -u 0.05 -l 1.0e-7 -bc cell | c2d > NaCl.xyz"""
    print("  RUN test_pnp_c2d_pipeline_mode")

    pnp = subprocess.Popen(
        ['pnp', '-c', '0.1', '0.1', '-z', '1', '-1',
            '-u', '0.05', '-l', '1.0e-7', '-bc', 'cell'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=tmpdir, encoding='utf-8')

    c2d = subprocess.Popen([ 'c2d' ],
        stdin=pnp.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=tmpdir, encoding='utf-8')
    pnp.stdout.close()  # Allow pnp to receive a SIGPIPE if p2 exits.

    assert c2d.wait() == 0
    assert pnp.wait() == 0

    c2d_output = c2d.communicate()[0]
    xyz_instream = io.StringIO(c2d_output)
    xyz = ase.io.read(xyz_instream, format='xyz')
    assert len(xyz) == len(ref_xyz)
    assert np.all(xyz.symbols == ref_xyz.symbols)
    assert np.all(xyz.get_initial_charges() == ref_xyz.get_initial_charges())
    assert np.all(xyz.cell == ref_xyz.cell)
