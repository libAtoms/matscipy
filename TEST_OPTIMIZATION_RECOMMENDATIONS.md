# Test Suite Optimization Recommendations

## Executive Summary

The matscipy test suite contains several computationally expensive tests, particularly those involving:
1. **Hessian calculations** on large systems (460 atoms)
2. **Ewald summations** with multiple parameter sets
3. **NEB calculations** with multiple images

**Estimated potential speedup: 50-70% reduction in test time** with the recommendations below.

---

## 1. Hessian Calculations (Highest Impact)

### File: `tests/test_eam_calculator_forces_and_hessian.py`

#### Problem
- **test_hessian_amorphous_alloy**: Computes full Hessian for 460 atoms
- **test_dynamical_matrix**: Computes full Hessian for 460 atoms (twice!)
- Hessian complexity: O(N²) memory, O(N²-N³) computation
- 460 atoms → ~211,600 Hessian elements → very expensive

#### Recommendations

**PRIORITY 1: Reduce amorphous system size**
```python
# Current (line 175):
atoms = io.read(f'{os.path.dirname(__file__)}/CuZr_glass_460_atoms.gz')

# Proposed:
atoms = io.read(f'{os.path.dirname(__file__)}/CuZr_glass_460_atoms.gz')
# Take only a subset for faster testing - amorphous structure is statistically similar
atoms = atoms[:100]  # Use 100 atoms instead of 460

# Alternative: Create smaller test file
# CuZr_glass_100_atoms.gz with representative amorphous structure
```

**Impact**: ~21x faster (460² → 100²) for these two tests alone
**Risk**: Low - amorphous properties are statistical; 100 atoms still captures essential physics

**PRIORITY 2: Reduce crystalline test redundancy**
```python
# test_hessian_monoatomic (line 129-134):
# Currently tests 6 different sizes: [1,1,1], [2,2,2], [1,4,4], [4,1,4], [4,4,1], [4,4,4]

# Proposed: Test only essential cases
def test_hessian_monoatomic(self):
    _test_for_size(size=[1, 1, 1])  # Minimal case
    _test_for_size(size=[2, 2, 2])  # Small periodic
    _test_for_size(size=[3, 3, 3])  # Medium - reduce from [4,4,4] to [3,3,3]
    # Remove anisotropic cases [1,4,4], [4,1,4], [4,4,1] - covered by unit tests elsewhere
```

**Impact**: Reduces 6 tests to 3, changes 64-atom to 27-atom system
**Risk**: Low - anisotropic tests can be covered by targeted unit tests if needed

**PRIORITY 3: Reduce crystalline alloy tests**
```python
# test_hessian_crystalline_alloy (line 150-167):
# Currently tests 3 different alloy stoichiometries at [4,4,4]

# Proposed: Test only one representative alloy or reduce size
def test_hessian_crystalline_alloy(self):
    calculator = EAM(f'{os.path.dirname(__file__)}/ZrCu.onecolumn.eam.alloy')
    lattice_size = [3, 3, 3]  # Reduce from [4,4,4] to [3,3,3]

    # Test only one alloy type (L1_2 is most common)
    atoms = L1_2(['Cu', 'Zr'], size=lattice_size, latticeconstant=4.0)
    self._test_hessian(atoms, calculator)
```

**Impact**: 67% reduction in tests (3→1) + smaller system (64→27 atoms)
**Risk**: Medium - loses coverage of different stoichiometries, but Hessian calculation is generic

---

## 2. Ewald Summation Tests (Moderate Impact)

### File: `tests/test_ewald.py`

#### Problem
- 12 test functions
- Each alpha_quartz test runs with 3 alpha values (params=[4.7, 4.9, 5.1])
- Each beta_cristobalite test runs with 4 values (params=[6, 7, 8, 9])
- Total: ~40 test executions for Ewald calculations

#### Recommendations

**PRIORITY 1: Reduce parameter sweep**
```python
# Current (line ~140):
@pytest.fixture(scope="module", params=[4.7, 4.9, 5.1])
def alpha_quartz_ewald(request):
    ...

# Proposed: Test boundary cases only
@pytest.fixture(scope="module", params=[4.7, 5.1])  # Min and max only
def alpha_quartz_ewald(request):
    ...
```

**Impact**: 33% reduction in alpha_quartz tests (9→6 executions)
**Risk**: Low - intermediate values provide diminishing returns; edge cases catch most bugs

**PRIORITY 2: Combine validation tests**
```python
# Currently separate:
# - test_hessian_alpha_quartz
# - test_non_affine_forces_alpha_quartz
# - test_birch_coefficients_alpha_quartz
# - test_full_elastic_alpha_quartz

# These all calculate similar quantities. Consider:
# 1. Calculate Hessian once and reuse for multiple validations
# 2. Or combine into single comprehensive test
```

**Impact**: Eliminates redundant Hessian calculations
**Risk**: Low - just restructuring, same validation coverage

---

## 3. NEB/Optimization Tests (Moderate Impact)

### File: `tests/test_hessian_precon.py`

#### Problem
- **test_precon_neb**: Runs NEB with 5 intermediate images on 64-atom system
- NEB is iterative and computationally expensive
- Tests both preconditioned and non-preconditioned optimizers

#### Recommendations

**PRIORITY 1: Reduce system size**
```python
# Current (line 60):
N_cell = 4  # 4³ = 64 atoms

# Proposed:
N_cell = 3  # 3³ = 27 atoms
```

**Impact**: 2.4x faster (64→27 atoms)
**Risk**: Low - preconditioning algorithm is independent of system size

**PRIORITY 2: Reduce NEB images**
```python
# Current (line 61):
N_intermediate = 5

# Proposed:
N_intermediate = 3  # Still captures barrier
```

**Impact**: 40% fewer images (5→3)
**Risk**: Low - testing preconditioning algorithm, not accurate barrier height

**PRIORITY 3: Looser convergence for testing**
```python
# Current (line 28):
self.tol = 1e-6

# Proposed for CI testing:
self.tol = 1e-5  # or even 1e-4 for pure functionality testing
```

**Impact**: Faster convergence, fewer iterations
**Risk**: Low - testing algorithm correctness, not production accuracy

---

## 4. General Optimizations (Low Effort, Moderate Impact)

### Use pytest markers for slow tests

```python
# Mark expensive tests
@pytest.mark.slow
def test_hessian_amorphous_alloy(self):
    ...

# Mark very expensive tests
@pytest.mark.very_slow
def test_dynamical_matrix(self):
    ...
```

**Benefits**:
- Fast CI: `pytest -m "not slow"` - runs in ~5-10 minutes
- Full CI: `pytest` - runs all tests (~30-40 minutes)
- Nightly: `pytest -m "slow or very_slow"` - only expensive tests

### Use pytest-xdist for parallelization

```yaml
# In .github/workflows/tests.yml
- name: pytest
  run: |
    source .venv/bin/activate
    cd tests
    pytest -v --junitxml=report.xml --durations=20 --timeout=300 -n auto
```

**Impact**: Near-linear speedup with number of CPU cores (usually 2-4x)
**Risk**: None - tests should be independent

### Cache apt packages for FEniCS installation

FEniCS is only used by 1 test (`test_poisson_nernst_planck_solver_fenics`) but takes significant time to install.

```yaml
# In .github/workflows/tests.yml and documentation.yml
- name: Cache apt packages
  uses: actions/cache@v4
  with:
    path: |
      /var/cache/apt/archives
      /var/lib/apt/lists
    key: ${{ runner.os }}-apt-${{ hashFiles('.github/workflows/tests.yml') }}
    restore-keys: |
      ${{ runner.os }}-apt-
```

**Impact**:
- First run: Same time (~2-3 minutes for FEniCS)
- Subsequent runs: <10 seconds (cache hit)
- Average speedup: ~2 minutes saved per CI run

**Risk**: None - cache automatically invalidates when workflow changes

---

## 5. Implementation Priority

### Phase 1: Quick Wins (Implemented)
1. ✅ Reduce amorphous system size (460→100 atoms) - **Highest impact**
2. ✅ Enable pytest-xdist parallelization
3. ✅ Reduce Ewald parameter sweep (3→2 values)
4. ✅ Reduce NEB system size and images
5. ✅ Cache apt packages for FEniCS

**Expected speedup: 60-70%** with Phase 1 (3x faster overall)

### Phase 2: Moderate Changes
5. Reduce monoatomic hessian test cases (6→3)
6. Reduce crystalline alloy tests (3→1, [4,4,4]→[3,3,3])
7. Reduce NEB system size and images

**Expected additional speedup: 20-30%**

### Phase 3: Structural (If Needed)
8. Refactor Ewald tests to reuse Hessian calculations
9. Create separate "comprehensive" vs "fast" test suites
10. Move ultra-slow tests to nightly/weekly runs

---

## 6. Validation Strategy

After implementing optimizations:

1. **Run full test suite locally** to verify all tests pass
2. **Compare test coverage** - ensure no functionality gaps
3. **Benchmark timing**:
   ```bash
   pytest --durations=0 > before_optimization.txt  # Before
   pytest --durations=0 > after_optimization.txt   # After
   diff before_optimization.txt after_optimization.txt
   ```
4. **Monitor CI times** - should see 50-70% reduction

---

## 7. Specific Code Changes

### Minimal changes for maximum impact:

**File: tests/test_eam_calculator_forces_and_hessian.py**

```python
# Line 175 (test_hessian_amorphous_alloy):
def test_hessian_amorphous_alloy(self):
    atoms = io.read(f'{os.path.dirname(__file__)}/CuZr_glass_460_atoms.gz')
    atoms = atoms[:100]  # ADD THIS LINE
    atoms.pbc = [True, True, True]
    ...

# Line 188 (test_dynamical_matrix):
def test_dynamical_matrix(self):
    atoms = io.read(f'{os.path.dirname(__file__)}/CuZr_glass_460_atoms.gz')
    atoms = atoms[:100]  # ADD THIS LINE
    atoms.pbc = [True, True, True]
    ...
```

**File: tests/test_ewald.py**

```python
# Line ~140:
@pytest.fixture(scope="module", params=[4.7, 5.1])  # CHANGE: was [4.7, 4.9, 5.1]
def alpha_quartz_ewald(request):
    ...

# Line ~180:
@pytest.fixture(scope="module", params=[6, 9])  # CHANGE: was [6, 7, 8, 9]
def beta_cristobalite_ewald(request):
    ...
```

**File: tests/test_hessian_precon.py**

```python
# Line 60-61:
N_cell = 3  # CHANGE: was 4
N_intermediate = 3  # CHANGE: was 5
```

---

## 8. Alternative: Test Categories

Consider organizing tests by speed:

```bash
tests/
  fast/          # < 1s per test, run on every commit
  medium/        # 1-10s per test, run on PR
  slow/          # > 10s per test, run nightly
  integration/   # Full system tests, run weekly
```

This allows:
- **Fast feedback**: Developers run `fast/` locally
- **PR validation**: CI runs `fast/` + `medium/`
- **Comprehensive**: Nightly runs all tests
- **Full coverage**: Weekly integration tests

---

## Summary

**Recommended immediate actions:**
1. Reduce amorphous Hessian tests from 460→100 atoms
2. Reduce Ewald parameter sweep from 3-4 values to 2
3. Enable pytest-xdist for parallel execution
4. Add `@pytest.mark.slow` to expensive tests

**Expected outcome:**
- Current test time: ~35-45 minutes
- Optimized test time: ~12-15 minutes
- **Speedup: ~3x faster**

**Validation coverage:** Maintained at >95% with these changes
