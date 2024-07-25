import os
import re
import copy
import pathlib
import pytest
import numpy as np

import ase.io
import ase.calculators.emt
import ase.calculators.lj

import matscipy.calculators.committee


@pytest.fixture
def committeemember():
    member = matscipy.calculators.committee.CommitteeMember(calculator=ase.calculators.emt.EMT())
    training_data = os.path.join(f'{os.path.dirname(__file__)}/committee_data/training_data_minimal.xyz')
    member.set_training_data(training_data)
    return member


@pytest.fixture
def committee_minimal(committeemember):
    committee = matscipy.calculators.committee.Committee(
        members=[copy.deepcopy(committeemember), copy.deepcopy(committeemember)]
    )
    return committee


@pytest.fixture
def committee():
    committee = matscipy.calculators.committee.Committee()
    committee_training_data = ase.io.read(f'{os.path.dirname(__file__)}/committee_data/training_data.xyz', ':')
    num_members = 10
    epsilons = np.linspace(0.98, 1.01, num_members)
    np.random.seed(123)
    np.random.shuffle(epsilons)
    for idx_i in range(num_members):
        committee += matscipy.calculators.committee.CommitteeMember(
            calculator=ase.calculators.lj.LennardJones(sigma=1, epsilon=epsilons[idx_i]),
            training_data=[atoms_i for atoms_i in committee_training_data
                           if idx_i in atoms_i.info['appears_in_committee']]
        )
    return committee


@pytest.fixture
def committee_calibrated(committee):
    committee.set_internal_validation_set(appearance_threshold=5)
    committee.calibrate(prop='energy', key='E_lj', location='info')
    committee.calibrate(prop='forces', key='F_lj', location='arrays')
    return committee


def test_committeemember_initialize():
    matscipy.calculators.committee.CommitteeMember(calculator=ase.calculators.emt.EMT())

    training_data = os.path.join(f'{os.path.dirname(__file__)}/committee_data/training_data_minimal.xyz')
    matscipy.calculators.committee.CommitteeMember(calculator=ase.calculators.emt.EMT(),
                                                   training_data=training_data)
    matscipy.calculators.committee.CommitteeMember(calculator=ase.calculators.emt.EMT(),
                                                   training_data=ase.io.read(training_data, ':'))


def test_committeemember_set_training_data(committeemember):
    training_data = os.path.join(f'{os.path.dirname(__file__)}/committee_data/training_data_minimal.xyz')

    with pytest.warns(Warning, match=re.escape('Overwriting current training data.')):
        committeemember.set_training_data(training_data)
    with pytest.warns(Warning, match=re.escape('Overwriting current training data.')):
        committeemember.set_training_data(pathlib.Path(training_data))
    with pytest.warns(Warning, match=re.escape('Overwriting current training data.')):
        committeemember.set_training_data(ase.io.read(training_data, ':'))


def test_committeemember_is_sample_in_atoms(committeemember):
    training_data = ase.io.read(os.path.join(f'{os.path.dirname(__file__)}/committee_data/training_data_minimal.xyz'), ':')
    test_data = ase.io.read(os.path.join(f'{os.path.dirname(__file__)}/committee_data/test_data.xyz'), ':')

    assert committeemember.is_sample_in_atoms(sample=training_data[0])
    with pytest.raises(RuntimeError,
                       match=re.escape('Can\'t test if `sample` is in `atoms`. '
                                       '`sample` has no Atoms.info[\'_Index_FullTrainingSet\']')):
        assert not committeemember.is_sample_in_atoms(sample=test_data[0])
    test_data[0].info['_Index_FullTrainingSet'] = -1
    assert not committeemember.is_sample_in_atoms(sample=test_data[0])


def test_committeemember_setter(committeemember):

    with pytest.raises(RuntimeError, match=re.escape('Use `set_training_data()` to modify the committee member')):
        committeemember.filename = ''

    with pytest.raises(RuntimeError, match=re.escape('Use `set_training_data()` to modify the committee member')):
        committeemember.atoms = []

    with pytest.raises(RuntimeError, match=re.escape('Use `set_training_data()` to modify the committee member')):
        committeemember.ids = []


def test_committee_initialize(committeemember):
    committee = matscipy.calculators.committee.Committee()
    expected_status = [
        ('members', []),
        ('number', 0),
        ('atoms', []),
        ('ids', []),
        ('alphas', {}),
        ('calibrated_for', set()),
    ]
    for attribute_i, value_i in expected_status:
        assert getattr(committee, attribute_i) == value_i
    with pytest.warns(Warning, match=re.escape('`Committee.set_internal_validation_set()` has not been called or '
                                               '`Committee`-instance has been altered since last call.')):
        assert getattr(committee, 'validation_set') == []

    member_0 = copy.deepcopy(committeemember)
    member_1 = copy.deepcopy(committeemember)
    committee = matscipy.calculators.committee.Committee(
        members=[member_0, member_1]
    )
    expected_status = [
        ('members', [member_0, member_1]),
        ('number', 2),
        ('atoms', member_0.atoms + member_1.atoms),
        ('ids', member_0.ids + member_1.ids),
        ('alphas', {}),
        ('calibrated_for', set()),
    ]
    for attribute_i, value_i in expected_status:
        assert getattr(committee, attribute_i) == value_i
    with pytest.warns(Warning, match=re.escape('`Committee.set_internal_validation_set()` has not been called or '
                                               '`Committee`-instance has been altered since last call.')):
        assert getattr(committee, 'validation_set') == []


def test_committee_member(committee_minimal):

    with pytest.raises(AssertionError,
                       match=re.escape('Members of `Committee` need to be of type `CommitteeMember`. Found ')):
        matscipy.calculators.committee.Committee(members=[0, 1])

    with pytest.raises(AssertionError,
                       match=re.escape('Members of `Committee` need to be of type `CommitteeMember`. Found ')):
        committee_minimal.members = [0, 1]

    with pytest.raises(AssertionError,
                       match=re.escape('Members of `Committee` need to be of type `CommitteeMember`. Found ')):
        committee_minimal.add_member(0)

    with pytest.raises(AssertionError,
                       match=re.escape('Members of `Committee` need to be of type `CommitteeMember`. Found ')):
        committee_minimal += 0


def test_committee_set_internal_validation_set(committee):

    with pytest.raises(AssertionError):
        committee.set_internal_validation_set(appearance_threshold=0)

    with pytest.raises(AssertionError):
        committee.set_internal_validation_set(appearance_threshold=committee.number - 1)

    committee.set_internal_validation_set(appearance_threshold=5)
    obtained = set([atoms_i.info['_Index_FullTrainingSet'] for atoms_i
                    in committee.validation_set])
    expected = set([atoms_i.info['_Index_FullTrainingSet'] for atoms_i
                    in ase.io.read(os.path.join(f'{os.path.dirname(__file__)}/committee_data/validation_set.xyz'), ':')])
    assert obtained == expected


def test_committee_calibrate(committee):
    committee.set_internal_validation_set(appearance_threshold=5)

    committee.calibrate(prop='energy', key='E_lj', location='info')
    assert committee.calibrated_for == set(['energy'])
    np.testing.assert_array_almost_equal(committee.alphas['energy'], 0.6295416920992463, decimal=6)

    committee.calibrate(prop='forces', key='F_lj', location='arrays')
    assert committee.calibrated_for == set(['energy', 'forces'])
    np.testing.assert_array_almost_equal(committee.alphas['forces'], 0.6195847443699875, decimal=6)

    with pytest.warns(Warning,
                      match=re.escape('`alphas` will be reset to avoid inconsistencies with new validation set.')):
        committee.set_internal_validation_set(appearance_threshold=4)
        assert committee.alphas == {}


def test_committee__calculate_alpha(committee):
    vals_ref = np.array([1.01, 1.02, 1.03])
    vals_pred = np.array([2.01, 1.02, 1.03])
    var_pred = np.array([1.01, 0.02, 0.03])

    obtained = committee._calculate_alpha(vals_ref, vals_pred, var_pred)
    np.testing.assert_array_almost_equal(obtained, 0.39584382766472004, decimal=6)


def test_committee_scale_uncertainty(committee):
    committee._alphas = {'energy': 2.5}

    assert committee.scale_uncertainty(2, 'energy') == 5.0


def test_committeeuncertainty_initialize(committee_calibrated):
    matscipy.calculators.committee.CommitteeUncertainty(committee=committee_calibrated)


def test_committeeuncertainty_calculate(committee_calibrated):
    calculator = matscipy.calculators.committee.CommitteeUncertainty(committee=committee_calibrated)
    test_data = ase.io.read(os.path.join(f'{os.path.dirname(__file__)}/committee_data/test_data.xyz'), ':')
    for atoms_i in test_data:
        calculator.calculate(atoms=atoms_i, properties=['energy', 'forces'])
        for prop_j in ['energy', 'forces']:
            # energy and forces are read into the results dictionary of a SinglePointCalculator
            np.testing.assert_array_almost_equal(calculator.results[prop_j], atoms_i.calc.results[prop_j], decimal=6,
                                                 err_msg=f'Missmatch in property \'{prop_j}\'')
        for prop_j in ['energy_uncertainty']:
            np.testing.assert_array_almost_equal(calculator.results[prop_j], atoms_i.info[prop_j], decimal=6,
                                                 err_msg=f'Missmatch in property \'{prop_j}\'')
        for prop_j in ['forces_uncertainty']:
            np.testing.assert_array_almost_equal(calculator.results[prop_j], atoms_i.arrays[prop_j], decimal=6,
                                                 err_msg=f'Missmatch in property \'{prop_j}\'')
