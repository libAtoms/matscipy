import warnings
import logging
from pathlib import Path
from collections import Counter

import numpy as np
import ase.io
from ase.calculators.calculator import Calculator, all_changes


logger = logging.getLogger('matscipy.calculators.committee')


class CommitteeUncertainty(Calculator):
    """
    Calculator for a committee of machine learned interatomic potentials (MLIP).

    The class assumes individual members of the committee already exist (i.e. their
    training is performed externally). Instances of this class are initialized with
    these committee members and results (energy, forces) are calculated as average
    over these members. In addition to these values, also the uncertainty (standard
    deviation) is calculated.

    The idea for this Calculator class is based on the following publication:
    Musil et al., J. Chem. Theory Comput. 15, 906−915 (2019)
    https://pubs.acs.org/doi/full/10.1021/acs.jctc.8b00959

    Parameter:
    ----------
    committee: Committee-instance
        Representation for a collection of Calculators.
    atoms : ase-Atoms, optional default=None
        Optional object to which the calculator will be attached.
    """

    def __init__(self, committee, atoms=None):

        self.implemented_properties = ['energy', 'forces', 'stress']

        self.committee = committee

        super().__init__(atoms=atoms)

        logger.info('Initialized committee uncertainty calculator')
        for line_i in self.committee.__repr__().splitlines():
            logger.debug(line_i)

    def calculate(self, atoms=None, properties=['energy', 'forces', 'stress'], system_changes=all_changes):
        """Calculates committee (mean) values and variances."""

        # ACEHAL compatibility: Ignore ACEHAL explicit requests for committee_energy etc
        properties = [p for p in properties if "committee" not in p]

        logger.info(f'Calculating properties {properties} with committee')
        super().calculate(atoms, properties, system_changes)

        property_committee = {k_i: [] for k_i in properties}
        self.results_extra = {}

        for cm_i in self.committee.members:
            cm_i.calculator.calculate(atoms=atoms, properties=properties, system_changes=system_changes)

            for p_i in properties:
                property_committee[p_i].append(cm_i.calculator.results[p_i])

        for p_i in properties:
            self.results[p_i] = np.mean(property_committee[p_i], axis=0)
            self.results[f'{p_i}_uncertainty'] = np.sqrt(np.var(property_committee[p_i], ddof=1, axis=0))

            # Compatibility with ACEHAL Bias Calculator
            # https://github.com/ACEsuit/ACEHAL/blob/main/ACEHAL/bias_calc.py
            self.results_extra[f'{p_i}_committee'] = property_committee[p_i]
            self.results_extra[f'err_{p_i}'] = self.results[f'{p_i}_uncertainty']
            self.results_extra[f'err_{p_i}_MAE'] = np.average([np.abs(comm - self.results[p_i]) for comm in property_committee[p_i]])

            if self.committee.is_calibrated_for(p_i):
                self.results[f'{p_i}_uncertainty'] = self.committee.scale_uncertainty(self.results[f'{p_i}_uncertainty'], p_i)
            else:
                warnings.warn(f'Uncertainty estimation has not been calibrated for {p_i}.')


class Committee:
    """
    Instances of this class represent a committee of models.

    It's use is to store the ```CommitteeMember```s representing the committee model
    and to calibrate the obtained uncertainties (required when sub-sampling is used
    to create the training data of the committee members).

    Parameter:
    ----------
    members: list(M)
        List of ```CommitteeMember``` instances representing the committee of (here `M`) models.
    """
    def __init__(self, members=None):
        self.members = [] if members is None else members

        logger.info('Initialized committee')
        for line_i in self.__repr__().splitlines():
            logger.debug(line_i)

    @property
    def members(self):
        """List with committee members."""
        return self._members

    @members.setter
    def members(self, members):
        """Set list with committee members."""
        for member_i in members:
            self._check_member_type(member_i)
        self._members = members
        logger.info(f'Set {len(self.members)} members to represent the committee')
        self._update()

    @property
    def number(self):
        """Number of committee members."""
        return self._number

    @property
    def atoms(self):
        """Combined Atoms/samples in the committee."""
        return self._atoms

    @property
    def ids(self):
        """Identifiers of atoms/samples in the committee."""
        return self._ids

    @property
    def id_to_atoms(self):
        """Dictionary to translate identifiers to Atoms-objects."""
        return self._id_to_atoms

    @property
    def id_counter(self):
        """Counter-object for identifier appearances in the committee."""
        return self._id_counter

    @property
    def alphas(self):
        """(Linear) scaling factors for committee uncertainties."""
        return self._alphas

    @property
    def calibrated_for(self):
        """Set of properties the committee has been calibrated for."""
        return self.alphas.keys()

    @property
    def validation_set(self):
        """List of Atoms-objects."""
        if not self._validation_set:
            msg = '`Committee.set_internal_validation_set()` has not been called or ' + \
                  '`Committee`-instance has been altered since last call.'
            logger.warning(msg)
            warnings.warn(msg)
        return self._validation_set

    def _update(self):
        """Update status when ```Committee```-instance has been altered."""
        self._number = len(self.members)
        self._atoms = [atoms_ij for cm_i in self.members for atoms_ij in cm_i.atoms]
        self._ids = [id_ij for cm_i in self.members for id_ij in cm_i.ids]
        self._id_to_atoms = {id_i: atoms_i for id_i, atoms_i in zip(self.ids, self.atoms)}
        self._id_counter = Counter(self.ids)
        self._validation_set = []
        self._alphas = {}
        logger.info('Updated committee status')

    def add_member(self, member):
        """Extend committee by new ```member``` (i.e. ```CommitteeMember```-instance)."""
        self._check_member_type(member)
        self.members.append(member)
        logger.info('Added +1 member to the committee')
        self._update()

    def __add__(self, member):
        """Extend committee by new ```member``` (i.e. ```CommitteeMember```-instance)."""
        self._check_member_type(member)
        self.add_member(member)
        return self

    @staticmethod
    def _check_member_type(member):
        """Make sure ```member``` is of type ```CommitteeMember```."""
        assert isinstance(member, CommitteeMember), \
            f'Members of `Committee` need to be of type `CommitteeMember`. Found {type(member)}'

    def set_internal_validation_set(self, appearance_threshold):
        """
        Define a validation set based on the Atoms-objects of sub-sampled committee training sets.

        Parameter:
        ----------
        appearance_threshold: int
            Number of times a sample for the validation set
            is maximally allowed to appear across the training sets
            of committee members.
        """
        if self._alphas:
            msg = '`alphas` will be reset to avoid inconsistencies with new validation set.'
            logger.warning(msg)
            warnings.warn(msg)
            self._reset_calibration_parameters()

        assert 0 < appearance_threshold <= self.number - 2

        self._validation_set = []
        for id_i, appearance_i in self.id_counter.most_common()[::-1]:
            if appearance_i > appearance_threshold:
                break
            self._validation_set.append(self.id_to_atoms[id_i])

        logger.info(f'Set internal validation set with {len(self.validation_set)} entries')

    def _reset_calibration_parameters(self):
        """Reset parameters obtained from calling ```self.calibrate()```."""
        self._alphas = {}
        logger.info('Reset calibration parameters')

    def calibrate(self, prop, key, location, system_changes=all_changes):
        """
        Obtain parameters to properly scale committee uncertainties and make
        them available as an attribute (```alphas```) with another associated
        attribute (```calibrated_for```) providing information about the property
        for which the uncertainty will be scaled by it.

        Parameter:
        ----------
        properties: list(str)
            Properties for which the calibration will determine scaling factors.
        key: str
            Key under which the reference values in the validation set are stored
            (i.e. under Atoms.info[```key```] / Atoms.arrays[```key```]).
        location: str
            Either 'info' or 'arrays'.
        """
        assert location in ['info', 'arrays'], f'`location` must be \'info\' or \'arrays\', not \'{location}\'.'

        validation_ref = [np.asarray(getattr(sample_i, location)[key]).flatten() for sample_i in self.validation_set]
        validation_pred, validation_pred_var = [], []

        for idx_i, sample_i in enumerate(self.validation_set):

            sample_committee_pred = []

            for cm_j in self.members:

                if cm_j.is_sample_in_atoms(sample_i):
                    continue

                cm_j.calculator.calculate(atoms=sample_i, properties=[prop], system_changes=system_changes)
                sample_committee_pred.append(cm_j.calculator.results[prop])

            validation_pred.append(np.mean(sample_committee_pred, axis=0).flatten())
            validation_pred_var.append(np.var(sample_committee_pred, ddof=1, axis=0).flatten())

        # For symmetry-reasons it can happen that e.g. all values for a force component of an atom are equal.
        # This would lead to a division-by-zero error in self._calculate_alpha() due to zero-variances.
        validation_ref = np.concatenate(validation_ref)
        validation_pred = np.concatenate(validation_pred)
        validation_pred_var = np.concatenate(validation_pred_var)
        ignore_indices = np.where(validation_pred_var == 0)[0]
        validation_ref = np.delete(validation_ref, ignore_indices)
        validation_pred = np.delete(validation_pred, ignore_indices)
        validation_pred_var = np.delete(validation_pred_var, ignore_indices)

        self._alphas.update(
                {prop: self._calculate_alpha(
                    vals_ref=validation_ref,
                    vals_pred=validation_pred,
                    vars_pred=validation_pred_var,
                    )
                 })

        logger.info(f'Calibrated committee for property \'{prop}\'')
        logger.debug(f'\talpha = {self.alphas[prop]}')

    def is_calibrated_for(self, prop):
        """Check whether committee has been calibrated for ```prop```."""
        return prop in self.calibrated_for

    def _calculate_alpha(self, vals_ref, vals_pred, vars_pred):
        """
        Get (linear) uncertainty scaling factor alpha.

        This implementation is based on:
        Imbalzano et al., J. Chem. Phys. 154, 074102 (2021)
        https://doi.org/10.1063/5.0036522

        Parameter:
        ----------
        vals_ref: ndarray(N)
            Reference values for validation set samples.
        vals_pred: ndarray(N)
            Values predicted by the committee for validation set samples.
        vars_pred: ndarray(N)
            Variance predicted by the committee for validation set samples.

        Returns:
        --------
        (Linear) uncertainty scaling factor alpha.
        """
        N_val = len(vals_ref)
        M = self.number
        alpha_squared = -1/M + (M - 3)/(M - 1) * 1/N_val * np.sum(np.power(vals_ref-vals_pred, 2) / vars_pred)
        logger.info(f'Calculated alpha')
        logger.debug(f'\tN_val          = {N_val}')
        logger.debug(f'\tM              = {M}')
        logger.debug(f'\talpha_squared  = {alpha_squared}')
        assert alpha_squared > 0, f'Obtained negative value for `alpha_squared`: {alpha_squared}'
        return np.sqrt(alpha_squared)

    def scale_uncertainty(self, value, prop):
        """
        Scale uncertainty ```value``` obtained with the committee according to the calibration
        for the corresponding property (```prop```).

        Parameter:
        ----------
        value: float / ndarray
            Represents the uncertainty values (e.g. energy, forces) to be scaled.
        prop: str
            The property associated with ```value``` (for which the committee needs to be calibrated).

        Returns:
        --------
        Scaled input ```value```.
        """
        return self.alphas[prop] * value

    def __repr__(self):
        s = ''

        s_i = f'Committee Status\n'
        s += s_i
        s += '='*len(s_i) + '\n\n'

        s += f'# members:                    {self.number:>10d}\n'
        s += f'# atoms:                      {len(self.atoms):>10d}\n'
        s += f'# ids:                        {len(self.ids):>10d}\n'
        s += f'# atoms validation set:       {len(self._validation_set):>10d}\n'
        if not self.calibrated_for:
            s += f'calibrated for:               {"-":>10}\n'
        else:
            s += f'calibrated for:\n'
            for p_i in sorted(self.calibrated_for):
                s += f'{"":>4s}{p_i:<18}{self.alphas[p_i]:>18}\n'

        for idx_i, cm_i in enumerate(self.members):
            s += '\n\n'
            s_i = f'Committee Member {idx_i}:\n'
            s += s_i
            s += '-'*len(s_i) + '\n'
            s += cm_i.__repr__()

        return s


class CommitteeMember:
    """
    Lightweight class defining a member (i.e. a sub-model) of a committee model.

    Parameter:
    ----------
    calculator: Calculator
        Instance of a Calculator-class (or heirs e.g. quippy.potential.Potential)
        representing a machine-learned model.
    training_data: str / Path / list(Atoms), optional default=None
        Path to or Atoms of (sub-sampled) training set used to create the machine-learned model
        defined by the ```calculator```.
    """
    def __init__(self, calculator, training_data=None):
        self._calculator = calculator

        self._filename = 'no filename'
        self._atoms = []
        self._ids = []

        if training_data is not None:
            self.set_training_data(training_data)

        logger.info('Created committee member')
        for line_i in self.__repr__().splitlines():
            logger.debug(line_i)

    @property
    def calculator(self):
        """Model of the committee member."""
        return self._calculator

    @property
    def filename(self):
        """Path to the atoms/samples in the committee member."""
        return self._filename

    @filename.setter
    def filename(self, filename):
        """Set path to the atoms/samples in the committee member."""
        msg = 'Use `set_training_data()` to modify the committee member'
        logger.error(msg)
        raise RuntimeError(msg)

    @property
    def atoms(self):
        """Atoms/samples in the committee member."""
        return self._atoms

    @atoms.setter
    def atoms(self, atoms):
        """Set Atoms/samples in the committee member."""
        msg = 'Use `set_training_data()` to modify the committee member'
        logger.error(msg)
        raise RuntimeError(msg)

    @property
    def ids(self):
        """Identifiers of atoms/samples in the committee member."""
        return self._ids

    @ids.setter
    def ids(self, ids):
        """Set identifiers of atoms/samples in the committee member."""
        msg = 'Use `set_training_data()` to modify the committee member'
        logger.error(msg)
        raise RuntimeError(msg)

    def set_training_data(self, training_data):
        """
        Read in and store the training data of this committee members from the passed ```filename```.

        Parameter:
        ----------
        training_data: str / Path / list(Atoms), optional default=None
            Path to or Atoms of (sub-sampled) training set used to create the machine-learned model
            defined by the ```calculator```. Individual Atoms need an Atoms.info['_Index_FullTrainingset']
            for unique identification.
        """
        if len(self.atoms) > 0:
            msg = 'Overwriting current training data.'
            logger.warning(msg)
            warnings.warn(msg)

        if isinstance(training_data, (str, Path)):
            self._filename = Path(training_data)
            self._atoms = ase.io.read(self.filename, ':')
        elif isinstance(training_data, list):
            self._filename = 'No Filename'
            self._atoms = training_data
        self._ids = [atoms_i.info['_Index_FullTrainingSet'] for atoms_i in self.atoms]

    def is_sample_in_atoms(self, sample):
        """Check if passed Atoms-object is part of this committee member (by comparing identifiers)."""
        if '_Index_FullTrainingSet' not in sample.info:
            msg = 'Can\'t test if `sample` is in `atoms`. `sample` has no Atoms.info[\'_Index_FullTrainingSet\']'
            logger.error(msg)
            raise RuntimeError(msg)
        else:
            return sample.info['_Index_FullTrainingSet'] in self.ids

    def __repr__(self):
        s = ''
        s += f'calculator: {str(self.calculator.__class__):>60s}\n'
        s += f'filename:   {str(self.filename):>60s}\n'
        s += f'# Atoms:    {len(self.atoms):>60d}\n'
        s += f'# IDs:      {len(self.ids):>60d}'
        return s

