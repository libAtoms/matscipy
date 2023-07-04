import numpy as np
import ase.io


def subsample(samples, output_files, num_subsets, num_subset_samples, keep_isolated_atoms=True):
    """
    Draw sub-samples (without replacement) from given configurations and write to files.

    Parameter:
    ----------
    samples: list(Atoms)
        List of configurations to draw sub-samples from (e.g. full training set).
    output_files: list(str / Path)
        Target locations for sub-sampled sets of configurations.
    num_subsets: int
        Number of sub-sets to be drawn.
    num_subset_samples: int
        Number of configurations per sub-sets.
    keep_isolated_atoms: bool, default True
        Make isolated atoms (if present) be part of each sub-set.

    Returns:
    --------
    subsamples: list(list(Atoms))
        Contains the lists of sub-sampled configurations.
    """
    # keep track of position in original set of configurations
    sample_atoms = []
    isolated_atoms = []  # keep isolated atoms for each sub-sample
    for idx_i, atoms_i in enumerate(samples):
        atoms_i.info['_Index_FullTrainingSet'] = idx_i
        if keep_isolated_atoms and len(atoms_i) == 1:
            isolated_atoms.append(atoms_i)
        else:
            sample_atoms.append(atoms_i)

    num_subset_samples -= len(isolated_atoms)
    assert 1 < num_subset_samples <= len(sample_atoms), 'Negative `num_subset_samples` (after reduction by number of isolated atoms)'
    assert len(output_files) == num_subsets, f'`outputs` requires `num_subsets` files to be specified.'

    subsample_indices = _get_subsample_indices(len(sample_atoms), num_subsets, num_subset_samples)
    subsamples = [isolated_atoms + [sample_atoms[idx_i] for idx_i in idxs] for idxs in subsample_indices]

    for output_file_i, subsample_i in zip(output_files, subsamples):
        ase.io.write(output_file_i, subsample_i)

    return subsamples


def _get_subsample_indices(num_samples, num_subsets, num_subset_samples):
    """
    Draw indices for sub-samples (without replacement).

    Parameter:
    ----------
    num_samples: int
        Number of configurations to draw sub-samples from (e.g. size of full training set).
    num_subsets: int
        Number of sub-sets to be drawn.
    num_subset_samples: int
        Number of configurations per sub-sets.

    Returns:
    --------
    subsample_indices: list(list(int))
        Contains the lists of indices representing sub-sampled configurations.
    """
    indice_pool = np.arange(num_samples)
    subsample_indices = []
    for _ in range(num_subsets):
        if num_subset_samples <= len(indice_pool):
            selected_indices = np.random.choice(indice_pool, num_subset_samples, False)
            indice_pool = indice_pool[~np.isin(indice_pool, selected_indices)]
            subsample_indices.append(selected_indices)
        else:
            selected_indices_part_1 = indice_pool
            # re-fill pool with indices, taking account of already selected ones,
            # in order to avoid duplicate selections
            indice_pool = np.arange(num_samples)
            indice_pool = indice_pool[~np.isin(indice_pool, selected_indices_part_1)]
            selected_indices_part_2 = np.random.choice(indice_pool, num_subset_samples - len(selected_indices_part_1), False)
            indice_pool = indice_pool[~np.isin(indice_pool, selected_indices_part_2)]
            selected_indices = np.concatenate((selected_indices_part_1, selected_indices_part_2))
            subsample_indices.append(selected_indices)

    return subsample_indices
