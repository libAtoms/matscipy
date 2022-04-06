"""Fixtures for Manybody potentials."""
import inspect
import pytest
import matscipy.calculators.manybody.potentials as potentials


_classes = inspect.getmembers(potentials, inspect.isclass)

_impl_potentials = {
    mother: [
        cls for _, cls in _classes
        if issubclass(cls, mother)
    ]

    for mother in (potentials.Manybody.Phi, potentials.Manybody.Theta)
}


@pytest.fixture(params=_impl_potentials[potentials.Manybody.Phi])
def pair_potential(request):
    """Fixture for pair potentials."""
    return request.param()


@pytest.fixture(params=_impl_potentials[potentials.Manybody.Theta])
def three_body_potential(request):
    """Fixture for three-body potentials."""
    return request.param()
