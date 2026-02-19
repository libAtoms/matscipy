
from abc import ABC, abstractmethod
import numpy as np


class WeightFunction(ABC):
    """Abstract base class for weight functions used in density estimation."""

    @abstractmethod
    def __call__(self, r) -> float:
        """Evaluate weight function W(r) for cutoff rc."""
        pass

    @abstractmethod
    def derivative(self, r) -> float:
        """Evaluate dW/dr for cutoff rc."""
        pass

    @abstractmethod
    def second_derivative(self, r) -> float:
        """Evaluate d²W/dr² for cutoff rc."""
        pass

    @abstractmethod
    def at_zero(self) -> float:
        """Evaluate W(0) for cutoff rc (self-contribution)."""
        pass


class LucyWeightFunction(WeightFunction):
    """Lucy weight function for SPH-like density estimation.

    The Lucy function is normalized to integrate to 1 over 3D space:

        W(r) = (105 / 16π rc³) * (1 + 3r/rc) * (1 - r/rc)³  for r < rc
        W(r) = 0                                             for r >= rc

    This weight function is smooth (C¹ continuous at r=rc) and commonly
    used in smoothed particle hydrodynamics.
    """

    def __init__(self, cutoff):
        self.cutoff = cutoff

    def __call__(self, r):
        """Evaluate Lucy weight function."""
        r = np.asarray(r)
        result = np.zeros_like(r, dtype=float)
        mask = r < self.cutoff
        x = r[mask] / self.cutoff
        # Normalization: 105 / (16 * pi * rc^3)
        norm = 105.0 / (16.0 * np.pi * self.cutoff**3)
        result[mask] = norm * (1.0 + 3.0 * x) * (1.0 - x) ** 3
        return result

    def derivative(self, r):
        """Evaluate dW/dr."""
        r = np.asarray(r)
        result = np.zeros_like(r, dtype=float)
        mask = r < self.cutoff
        x = r[mask] / self.cutoff
        norm = 105.0 / (16.0 * np.pi * self.cutoff**3)
        # d/dr[(1 + 3x)(1 - x)^3] = (1/rc) * [3(1-x)^3 - 3(1+3x)(1-x)^2]
        #                        = (1/rc) * 3(1-x)^2 * [(1-x) - (1+3x)]
        #                        = (1/rc) * 3(1-x)^2 * (-4x)
        #                        = -12x(1-x)^2 / rc
        result[mask] = norm * (-12.0 * x * (1.0 - x) ** 2) / self.cutoff
        return result

    def second_derivative(self, r):
        """Evaluate d²W/dr²."""
        r = np.asarray(r)
        result = np.zeros_like(r, dtype=float)
        mask = r < self.cutoff
        x = r[mask] / self.cutoff
        norm = 105.0 / (16.0 * np.pi * self.cutoff**3)
        # d²/dr²[(1 + 3x)(1 - x)^3]
        # First derivative: dW/dr = -12*norm*x*(1-x)^2 / rc
        # Second derivative: d/dr[-12*norm*x*(1-x)^2 / rc] / rc
        # = -12*norm/rc² * d/dx[x*(1-x)^2]
        # = -12*norm/rc² * [(1-x)^2 + x*2*(1-x)*(-1)]
        # = -12*norm/rc² * [(1-x)^2 - 2x*(1-x)]
        # = -12*norm/rc² * (1-x)*[(1-x) - 2x]
        # = -12*norm/rc² * (1-x)*(1-3x)
        result[mask] = norm * (-12.0 * (1.0 - x) * (1.0 - 3.0 * x)) / self.cutoff**2
        return result

    def at_zero(self):
        """Evaluate W(0)."""
        return 105.0 / (16.0 * np.pi * self.cutoff**3)


class LucyWeightFunction2D(WeightFunction):
    """Lucy weight function for SPH-like density estimation.

    The Lucy function is normalized to integrate to 1 over 2D space:

        W(r) = (5 / π rc²) * (1 + 3r/rc) * (1 - r/rc)³  for r < rc
        W(r) = 0                                         for r >= rc

    This weight function is smooth (C¹ continuous at r=rc) and commonly
    used in smoothed particle hydrodynamics.
    """

    dim = 2

    def __init__(self, cutoff):
        self.cutoff = cutoff

    def __call__(self, r):
        """Evaluate Lucy weight function."""
        r = np.asarray(r)
        result = np.zeros_like(r, dtype=float)
        mask = r < self.cutoff
        x = r[mask] / self.cutoff
        # Normalization: 5 / (pi * rc^2)
        norm = 5.0 / (np.pi * self.cutoff**2)
        result[mask] = norm * (1.0 + 3.0 * x) * (1.0 - x) ** 3
        return result

    def derivative(self, r):
        """Evaluate dW/dr."""
        r = np.asarray(r)
        result = np.zeros_like(r, dtype=float)
        mask = r < self.cutoff
        x = r[mask] / self.cutoff
        norm = 5.0 / (np.pi * self.cutoff**2)
        # d/dr[(1 + 3x)(1 - x)^3] = (1/rc) * [3(1-x)^3 - 3(1+3x)(1-x)^2]
        #                        = (1/rc) * 3(1-x)^2 * [(1-x) - (1+3x)]
        #                        = (1/rc) * 3(1-x)^2 * (-4x)
        #                        = -12x(1-x)^2 / rc
        result[mask] = norm * (-12.0 * x * (1.0 - x) ** 2) / self.cutoff
        return result

    def second_derivative(self, r):
        """Evaluate d²W/dr²."""
        r = np.asarray(r)
        result = np.zeros_like(r, dtype=float)
        mask = r < self.cutoff
        x = r[mask] / self.cutoff
        norm = 5.0 / (np.pi * self.cutoff**2)
        # d²/dr²[(1 + 3x)(1 - x)^3]
        # First derivative: dW/dr = -12*norm*x*(1-x)^2 / rc
        # Second derivative: d/dr[-12*norm*x*(1-x)^2 / rc] / rc
        # = -12*norm/rc² * d/dx[x*(1-x)^2]
        # = -12*norm/rc² * [(1-x)^2 + x*2*(1-x)*(-1)]
        # = -12*norm/rc² * [(1-x)^2 - 2x*(1-x)]
        # = -12*norm/rc² * (1-x)*[(1-x) - 2x]
        # = -12*norm/rc² * (1-x)*(1-3x)
        result[mask] = norm * (-12.0 * (1.0 - x) * (1.0 - 3.0 * x)) / self.cutoff**2
        return result

    def at_zero(self):
        """Evaluate W(0)."""
        return 5.0 / (np.pi * self.cutoff**2)


