"""Physical constants used throughout the package (SI units)."""

import scipy.constants as _sc

#: Planck constant [J s]
H = _sc.h
#: Speed of light in vacuum [m/s]
C = _sc.c
#: Boltzmann constant [J/K]
KB = _sc.k
#: Elementary charge [C]
Q_E = _sc.e
#: Stefan-Boltzmann constant [W m^-2 K^-4]
SIGMA = _sc.sigma
#: Wien displacement constant [m K]
WIEN_B = _sc.Wien

#: First radiation constant for spectral radiance, 2 h c^2 [W m^2 sr^-1]
C1L = 2.0 * H * C**2
#: Second radiation constant, h c / k [m K]
C2 = H * C / KB
