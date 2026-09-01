"""ircam: physics analysis toolkit for infrared camera requirements."""

from . import atmosphere, bands, constants, detector, optics, planck, radiometry, range_performance
from .atmosphere import CLEAR_SEA_LEVEL, SimpleAtmosphere
from .bands import LWIR, MWIR, STANDARD_BANDS, SWIR, Band
from .detector import Fpa, blip_dstar, dstar_from_jones, netd_from_dstar
from .optics import Optics
from .radiometry import PhotonDetectorChain

__all__ = [
    "atmosphere", "bands", "constants", "detector", "optics", "planck",
    "radiometry", "range_performance",
    "CLEAR_SEA_LEVEL", "SimpleAtmosphere",
    "LWIR", "MWIR", "SWIR", "STANDARD_BANDS", "Band",
    "Fpa", "blip_dstar", "dstar_from_jones", "netd_from_dstar",
    "Optics", "PhotonDetectorChain",
]

__version__ = "0.1.0"
