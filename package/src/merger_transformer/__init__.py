"""Waveform generation and automatic differentiation for the released model."""
from .api import WaveformModel
from ._network import MODES

__version__ = "0.4.0"
__all__ = ["WaveformModel", "MODES", "__version__"]
