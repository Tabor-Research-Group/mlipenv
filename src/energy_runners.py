import os
import abc

import numpy as np
from ase import Atoms

from src.calculators import get_calc
from src.optimization_options import ASEOptimizerConfiguration
from src.enums.output_enum import _output_file_registry

class EnergyRunner:
    