"""
SimEx - Systematic Exploration Tool

A Python package for systematic exploration of simulation models.
This tool was used in the Evaluation of the traffic controller use case.

Reference:
K. Kušić et al., "Evaluation of Traffic Controller Performance via Systematic
Exploration," 2024 International Symposium ELMAR, Zadar, Croatia, 2024,
pp. 165-168, doi: 10.1109/ELMAR62909.2024.10694499.
"""

__version__ = "1.0.0"
__author__ = "SimEx Team"
__email__ = "amy.liffey@hevs.ch"

# Core imports
from .core.simex import Simex
from .core.settings import SimexSettings

# Component imports
from .components.modifier import Modifier
from .components.simulator import Simulator
from .components.validator import Validator

# Controller imports
from .controllers.modifier_controller import ModifierController
from .controllers.simulator_controller import SimulatorController
from .controllers.validator_controller import ValidatorController

# Utility imports
from .utils.logger_utils import Logger

# Configuration imports
from .config.components_configuration import components

__all__ = [
    # Core classes
    "Simex",
    "SimexSettings",
    # Components
    "Modifier",
    "Simulator",
    "Validator",
    # Controllers
    "ModifierController",
    "SimulatorController",
    "ValidatorController",
    # Utils
    "Logger",
    # Config
    "components",
]
