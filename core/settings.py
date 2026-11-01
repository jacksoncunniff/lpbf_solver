from dataclasses import dataclass, field
from typing import Literal

RunMode = Literal["test", "solve"]


@dataclass
class SolveSettings:
    """
    Settings class to define what visualization or run-time settings to execute
    """



    # Solver settings

    convergence_depth_delta: float
    simple_iterations_max: int
    mode: RunMode = "solve"
    relaxation_factor: float = 0.5 # random guess for now



@dataclass
class PlotSettings:
    """
    
    """