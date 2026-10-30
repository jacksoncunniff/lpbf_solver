from dataclasses import dataclass, field
from token import ISNONTERMINAL



@dataclass
class SolveSettings:
    """
    Settings class to define what visualization or run-time settings to execute
    """



    # Solver settings

    convergence_depth_delta: float
    simple_iterations_max: int



@dataclass
class PlotSettings:
    """
    
    """