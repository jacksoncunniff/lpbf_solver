import numpy as np
from dataclasses import dataclass, field

@dataclass
class Material:
    """
    Class to handle and define the data for different types of powder alloys

    This should handle the static data/properties for each
    """

    # Properties as a function of temperature   
    rho_t: np.ndarray
    mu_t: np.ndarray
    dgamma_dT: np.ndarray
    specific_heat: float                # Cp of powder metal



Ti6Al4V = Material(
    rho_t = np.ndarray()


)
