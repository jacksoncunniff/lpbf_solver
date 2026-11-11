# Ti6Al4V data
import numpy as np

def density_ti(T):
    # kg / m3

    return np.where(
        T < 1923,
        3920 - 0.68 * (T - 273),
        3920 - 0.68 * (T - 1923)

    )


def Cp_ti(T):
    # J / kg-K

    return np.where(
        T < 1923,
        614.0582 - 0.3248 * T + T**2 * 7.625e-4,
        831
    )


def K_ti(T):
    # W/m-K
    return np.where(
        T < 1923,
        1.57 + 0.016 * (T - 273) - (T - 273)**2 * 1e-6,
        -6.66 + T * 1.83 * 1e-2
    )

def mu_ti(T):
    # kg/m-s
    return 4e-3

def absorption_coeff_ti():

    return 0.8


