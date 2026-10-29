import numpy as np
from dataclasses import dataclass, field

@dataclass
class FlowField:
    """
    TODO docstring for fields classes
    """

    uq: np.ndarray
    w: np.ndarray
    p: np.ndarray
    T: np.ndarray
    mu: np.ndarray
    rho: np.ndarray

    uq_old: np.ndarray
    w_old: np.ndarray

    uq_star: np.ndarray
    w_star: np.ndarray
    p_prime: np.ndarray

    dpdx: np.ndarray
    dpdz: np.ndarray

    dudx: np.ndarray
    dwdx: np.ndarray

@dataclass
class FaceField:
    """
    """

    flux_q: np.ndarray
    flux_w: np.ndarray

