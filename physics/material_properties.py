# Ti6Al4V data

def density_ti(T):
    # kg / m3

    if T < 1923:
        return 3920 - 0.68 * (T - 273)

    if T > 1923: 
        return 3920 - 0.68 * (T - 1923)


def Cp_ti(T):
    # J / kg-K

    if T < 1923:
        return 614.0582 - 0.3248 * T + T**2 * 7.625e-4

    if T > 1923: 
        return 831


def K_ti(T):
    # W/m-K
    if T < 1923:
        return 1.57 + 0.016 * (T - 273) - (T - 273)**2 * 1e-6

    if T > 1923: 
        return -6.66 + T * 1.83 * 1e-2

def mu_ti(T):
    # kg/m-s
    return 4e-3

def absorption_coeff_ti():

    return 0.8


