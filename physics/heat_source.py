import numpy as np
from dataclasses import dataclass, field








#. Laser variables
P = 1  # power
A = 1  # 
C = 1  # powder Cp
a = 1  # half of laser diam in Y 
b = 1  # half of laser diam in X
c = 1  # laser penetration depth
α = 1  # thermal diffisuvity of powder bed
C = 1  # heat capacity powder?
V = 1  # speed of the heat source [m/s]
t_prime = 1  # some kind of time ?
delta_time = 0.0025 # time increment


def temperature_update(x, y, z, current_time, dt):
    """
    Source temperature method for calculating the increment in temperature according to eq (2)

    This will return a Nx by Nz array of temperature increment depending on the current time 
    of the simulation

    The temperature source equation does not explicitly take a parameter for laser center, instead 
    it calculated the position as distance traveled from x with the V*t_prime term in X2

    current_time is assumed to be the time elapsed in the laser heating, which will be the total time 
    elapsed in the whole simulation in this case
    
    """

    X1 = (3*np.sqrt(3)*P*A)/(rho*C*np.pi*np.sqrt(np.pi))
    X2 = np.exp(-3*((x-V*current_time)**2/(12*α*dt+a**2) + y**2/(12*α*dt+b**2) + z**2/(12*α*dt+c**2))) 
    X3 = np.sqrt(12*α*dt + a**2) * np.sqrt(12*α*dt + b**2) * np.sqrt(12*α*dt + c**2)

    return X1*X2/X3*dt

def heating_module(T_fields, current_time):
    """
    This method is intended to update the provided temperature fields
    from either the X-Z or Y-Z dict using the temperature_source method.

    """

    dT_fields = temperature_update(x=x_coords, 
                                   y=0, 
                                   z=z_coords, 
                                   current_time=current_time, 
                                   dt=delta_time)
    T_fields += dT_fields