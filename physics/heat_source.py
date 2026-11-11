import numpy as np
from dataclasses import dataclass, field
from core.fields import FlowField
from core.mesh import Mesh


"""
TODO 

refactor from base routines into class based factories

"""

@dataclass
class LaserSource:
    """
    Class to handle the factory for laser properties, temperature updates and anything related to the heating of surface
    """

    # Laser Properties
    P: float                # laser power [W]
    A: float
    Cp: np.ndarray              # Powder
    a: float                # half of laser diameter in Y
    b: float                # half of laser diameter in X
    c: float                # laser penetration depth
    α: float                # thermal diffusivity of powder bed
    V: float                # speed of laser [m/s]
    t_prime: float          # some kind of ref time?
    delta_t: float          # time increment since previous update


    def _laser_module(self, x, y, z, current_time, dt, rho: np.ndarray):
        """
        Source temperature method for calculating the increment in temperature according to eq (2)

        This will return a Nx by Nz array of temperature increment depending on the current time 
        of the simulation

        The temperature source equation does not explicitly take a parameter for laser center, instead 
        it calculated the position as distance traveled from x with the V*t_prime term in X2

        current_time is assumed to be the time elapsed in the laser heating, which will be the total time 
        elapsed in the whole simulation in this case
        
        """
        P = self.P
        A = self.A
        α = self.α
        V = self.V
        C = self.Cp
        a = self.a
        b = self.b
        c = self.c

        X1 = (3*np.sqrt(3.0)*P*A)/(rho * C*np.pi*np.sqrt(np.pi))
        X2 = np.exp(-3*((x-V*current_time)**2/(12*α*dt+a**2) + y**2/(12*α*dt+b**2) + z**2/(12*α*dt+c**2))) 
        X3 = np.sqrt(12*α*dt + a**2) * np.sqrt(12*α*dt + b**2) * np.sqrt(12*α*dt + c**2)

        return X1*X2/X3*dt

    def temperature_update(self, fields: FlowField, mesh: Mesh, current_time: float, delta_time: float):
        """
        This method is intended to update the provided temperature fields
        from either the X-Z or Y-Z dict using the temperature_source method.

        """

        rho_field = fields.rho
        x = mesh.x
        z = mesh.z


        dT_fields = self._laser_module(
                                    x=x,
                                    y=0, 
                                    z=z, 
                                    current_time=current_time, 
                                    dt=delta_time, 
                                    rho=rho_field
                                    )
        return dT_fields