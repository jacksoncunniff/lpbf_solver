import numpy as np
import matplotlib.pyplot as plt

from core.mesh import Mesh
from core.fields import FlowField
from core.settings import SolveSettings, PlotSettings

def plot_temperature(fields: FlowField):

    plt.matshow(fields.T)
    plt.show()