import timeit
from numba import njit
import numpy as np



def _TDMA(a, b, c, d, x):
    for i in range(1, len(b)):
        m = a[i] / b[i-1]
        b[i] = b[i] - m * c[i-1]
        d[i] = d[i] - m * d[i-1]
    x[-1] = d[-1] / b[-1]
    for i in range(len(x) -2, -1, -1):
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]
    
    return x

_TDMA_jit = njit(_TDMA)

# Test arrays
N = 250
lower = np.full(N, -1.0)
main = np.full(N, 4.0)
upper = np.full(N, -1.0)
rhs = np.full(N, 2.0)

lower[0] = 0.0
upper[-1] = 0.0

def run(solver):
    return solver(
        lower.copy(),
        main.copy(),
        upper.copy(),
        rhs.copy(),
        np.zeros(N)
    )


python_time = timeit.timeit(lambda: run(_TDMA), number=62500)
jit_time = timeit.timeit(lambda: run(_TDMA_jit), number=62500)

print(f"Speedup: {python_time/jit_time:.1f}x")