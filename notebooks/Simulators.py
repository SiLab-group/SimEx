import numpy as np
import math

class Simulators:
    
    def sim_func_A(x):
        # print('\nthis should be a single point: ',x)
        noise = np.random.normal(-10, 10, 1)
        return float(x**3 - x**2 + noise)

    def sim_func_B(x):
        noise = np.random.normal(-10, 10, 1)

        return float(math.sin(x * 2 / 3) + noise)
    
    def sim_func_C(x):
        noise = np.random.normal(-1, 1, 1)

        return float(math.sin(x**2) + x * 2 / 3 + noise)