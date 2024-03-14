import numpy as np
import math

class Simulator:
    def sim_func_A(x):
        # print('\nthis should be a single point: ',x)
        noise = np.random.normal(-10, 10, 1)
        return float(x**3 - x**2 + noise)

    def sim_func_B(x):
        noise = np.random.normal(-10, 10, 1)

        return float(math.sin(x * 2 / 3) + noise)
    
    def sim_func_C(x):
        noise = np.random.normal(-3, 3, 1)

        return float(math.sin(x**2) + x * 2 / 3 + noise)

    def simulator_controller(mod_x, selected_function=sim_func_C):
        print("\nSimulator...")
        # shape_mod_x = np.shape(mod_x)
        if mod_x is False: 
            return False  # Possible iterations have ended
        
        flat_mod_x = [item for sublist in mod_x for item in sublist]
        # flat_mod_x = np.reshape(mod_x,shape_mod_x[0]*shape_mod_x[1])
        # print(flat_mod_x)
        simulated_y = [selected_function(x) for x in flat_mod_x]
        # print('  * Sim_y shape:   ', np.shape(simulated_y))
        return flat_mod_x, simulated_y
