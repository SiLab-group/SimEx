import numpy as np
from global_settings import sfs

class SimulatorController:

    def simulatorController(mod_x, selected_function):
        print("\nSimulator...")
        # shape_mod_x = np.shape(mod_x)
        if mod_x is False: 
            return False  # Possible iterations have ended
        
        flat_mod_x = [item for sublist in mod_x for item in sublist]
        # flat_mod_x = np.reshape(mod_x,shape_mod_x[0]*shape_mod_x[1])
        # print(flat_mod_x)
        simulated_y = [selected_function(x) for x in flat_mod_x]
        print('  * Sim_y:   ', simulated_y)
        return flat_mod_x, simulated_y
