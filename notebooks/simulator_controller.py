class SimulatorController:

    def simulate(mod_x, selected_simulator):
        print("Simulator...")
        print(mod_x)
        if mod_x is False:
            return False  # Possible iterations have ended
        print(f" IN SIMULATE: {mod_x}")
        
        flat_mod_x = [item for sublist in mod_x for item in sublist]
        print(f"Flatten mod_x {flat_mod_x}")
        simulated_y = [selected_simulator(x) for x in flat_mod_x]
        # print('  * Sim_y:   ', simulated_y)
        return flat_mod_x, simulated_y

    # Amy test
    def simulator(mod_x, selected_simulator):
        simulated_y = []
        print("Simulator...")
        if mod_x is False:
            return False  # Possible iterations have ended
        ids = [mod_x[x][0] for x in range(0, len(mod_x))]
        arrays = [mod_x[x][1] for x in range(0, len(mod_x))]
        # print(f" IN SIMULATE arrays: {arrays}")
        simulated_y = [selected_simulator(array) for array in arrays]
        return ids, simulated_y

    def sumo_simulate(mod_x, selected_simulator):
        simulated_y = []
        print(f"Simulator... {mod_x}")
        if mod_x is False:
            return False  # Possible iterations have ended
        flat_mod_x = [item for sublist in mod_x for item in sublist]
        arrays = [[i] * 90 for i in flat_mod_x]
        print(f" IN SIMULATE arrays: {arrays}")
        print(f" IN SIMULATE arrays: {mod_x}")
        simulated_y = [selected_simulator(array) for array in arrays]
        return flat_mod_x,simulated_y
