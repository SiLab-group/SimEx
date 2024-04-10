class SimulatorController:

    def simulate(mod_x, selected_simulator):
        print("Simulator...")
        if mod_x is False:
            return False  # Possible iterations have ended

        flat_mod_x = [item for sublist in mod_x for item in sublist]
        # print(flat_mod_x)
        simulated_y = [selected_simulator(x) for x in flat_mod_x]
        # print('  * Sim_y:   ', simulated_y)
        return flat_mod_x, simulated_y
