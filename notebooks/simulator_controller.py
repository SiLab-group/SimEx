from concurrent.futures import ProcessPoolExecutor

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

    def simulate_parallel(mod_x, selected_simulator, workers):
        print("Simulator...")
        print(mod_x)
        if mod_x is False:
            return False  # Possible iterations have ended
        print(f" IN SIMULATE: {mod_x}")

        flat_mod_x = [item for sublist in mod_x for item in sublist]
        print(f"Flatten mod_x {flat_mod_x}")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            simulated_y = list(executor.map(selected_simulator, flat_mod_x))

            print("Simulation output:", simulated_y)

        # print('  * Sim_y:   ', simulated_y)
        return flat_mod_x, simulated_y
