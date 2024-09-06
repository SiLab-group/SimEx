import numpy as np
import math
# Sumo vsl imports
import traci
import sim_get_set as sim
import vsl_controller as VSL
from global_settings import SumoVsl

class Simulator:

    def sim_func_A(x):
        # print('\nthis should be a single point: ',x)
        noise = np.random.normal(-10, 10, 1)
        return float(x ** 3 - x ** 2 + noise)

    def sim_func_B(x):
        noise = np.random.normal(-10, 10, 1)
        print(f"FUNCTION: {float(math.sin(x * 2 / 3) + noise)}")
        return float(math.sin(x * 2 / 3) + noise)

    def sim_func_C(x):
        noise = np.random.normal(-1, 1, 1)

        return float(math.sin(x ** 2) + x * 2 / 3 + noise)

    def sumo_simulator_vsl(flat_mod_x):
        modifier_list = [flat_mod_x] * 90
        print(f" len {len(modifier_list)}x list {modifier_list}")
        # Given modifier_list this returns list of TTS_sim NOT Aggregated for id
        tts_sim = []
        print("Running sumo simulator")
        # print(modifier_list)
        TTSsim = 0
        step = 0
        sim_duration = 1.5 # [h]
        density_before = 0
        VSLmin = 60  # Min speed limit
        VSL_before = VSLmax = 120  # Max speed limit
        run_count = 0
        step_length = 0.25
        time_sample_TTSreward = 50  # [s]  because control interavl of VSL is 150 [s]
        control_time_step = 150  # [s]
        activation_density = 25  # Controls when VSL controller will be active and compute new speed limit
        Kv_gain = 4.5  # VSL controller gain  (in this scenario we use proportional controller)
        Cv = 20  # max speed difference between speed limits

        VSLsection3 = VSL.VSLController("VSLsection3")

        # SUMO variables
        # Adjust path for the sumo
        root = SumoVsl.model_path
        sumoBinary = SumoVsl.sumo_path
        def SUMO_Cmd(seed_ID):
            # def SUMOCmd():
            # seeds=[21121,29350,23496,22287,22903,25614,23864,22426,23345,28815]
            seeds = [28815]
            # rnd=random.randint(20000, 30000)
            rnd = seeds[seed_ID]
            # sumoCmd = [sumoBinary, "-c", root + "highway_model.sumocfg", "--seed", str(rnd), "--start", "1", "--quit-on-end", "1","--remote-port", "9999"]
            sumoCmd = [sumoBinary, "-c", root + "highway_model.sumocfg", "--seed", str(rnd), "--start", "1","--quit-on-end","1"]
            return sumoCmd
        
        #==================================================================
        # run simulation
        sumoCmd = SUMO_Cmd(0)  # At the moment we use one seed
        traci.start(sumoCmd)

        while (step < sim_duration * 3600 * (1 / step_length)):
            traci.simulationStep()
            # ==================================================================
    
            # ==============================================================
            # OBJECTIVE function, cumulative travel time of all vehicles in the network per simulation
            if step % (time_sample_TTSreward * (1 / step_length)) == 0 and step > 0:
                TTSsim += sim.GetNumberOfVeh.numVehNet() * (time_sample_TTSreward / 3600)
                # Returning the value for each y and not aggregated.. maybe it can be aggregated later
                #tts_sim.append(sim.GetNumberOfVeh.numVehNet() * (time_sample_TTSreward / 3600))
            # ==============================================================
    
            # ==============================================================
            # VSL calculation
            if step % (control_time_step * (1 / step_length)) == 0 and step > 0:
                # Get density in the bottleneck
                density_current = sim.GetCurrentDensity.section4()
                # Calculate new speed limit by VSL vontroller
                VSL_new = VSLsection3.P(Kv_gain, density_current, density_before, VSL_before, Cv, VSLmax, VSLmin)
    
                if density_current < activation_density:
                    VSL_before = VSLmax
                    sim.SetSpeedLimit.section3(np.round(VSLmax / 3.6, 2))
                else:
                    sim.SetSpeedLimit.section3(np.round(VSL_new / 3.6, 2))
                    VSL_before = VSL_new
                density_before = density_current
                # ==========================================================
    
            # SetFlow
            # Generate/modify traffic flows
            # ==============================================================
            if step % (60 * (1 / step_length)) == 0:
                # This can be modified by modifiers
                # Mainstream Flow rate [2800, 3800] veh/h
                # print(f"Last element {int((step * step_length) / 60)}")
                # print(
                #     f'Generate traffic flows RUN: {run_count} {step} and step length {step_length} mod list {modifier_list[run_count][1][int((step * step_length) / 60)]}')
                # flow = modifier_list[run_count][1][int((step * step_length) / 60)]
                flow = modifier_list[int((step * step_length )/ 60)]
                # print(f'Mainstream flowrate {flow} for {step} with step length {step_length}')
                sim.FlowDemand.set_mainstream_flow(flow, step, step_length)
    
                # No modification at the moment
                sim.FlowDemand.set_flow_on_ramp1(step, step_length)
                sim.FlowDemand.set_flow_on_ramp2(step, step_length)
            # ==============================================================
    
            step += 1
        traci.close()
        return TTSsim

    def sumo_simulator_novsl(flat_mod_x):
        modifier_list = [flat_mod_x] * 90
        print(f"x list {modifier_list}")
        # Disabled vsl functionss
        # Given modifier_list this returns list of TTS_sim NOT Aggregated for id
        tts_sim = []
        print("Running sumo simulator")
        # print(modifier_list)
        TTSsim = 0
        step = 0
        sim_duration = 1.5  # [h]
        density_before = 0
        VSLmin = 60  # Min speed limit
        VSL_before = VSLmax = 120  # Max speed limit
        run_count = 0
        step_length = 0.25
        time_sample_TTSreward = 50  # [s]  because control interavl of VSL is 150 [s]
        control_time_step = 150  # [s]
        activation_density = 25  # Controls when VSL controller will be active and compute new speed limit
        Kv_gain = 4.5  # VSL controller gain  (in this scenario we use proportional controller)
        Cv = 20  # max speed difference between speed limits

        VSLsection3 = VSL.VSLController("VSLsection3")

        # SUMO variables
        # Adjust path for the sumo
        root = SumoVsl.model_path
        sumoBinary = SumoVsl.sumo_path

        def SUMO_Cmd(seed_ID):
            # def SUMOCmd():
            # seeds=[21121,29350,23496,22287,22903,25614,23864,22426,23345,28815]
            seeds = [28815]
            # rnd=random.randint(20000, 30000)
            rnd = seeds[seed_ID]
            sumoCmd = [sumoBinary, "-c", root + "highway_model.sumocfg", "--seed", str(rnd), "--start", "1",
                       "--quit-on-end", "1"]
            return sumoCmd

        # ==================================================================
        # run simulation
        sumoCmd = SUMO_Cmd(0)  # At the moment we use one seed
        traci.start(sumoCmd)

        while (step < sim_duration * 3600 * (1 / step_length)):
            traci.simulationStep()
            # ==================================================================

            # ==============================================================
            # OBJECTIVE function, cumulative travel time of all vehicles in the network per simulation
            if step % (time_sample_TTSreward * (1 / step_length)) == 0 and step > 0:
                TTSsim += sim.GetNumberOfVeh.numVehNet() * (time_sample_TTSreward / 3600)
                # Returning the value for each y and not aggregated.. maybe it can be aggregated later
                # tts_sim.append(sim.GetNumberOfVeh.numVehNet() * (time_sample_TTSreward / 3600))
            # ==============================================================

            # ==============================================================
            # VSL calculation
            if step % (control_time_step * (1 / step_length)) == 0 and step > 0:
                # Get density in the bottleneck
                density_current = sim.GetCurrentDensity.section4()
                # Calculate new speed limit by VSL vontroller
                # VSL_new = VSLsection3.P(Kv_gain, density_current, density_before, VSL_before, Cv, VSLmax, VSLmin)
                VSL_new = 120

                if density_current < activation_density:
                    VSL_before = VSLmax
                    sim.SetSpeedLimit.section3(np.round(VSLmax / 3.6, 2))
                else:
                    sim.SetSpeedLimit.section3(np.round(VSL_new / 3.6, 2))
                    VSL_before = VSL_new
                density_before = density_current
                # ==========================================================

            # SetFlow
            # Generate/modify traffic flows
            # ==============================================================
            if step % (60 * (1 / step_length)) == 0:
                # This can be modified by modifiers
                # Mainstream Flow rate [2800, 3800] veh/h
                # print(f"Last element {int((step * step_length) / 60)}")
                # print(
                #     f'Generate traffic flows RUN: {run_count} {step} and step length {step_length} mod list {modifier_list[run_count][1][int((step * step_length) / 60)]}')
                # flow = modifier_list[run_count][1][int((step * step_length) / 60)]
                flow = modifier_list[int((step * step_length) / 60)]
                # print(f'Mainstream flowrate {flow} for {step} with step length {step_length}')
                sim.FlowDemand.set_mainstream_flow(flow, step, step_length)

                # No modification at the moment
                sim.FlowDemand.set_flow_on_ramp1(step, step_length)
                sim.FlowDemand.set_flow_on_ramp2(step, step_length)
            # ==============================================================

            step += 1
        traci.close()
        # print(f"TTS SIM array {tts_sim}")
        # return tts_sim
        return TTSsim