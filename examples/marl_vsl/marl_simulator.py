import os
import numpy as np
import traci

import marl.sumo.sim_get_set as sim_old
from marl import vsl_controller as VSL
import marl.agent as Ag
import marl.sumo.sim_set_get_sumo as sim
from simex.config.settings import SumoVsl


def sumo_cmd(seed_ID, model_path):
    sumo_binary = SumoVsl.sumo_path
    # Derive SUMO_HOME from the binary path if not already set (needed by traci)
    if not os.environ.get('SUMO_HOME'):
        os.environ['SUMO_HOME'] = str(os.path.dirname(os.path.dirname(sumo_binary)))
    seeds = [28815]
    rnd = seeds[seed_ID]
    sumo_command = [sumo_binary, "-c", model_path + "highway_model.sumocfg", "--seed", str(rnd), "--start", "1", "--quit-on-end", "1"]
    return sumo_command


class Simulator:

    def sumo_simulator_vsl(flat_mod_x):
        modifier_list = [flat_mod_x] * 90
        print(f" len {len(modifier_list)}x list {modifier_list}")
        print("Running sumo simulator")
        TTSsim = 0
        step = 0
        sim_duration = 1.5  # [h]
        density_before = 0
        VSLmin = 60  # Min speed limit
        VSL_before = VSLmax = 120  # Max speed limit
        step_length = 0.25
        time_sample_TTSreward = 50  # [s]
        control_time_step = 150  # [s]
        activation_density = 25
        Kv_gain = 4.5
        Cv = 20

        VSLsection3 = VSL.VSLController("VSLsection3")
        model_path_simex = os.getenv('MODEL_PATH', f'{SumoVsl.model_path}/')
        sumoCmd = sumo_cmd(0, model_path_simex)
        traci.start(sumoCmd)

        while step < sim_duration * 3600 * (1 / step_length):
            traci.simulationStep()
            if step % (time_sample_TTSreward * (1 / step_length)) == 0 and step > 0:
                TTSsim += sim_old.GetNumberOfVeh.numVehNet() * (time_sample_TTSreward / 3600)
            if step % (control_time_step * (1 / step_length)) == 0 and step > 0:
                density_current = sim_old.GetCurrentDensity.section4()
                VSL_new = VSLsection3.P(Kv_gain, density_current, density_before, VSL_before, Cv, VSLmax, VSLmin)
                if density_current < activation_density:
                    VSL_before = VSLmax
                    sim_old.SetSpeedLimit.section3(np.round(VSLmax / 3.6, 2))
                else:
                    sim_old.SetSpeedLimit.section3(np.round(VSL_new / 3.6, 2))
                    VSL_before = VSL_new
                density_before = density_current
            if step % (60 * (1 / step_length)) == 0:
                flow = modifier_list[int((step * step_length) / 60)]
                sim_old.FlowDemand.set_mainstream_flow(flow, step, step_length)
                sim_old.FlowDemand.set_flow_on_ramp1(step, step_length)
                sim_old.FlowDemand.set_flow_on_ramp2(step, step_length)
            step += 1
        traci.close()
        return TTSsim

    def sumo_simulator_novsl(flat_mod_x):
        modifier_list = [flat_mod_x] * 90
        print(f"x list {modifier_list}")
        print("Running sumo simulator")
        TTSsim = 0
        step = 0
        sim_duration = 1.5  # [h]
        density_before = 0
        VSL_before = VSLmax = 120
        step_length = 0.25
        time_sample_TTSreward = 50
        control_time_step = 150
        activation_density = 25

        model_path_simex = os.getenv('MODEL_PATH', f'{SumoVsl.model_path}/')
        sumoCmd = sumo_cmd(0, model_path_simex)
        traci.start(sumoCmd)

        while step < sim_duration * 3600 * (1 / step_length):
            traci.simulationStep()
            if step % (time_sample_TTSreward * (1 / step_length)) == 0 and step > 0:
                TTSsim += sim_old.GetNumberOfVeh.numVehNet() * (time_sample_TTSreward / 3600)
            if step % (control_time_step * (1 / step_length)) == 0 and step > 0:
                density_current = sim_old.GetCurrentDensity.section4()
                VSL_new = 120
                if density_current < activation_density:
                    VSL_before = VSLmax
                    sim_old.SetSpeedLimit.section3(np.round(VSLmax / 3.6, 2))
                else:
                    sim_old.SetSpeedLimit.section3(np.round(VSL_new / 3.6, 2))
                    VSL_before = VSL_new
                density_before = density_current
            if step % (60 * (1 / step_length)) == 0:
                flow = modifier_list[int((step * step_length) / 60)]
                sim_old.FlowDemand.set_mainstream_flow(flow, step, step_length)
                sim_old.FlowDemand.set_flow_on_ramp1(step, step_length)
                sim_old.FlowDemand.set_flow_on_ramp2(step, step_length)
            step += 1
        traci.close()
        return TTSsim

    def marl_vsl_simulator(flat_mod_x):
        Ag.Clean_up_TTS_var()
        Ag.Clean_up_TTS_Net_Sim_var()
        print(f"AG SIM {Ag.A2.TTS_Net_Sim}")
        path_sumo_model = os.getenv('MARL_MODEL_PATH', f'{SumoVsl.marl_model_path}/')
        path_results_training = os.getenv('MARL_PATH_TRAIN', f'{SumoVsl.marl_results_path}')
        print(f"Results path: {path_results_training} and model {path_sumo_model}")

        run = 1
        epsilon = 0
        end_learning = 0
        vsl_on_off = 1
        load = 1
        sim_run_training_phase = int(os.getenv('END_RUN_MARL', '6000'))
        print(f"SIM RUN training phase {sim_run_training_phase}")
        update_QW = 0
        plotHeatMap = 0
        step_length = 0.25
        sim_duration = 1.5  # [h]
        time_sampling_TTSreward = 5
        sampling_rate_states = 50
        control_time_perid = 150  # sec
        numberSaveData = 1

        Ag.AgentsInitSimParam(vsl_on_off)
        Ag.LoadKnowlede(path_results_training, sim_run_training_phase)
        sumoCmd = sumo_cmd(0, path_sumo_model)
        traci.start(sumoCmd)
        step = 0
        controlFile = np.zeros((1, 104))
        speedLimitList = [33.33] * 160
        speedHeatMap = np.zeros((1, 160))
        speedLimitHeatMap = np.zeros((1, 160))
        numLanes = 2
        i = 0
        j = 0
        C = 0.75
        gama = 0.8
        alpha = 0.75
        beta = 1.25
        speed_free_flow = 28.33
        w = 1.5

        while step <= sim_duration * 3600 * (1 / step_length):
            traci.simulationStep()

            flow = flat_mod_x
            if step % (60 * (1 / step_length)) == 0:
                sim.My_traci.FlowDemand.set_mainstream_flow(flow, step, step_length)
                sim.My_traci.FlowDemand.set_flow_on_ramp1(step, step_length)
                sim.My_traci.FlowDemand.set_flow_on_ramp2(step, step_length)

            if step % (time_sampling_TTSreward * (1 / step_length)) == 0 and step > 0:
                Ag.A1.TTS_sectionL1 += sim.My_traci.GetNumberVeh.section_L1() * (time_sampling_TTSreward / 3600)
                Ag.A1.TTS_sectionL2 += sim.My_traci.GetNumberVeh.section_L2() * (time_sampling_TTSreward / 3600)
                Ag.A1.TTS_sectionL3 += sim.My_traci.GetNumberVeh.section_L3() * (time_sampling_TTSreward / 3600)
                Ag.A2.TTS_sectionL1 += sim.My_traci.GetNumberVeh.section_L1() * (time_sampling_TTSreward / 3600)
                Ag.A2.TTS_sectionL2 += sim.My_traci.GetNumberVeh.section_L2() * (time_sampling_TTSreward / 3600)
                Ag.A2.TTS_sectionL3 += sim.My_traci.GetNumberVeh.section_L3() * (time_sampling_TTSreward / 3600)

            if step % (sampling_rate_states * (1 / step_length)) == 0 and step > 0:
                j += 1
                Ag.A1.TTS_Net_Sim += sim.My_traci.GetNumberVeh.numVehNet() * (sampling_rate_states / 3600)
                Ag.A2.TTS_Net_Sim = Ag.A1.TTS_Net_Sim
                Ag.A1.speedSectionL0[j] = sim.My_traci.GetSpeed.section_L0()
                Ag.A1.speedSectionL1[j] = sim.My_traci.GetSpeed.section_L1()
                Ag.A1.speedSectionL2[j] = sim.My_traci.GetSpeed.section_L2()
                Ag.A1.speedSectionL3[j] = sim.My_traci.GetSpeed.section_L3()
                Ag.A2.speedSectionL0[j] = Ag.A1.speedSectionL0[j]
                Ag.A2.speedSectionL1[j] = Ag.A1.speedSectionL1[j]
                Ag.A2.speedSectionL2[j] = Ag.A1.speedSectionL2[j]
                Ag.A2.speedSectionL3[j] = Ag.A1.speedSectionL3[j]
                Ag.A1.densitySectionL0[j] = sim.My_traci.GetDensity.section_L0()
                Ag.A1.densitySectionL1[j] = sim.My_traci.GetDensity.section_L1(numLanes)
                Ag.A1.densitySectionL2[j] = sim.My_traci.GetDensity.section_L2(numLanes)
                Ag.A1.densitySectionL3[j] = sim.My_traci.GetDensity.section_L3(numLanes)
                Ag.A2.densitySectionL0[j] = Ag.A1.densitySectionL0[j]
                Ag.A2.densitySectionL1[j] = Ag.A1.densitySectionL1[j]
                Ag.A2.densitySectionL2[j] = Ag.A1.densitySectionL2[j]
                Ag.A2.densitySectionL3[j] = Ag.A1.densitySectionL3[j]

                if plotHeatMap == 1:
                    speedList = sim.My_traci.HeatMapPlot.speedPerSegments()
                    speedHeatMap = np.vstack([speedHeatMap, np.array(speedList)])

                if step > 0 and step % (control_time_perid * (1 / step_length)) == 0 and i == 0:
                    Ag.AgentState_x(j)
                    Ag.Agent1Action(epsilon, run, numberSaveData, C)
                    Ag.Agent2Action(epsilon, run, numberSaveData, C)
                    Ag.A1.VSL_beforeA1[1] = Ag.A1.VSLspeed
                    Ag.A2.VSL_beforeA1[1] = Ag.A2.VSLspeed
                    Ag.A1.prevActionWinIndex = Ag.A1.actionWinIndex
                    Ag.A2.prevActionWinIndex = Ag.A2.actionWinIndex
                    speedLimitList = sim.My_traci.SetSpeedLimit.speed_VSLzones(
                        Ag.A1.VSLspeed, Ag.A2.VSLspeed, Ag.A1.VSLposition, Ag.A2.VSLposition, plotHeatMap, speedLimitList)
                    Ag.A1.VSL_beforeA1[0] = Ag.A1.VSL_beforeA1[1]
                    Ag.A2.VSL_beforeA2[0] = Ag.A2.VSL_beforeA2[1]
                    Ag.updateCurrentSpeedStateAsPrevious(j)
                    Ag.updateCurrentDensityStateAsPrevious(j)
                    Ag.Clean_up_TTS_var()
                    i = 1
                    j = 0

                elif step > 0 and step % (control_time_perid * (1 / step_length)) == 0 and i == 1:
                    Ag.UpdateRewards(j, speed_free_flow, alpha, beta)
                    Ag.AgentState_y(j)
                    if update_QW == 1:
                        Ag.UpdateLocalpolicies(run, numberSaveData, gama, w)
                        Ag.UpdateRemotepolicies(run, numberSaveData, gama, w)
                    Ag.Clean_up_TTS_var()
                    Ag.DeepCopyOfStates()
                    Ag.Agent1Action(epsilon, run, numberSaveData, C)
                    Ag.Agent2Action(epsilon, run, numberSaveData, C)
                    Ag.A1.VSL_beforeA1[1] = Ag.A1.VSLspeed
                    Ag.A2.VSL_beforeA1[1] = Ag.A2.VSLspeed
                    Ag.A1.prevActionWinIndex = Ag.A1.actionWinIndex
                    Ag.A2.prevActionWinIndex = Ag.A2.actionWinIndex
                    speedLimitList = sim.My_traci.SetSpeedLimit.speed_VSLzones(
                        Ag.A1.VSLspeed, Ag.A2.VSLspeed, Ag.A1.VSLposition, Ag.A2.VSLposition, plotHeatMap, speedLimitList)
                    Ag.A1.VSL_beforeA1[0] = Ag.A1.VSL_beforeA1[1]
                    Ag.A2.VSL_beforeA2[0] = Ag.A2.VSL_beforeA2[1]
                    speedLimitHeatMap = np.vstack([speedLimitHeatMap, np.array(speedLimitList)])
                    if run % numberSaveData == 0 or run > end_learning:
                        controlFile = Ag.ControlFileVStack(controlFile, j)
                    Ag.updateCurrentSpeedStateAsPrevious(j)
                    Ag.updateCurrentDensityStateAsPrevious(j)
                    j = 0

            step += 1
        traci.close()
        return Ag.A2.TTS_Net_Sim

    def marl_novsl_simulator(flat_mod_x):
        path_sumo_model = os.getenv('MARL_MODEL_PATH', f'{SumoVsl.marl_model_path}/')
        Ag.Clean_up_TTS_var()
        Ag.Clean_up_TTS_Net_Sim_var()
        print(f"AG SIM {Ag.A2.TTS_Net_Sim}")
        run = 1
        epsilon = 0
        end_learning = 0
        vsl_on_off = 0
        sim_run_training_phase = 6000
        update_QW = 0
        plotHeatMap = 0
        step_length = 0.25
        sim_duration = 1.5  # [h]
        time_sampling_TTSreward = 5
        sampling_rate_states = 50
        control_time_perid = 150
        numberSaveData = 1

        Ag.AgentsInitSimParam(vsl_on_off)
        sumoCmd = sumo_cmd(0, path_sumo_model)
        traci.start(sumoCmd)
        step = 0
        controlFile = np.zeros((1, 104))
        speedLimitList = [33.33] * 160
        speedHeatMap = np.zeros((1, 160))
        speedLimitHeatMap = np.zeros((1, 160))
        numLanes = 2
        i = 0
        j = 0
        C = 0.75
        gama = 0.8
        alpha = 0.75
        beta = 1.25
        speed_free_flow = 28.33
        w = 1.5

        while step <= sim_duration * 3600 * (1 / step_length):
            traci.simulationStep()

            flow = flat_mod_x
            if step % (60 * (1 / step_length)) == 0:
                sim.My_traci.FlowDemand.set_mainstream_flow(flow, step, step_length)
                sim.My_traci.FlowDemand.set_flow_on_ramp1(step, step_length)
                sim.My_traci.FlowDemand.set_flow_on_ramp2(step, step_length)

            if step % (time_sampling_TTSreward * (1 / step_length)) == 0 and step > 0:
                Ag.A1.TTS_sectionL1 += sim.My_traci.GetNumberVeh.section_L1() * (time_sampling_TTSreward / 3600)
                Ag.A1.TTS_sectionL2 += sim.My_traci.GetNumberVeh.section_L2() * (time_sampling_TTSreward / 3600)
                Ag.A1.TTS_sectionL3 += sim.My_traci.GetNumberVeh.section_L3() * (time_sampling_TTSreward / 3600)
                Ag.A2.TTS_sectionL1 += sim.My_traci.GetNumberVeh.section_L1() * (time_sampling_TTSreward / 3600)
                Ag.A2.TTS_sectionL2 += sim.My_traci.GetNumberVeh.section_L2() * (time_sampling_TTSreward / 3600)
                Ag.A2.TTS_sectionL3 += sim.My_traci.GetNumberVeh.section_L3() * (time_sampling_TTSreward / 3600)

            if step % (sampling_rate_states * (1 / step_length)) == 0 and step > 0:
                j += 1
                Ag.A1.TTS_Net_Sim += sim.My_traci.GetNumberVeh.numVehNet() * (sampling_rate_states / 3600)
                Ag.A2.TTS_Net_Sim = Ag.A1.TTS_Net_Sim
                Ag.A1.speedSectionL0[j] = sim.My_traci.GetSpeed.section_L0()
                Ag.A1.speedSectionL1[j] = sim.My_traci.GetSpeed.section_L1()
                Ag.A1.speedSectionL2[j] = sim.My_traci.GetSpeed.section_L2()
                Ag.A1.speedSectionL3[j] = sim.My_traci.GetSpeed.section_L3()
                Ag.A2.speedSectionL0[j] = Ag.A1.speedSectionL0[j]
                Ag.A2.speedSectionL1[j] = Ag.A1.speedSectionL1[j]
                Ag.A2.speedSectionL2[j] = Ag.A1.speedSectionL2[j]
                Ag.A2.speedSectionL3[j] = Ag.A1.speedSectionL3[j]
                Ag.A1.densitySectionL0[j] = sim.My_traci.GetDensity.section_L0()
                Ag.A1.densitySectionL1[j] = sim.My_traci.GetDensity.section_L1(numLanes)
                Ag.A1.densitySectionL2[j] = sim.My_traci.GetDensity.section_L2(numLanes)
                Ag.A1.densitySectionL3[j] = sim.My_traci.GetDensity.section_L3(numLanes)
                Ag.A2.densitySectionL0[j] = Ag.A1.densitySectionL0[j]
                Ag.A2.densitySectionL1[j] = Ag.A1.densitySectionL1[j]
                Ag.A2.densitySectionL2[j] = Ag.A1.densitySectionL2[j]
                Ag.A2.densitySectionL3[j] = Ag.A1.densitySectionL3[j]

                if plotHeatMap == 1:
                    speedList = sim.My_traci.HeatMapPlot.speedPerSegments()
                    speedHeatMap = np.vstack([speedHeatMap, np.array(speedList)])

                if step > 0 and step % (control_time_perid * (1 / step_length)) == 0 and i == 0:
                    Ag.AgentState_x(j)
                    Ag.Agent1Action(epsilon, run, numberSaveData, C)
                    Ag.Agent2Action(epsilon, run, numberSaveData, C)
                    Ag.A1.VSL_beforeA1[1] = Ag.A1.VSLspeed
                    Ag.A2.VSL_beforeA1[1] = Ag.A2.VSLspeed
                    Ag.A1.prevActionWinIndex = Ag.A1.actionWinIndex
                    Ag.A2.prevActionWinIndex = Ag.A2.actionWinIndex
                    speedLimitList = sim.My_traci.SetSpeedLimit.speed_VSLzones(
                        Ag.A1.VSLspeed, Ag.A2.VSLspeed, Ag.A1.VSLposition, Ag.A2.VSLposition, plotHeatMap, speedLimitList)
                    Ag.A1.VSL_beforeA1[0] = Ag.A1.VSL_beforeA1[1]
                    Ag.A2.VSL_beforeA2[0] = Ag.A2.VSL_beforeA2[1]
                    Ag.updateCurrentSpeedStateAsPrevious(j)
                    Ag.updateCurrentDensityStateAsPrevious(j)
                    Ag.Clean_up_TTS_var()
                    i = 1
                    j = 0

                elif step > 0 and step % (control_time_perid * (1 / step_length)) == 0 and i == 1:
                    Ag.UpdateRewards(j, speed_free_flow, alpha, beta)
                    Ag.AgentState_y(j)
                    if update_QW == 1:
                        Ag.UpdateLocalpolicies(run, numberSaveData, gama, w)
                        Ag.UpdateRemotepolicies(run, numberSaveData, gama, w)
                    Ag.Clean_up_TTS_var()
                    Ag.DeepCopyOfStates()
                    Ag.Agent1Action(epsilon, run, numberSaveData, C)
                    Ag.Agent2Action(epsilon, run, numberSaveData, C)
                    Ag.A1.VSL_beforeA1[1] = Ag.A1.VSLspeed
                    Ag.A2.VSL_beforeA1[1] = Ag.A2.VSLspeed
                    Ag.A1.prevActionWinIndex = Ag.A1.actionWinIndex
                    Ag.A2.prevActionWinIndex = Ag.A2.actionWinIndex
                    speedLimitList = sim.My_traci.SetSpeedLimit.speed_VSLzones(
                        Ag.A1.VSLspeed, Ag.A2.VSLspeed, Ag.A1.VSLposition, Ag.A2.VSLposition, plotHeatMap, speedLimitList)
                    Ag.A1.VSL_beforeA1[0] = Ag.A1.VSL_beforeA1[1]
                    Ag.A2.VSL_beforeA2[0] = Ag.A2.VSL_beforeA2[1]
                    speedLimitHeatMap = np.vstack([speedLimitHeatMap, np.array(speedLimitList)])
                    if run % numberSaveData == 0 or run > end_learning:
                        controlFile = Ag.ControlFileVStack(controlFile, j)
                    Ag.updateCurrentSpeedStateAsPrevious(j)
                    Ag.updateCurrentDensityStateAsPrevious(j)
                    j = 0

            step += 1
        traci.close()
        return Ag.A2.TTS_Net_Sim
