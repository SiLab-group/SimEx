"""
This module is used to train the VSL agents.
"""

import marl.agent as Ag
import marl.sumo.sim_set_get_sumo as sim
import os, sys
import traci
import numpy as np
from numpy import savetxt
import math
import datetime as dt

from simex.config.settings import SumoVsl


class Controller:
    def __init__(self, results_path):
        self.sumo_binary = SumoVsl.sumo_path
        # self.root = SumoVsl.marl_root_path
        self.path_sumo_model = SumoVsl.marl_model_path
        self.path_results = results_path

    def marl_vsl_training(self, training_data, run_start, iterations, vsl_on=1):
        path_results = self.path_results
        sumo_cmd = [self.sumo_binary, "-c", os.path.join(self.path_sumo_model, "highway_model.sumocfg"), "--seed", str(28815), "--start", "1", "--quit-on-end", "1"]
        print(f"====Start training ========= run {run_start}")
        # ===== run simulation and train VSL agents
        run = run_start + 1
        end_learning = run + iterations + 1  # 14000
        vsl_on_off = vsl_on  # (1 active VSL)
        load = run - 1  # load=run-1 # (# load old Qs and Ws, 0 start learning from scratch)
        update_QW = 1  # 0 - no update, 1 - update
        plotHeatMap = 0
        print(f"====Start training ========= run {run} and {end_learning}")
        # sim parameters
        step_length = 0.25
        """
        Step lenght of simulation integration
        """
        sim_duration = 1.5  # [h]

        time_sampling_TTSreward = 5
        """
        Time step for sampling TTS for reward function.
        Finer granularity more precise TTS. However, slower simulation!
        """
        sampling_rate_states = 50
        """
        Time step for sampling traffic states
        """
        control_time_perid = sampling_rate_states * 3  # 150 [sec]
        """
        Time step at which decision are taken by VSL agents.
        In current code version must be a  multiple of sampling_rate_states!
        """
        numberSaveData = 20  # number of simulations after which data must be saved (you can change this e.g., 40 which might speed up the process)
        # (safety property, if crashes, learning parameters can be re-loaded from the last saved sim)

        state_training_samples = training_data
        # Load old knowledge in case training process (simulations) break!
        if load != 0 and vsl_on_off == 1:
            Ag.LoadKnowlede(self.path_results, load)
            load = 0

        # run simulations
        x = 1
        while (run < end_learning ):
            print(f"Run: {run} and {end_learning}")

            if run <= end_learning:
                # We give agents ability to explore again but starting from probability 0.5
                # and asymptotically approaching 0 for additional 6000 simulations.
                # It might be a naive approach, however, this requires an upgrade with some RL continual learning techniques.
                epsilon = math.exp((((-1) * math.log(20)) / 6000) * run) * (0.5)
            else:
                epsilon = 0
                plotHeatMap = 1
                vsl_on_off = 0
                Ag.AgentsInitSimParam(vsl_on_off)


            # # Load old knowledge in case training process (simulations) break!
            # if load != 0 and vsl_on_off == 1:
            #     Ag.LoadKnowlede(path_results)
            #     load = 0

            # ===================================================== Init Agents

            traci.start(sumo_cmd)
            step = 0

            controlFile = np.zeros((1, 104))

            speedLimitList = []
            for ll in range(0, 160):
                speedLimitList.append(33.33)
            speedHeatMap = np.zeros((1, 160))
            speedLimitHeatMap = np.zeros((1, 160))

            numLanes = 2

            i = 0
            vf = 41.6  # 150 km/h
            j = 0

            C = 0.75  # cooperation coefficient (see our papers)
            gama = 0.8
            alpha = 0.75
            beta = 1.25
            speed_free_flow = 28.33
            w = 1.5  # parameter which controls how fast W converges (see papers)
            # Epsilon calculation
            epsilon = math.exp((((-1) * math.log(20)) / 6000) * x) * (0.5)

            # run (update) current simulation (duration 5400 s = 1.5 h)
            # while(step <= 5400):

            while (step < sim_duration * 3600 * (1 / step_length)):
                traci.simulationStep()

                # SetFlow
                # Generate/modify traffic flows
                if step % (60 * (1 / step_length)) == 0:
                    # This can be modified by modifiers
                    # Mainstream Flow rate [2800, 3800] veh/h
                    # print(f"Last element {int((step * step_length) / 60)}")
                    # print(
                    #     f'Generate traffic flows RUN: {run_count} {step} and step length {step_length} mod list {modifier_list[run_count][1][int((step * step_length) / 60)]}')
                    # flow = modifier_list[run_count][1][int((step * step_length) / 60)]
                    # flow = modifier_list[int((step * step_length )/ 60)]
                    # print(f'Mainstream flowrate {flow} for {step} with step length {step_length}')
                    # sim.FlowDemand.set_mainstream_flow(flow, step, step_length)
                    flow_volume_mainstream = np.random.choice(state_training_samples + [1385 * 2 + 500])
                    # flow_volume_mainstream = 1385 * 2 + 500
                    sim.My_traci.FlowDemand.set_mainstream_flow(flow_volume_mainstream, step, step_length)

                    # No modification at the moment
                    sim.My_traci.FlowDemand.set_flow_on_ramp1(step, step_length)
                    sim.My_traci.FlowDemand.set_flow_on_ramp2(step, step_length)

                if step % (time_sampling_TTSreward * (1 / step_length)) == 0 and step > 0:
                    Ag.A1.TTS_sectionL1 += \
                        sim.My_traci.GetNumberVeh.section_L1() * (time_sampling_TTSreward / 3600)
                    # print(f'run TTS_L1 id: {id(Ag.A1.TTS_sectionL1)}, \n run TTS_L1: {Ag.A1.TTS_sectionL1}')
                    Ag.A1.TTS_sectionL2 += \
                        sim.My_traci.GetNumberVeh.section_L2() * (time_sampling_TTSreward / 3600)
                    Ag.A1.TTS_sectionL3 += \
                        sim.My_traci.GetNumberVeh.section_L3() * (time_sampling_TTSreward / 3600)

                    Ag.A2.TTS_sectionL1 += \
                        sim.My_traci.GetNumberVeh.section_L1() * (time_sampling_TTSreward / 3600)
                    Ag.A2.TTS_sectionL2 += \
                        sim.My_traci.GetNumberVeh.section_L2() * (time_sampling_TTSreward / 3600)
                    Ag.A2.TTS_sectionL3 += \
                        sim.My_traci.GetNumberVeh.section_L3() * (time_sampling_TTSreward / 3600)

                if step % (sampling_rate_states * (1 / step_length)) == 0 and step > 0:
                    j += 1

                    Ag.A1.TTS_Net_Sim += \
                        sim.My_traci.GetNumberVeh.numVehNet() * (sampling_rate_states / 3600)
                    Ag.A2.TTS_Net_Sim = Ag.A1.TTS_Net_Sim

                    # speedL1, speedL2, speedL3, speedL4 = currentSpeed_L1L2L3L4()
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
                        arr = np.array(speedList)
                        speedHeatMap = np.vstack([speedHeatMap, arr])

                    if step > 0 and step % (control_time_perid * (1 / step_length)) == 0 and i == 0:

                        # Get state spaces of agents
                        Ag.AgentState_x(j)

                        # Compute actions
                        Ag.Agent1Action(epsilon, run, numberSaveData, C)
                        Ag.Agent2Action(epsilon, run, numberSaveData, C)

                        # Seems this is not used in this version as such can be removed as I store prevActionWinIndex
                        Ag.A1.VSL_beforeA1[1] = Ag.A1.VSLspeed
                        Ag.A2.VSL_beforeA1[1] = Ag.A2.VSLspeed

                        Ag.A1.prevActionWinIndex = Ag.A1.actionWinIndex
                        Ag.A2.prevActionWinIndex = Ag.A2.actionWinIndex

                        speedLimitList = sim.My_traci.SetSpeedLimit.speed_VSLzones(Ag.A1.VSLspeed, Ag.A2.VSLspeed, \
                                                                                   Ag.A1.VSLposition, Ag.A2.VSLposition, \
                                                                                   plotHeatMap, speedLimitList)

                        Ag.A1.VSL_beforeA1[0] = Ag.A1.VSL_beforeA1[1]
                        Ag.A2.VSL_beforeA2[0] = Ag.A2.VSL_beforeA2[1]

                        Ag.updateCurrentSpeedStateAsPrevious(j)
                        Ag.updateCurrentDensityStateAsPrevious(j)

                        Ag.Clean_up_TTS_var()

                        i = 1
                        j = 0

                    elif step > 0 and step % (control_time_perid * (1 / step_length)) == 0 and i == 1:

                        Ag.UpdateRewards(j, speed_free_flow, alpha, beta)

                        # get Agents state
                        # Ag.AgentState_y(j)
                        Ag.AgentState_y(j)

                        # update Agent's policies
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

                        speedLimitList = sim.My_traci.SetSpeedLimit.speed_VSLzones(Ag.A1.VSLspeed, Ag.A2.VSLspeed, \
                                                                                   Ag.A1.VSLposition, Ag.A2.VSLposition, \
                                                                                   plotHeatMap, speedLimitList)

                        Ag.A1.VSL_beforeA1[0] = Ag.A1.VSL_beforeA1[1]
                        Ag.A2.VSL_beforeA2[0] = Ag.A2.VSL_beforeA2[1]

                        arra = np.array(speedLimitList)
                        speedLimitHeatMap = np.vstack([speedLimitHeatMap, arra])

                        if run % numberSaveData == 0 or run > end_learning:
                            print(f"Save data run: {run} and {numberSaveData} is {run%numberSaveData}")
                            controlFile = Ag.ControlFileVStack(controlFile, j)

                        Ag.updateCurrentSpeedStateAsPrevious(j)
                        Ag.updateCurrentDensityStateAsPrevious(j)
                        j = 0

                step += 1
            Ag.saveData(vsl_on_off, run, numberSaveData, path_results, controlFile, speedHeatMap, speedLimitHeatMap)
            Ag.Clean_up_TTS_Net_Sim_var()

            traci.close()
            run += 1
            x += 1
            print(f" End learning: {end_learning} and run {run}")
        return run
