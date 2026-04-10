import traci
import numpy as np

class My_traci:

    class FlowDemand:
        # mainstream flow demand that can be modified
        def set_mainstream_flow(flow, step, step_length):
            traci.calibrator.setFlow("cali_mainstream", step*step_length, (step*step_length)+60, flow, 33.3,\
                                    "myVehDist", "routedist_mainstream_flow",\
                                    departLane="free", departSpeed="max")

        # constant demand on-ramp1
        def set_flow_on_ramp1(step, step_length):    
            traci.calibrator.setFlow("cali_onramp1", step*step_length, (step*step_length)+60, \
                                        385, 22.2,\
                                        'myVehDist', 'routedist_onramp1_flow',\
                                        departLane="free", departSpeed='max')
        # picewise function for traffic demand at on-ramp 2 
        def set_flow_on_ramp2(step, step_length):
            x=(step*step_length)/60 # this gives current sim. minute
            if x>=0 and x<13:
                x1=0
                x2=13
                y1=200
                y2=200
                flow = ((y2-y1)/(x2-x1))*(x-x1)+y1
            elif x>=13 and x<16:
                x1=13
                x2=16
                y1=200
                y2=700
                flow = ((y2-y1)/(x2-x1))*(x-x1)+y1
            elif x>=16 and x<23:
                x1=16
                x2=23
                y1=700
                y2=700
                flow = ((y2-y1)/(x2-x1))*(x-x1)+y1
            elif x>=23 and x<26:
                x1=23
                x2=26
                y1=700
                y2=1315
                flow = ((y2-y1)/(x2-x1))*(x-x1)+y1
            elif x>=26 and x<64:
                x1=26
                x2=64
                y1=1315
                y2=1315
                flow = ((y2-y1)/(x2-x1))*(x-x1)+y1
            elif x>=64 and x<67:
                x1=64
                x2=67
                y1=1315
                y2=700
                flow = ((y2-y1)/(x2-x1))*(x-x1)+y1
            elif x>=67 and x<74:
                x1=67
                x2=74
                y1=700
                y2=700
                flow = ((y2-y1)/(x2-x1))*(x-x1)+y1
            elif x>=74 and x<77:
                x1=74
                x2=77
                y1=700
                y2=200
                flow = ((y2-y1)/(x2-x1))*(x-x1)+y1
            else:
                flow = 200

            traci.calibrator.setFlow("cali_onramp2", step*step_length, (step*step_length)+60, \
                                            flow, 22.2,\
                                            'myVehDist', 'routedist_onramp2_flow',\
                                            departLane="free", departSpeed='max')

            return flow

    class GetNumberVeh:

        def section_L0():
            # not in use for state description
            return 0

        def section_L1():
            numVeh_L1 = np.sum([traci.edge.getLastStepVehicleNumber("v61"),\
                                        traci.edge.getLastStepVehicleNumber("v62"),\
                                        traci.edge.getLastStepVehicleNumber("v63"),\
                                        traci.edge.getLastStepVehicleNumber("v64"),\
                                        traci.edge.getLastStepVehicleNumber("v65"),\
                                        traci.edge.getLastStepVehicleNumber("v66"),\
                                        traci.edge.getLastStepVehicleNumber("v67"),\
                                        traci.edge.getLastStepVehicleNumber("v68"),\
                                        traci.edge.getLastStepVehicleNumber("v69"),\
                                        traci.lane.getLastStepVehicleNumber("v70_1"),\
                                        traci.lane.getLastStepVehicleNumber("v70_2"),\
                                        traci.lane.getLastStepVehicleNumber("v71_1"),\
                                        traci.lane.getLastStepVehicleNumber("v71_2"),\
                                        traci.lane.getLastStepVehicleNumber("v72_1"),\
                                        traci.lane.getLastStepVehicleNumber("v72_2"),\
                                        traci.lane.getLastStepVehicleNumber("v73_1"),\
                                        traci.lane.getLastStepVehicleNumber("v73_2"),\
                                        traci.lane.getLastStepVehicleNumber("v74_1"),\
                                        traci.lane.getLastStepVehicleNumber("v74_2"),\
                                        traci.edge.getLastStepVehicleNumber("v75"),\
                                        traci.edge.getLastStepVehicleNumber("v76"),\
                                        traci.edge.getLastStepVehicleNumber("v77"),\
                                        traci.edge.getLastStepVehicleNumber("v78"),\
                                        traci.edge.getLastStepVehicleNumber("v79"),\
                                        traci.edge.getLastStepVehicleNumber("v80")])
            return numVeh_L1  
                    
        def section_L2():
            numVeh_L2 = np.sum([traci.edge.getLastStepVehicleNumber("v81"),\
                                            traci.edge.getLastStepVehicleNumber("v82"),\
                                            traci.edge.getLastStepVehicleNumber("v83"),\
                                            traci.edge.getLastStepVehicleNumber("v84"),\
                                            traci.edge.getLastStepVehicleNumber("v85"),\
                                            traci.edge.getLastStepVehicleNumber("v86"),\
                                            traci.edge.getLastStepVehicleNumber("v87"),\
                                            traci.edge.getLastStepVehicleNumber("v88"),\
                                            traci.edge.getLastStepVehicleNumber("v89"),\
                                            traci.edge.getLastStepVehicleNumber("v90"),\
                                            traci.edge.getLastStepVehicleNumber("v91"),\
                                            traci.edge.getLastStepVehicleNumber("v92"),\
                                            traci.edge.getLastStepVehicleNumber("v93"),\
                                            traci.edge.getLastStepVehicleNumber("v94"),\
                                            traci.edge.getLastStepVehicleNumber("v95"),\
                                            traci.edge.getLastStepVehicleNumber("v96"),\
                                            traci.edge.getLastStepVehicleNumber("v97"),\
                                            traci.edge.getLastStepVehicleNumber("v98"),\
                                            traci.edge.getLastStepVehicleNumber("v99"),\
                                            traci.edge.getLastStepVehicleNumber("v100")])
            return numVeh_L2

        def section_L3():
            numVeh_L3 = np.sum([traci.edge.getLastStepVehicleNumber("v101"),\
                                        traci.edge.getLastStepVehicleNumber("v102"),\
                                        traci.edge.getLastStepVehicleNumber("v103"),\
                                        traci.edge.getLastStepVehicleNumber("v104"),\
                                        traci.edge.getLastStepVehicleNumber("v105"),\
                                        traci.edge.getLastStepVehicleNumber("v106"),\
                                        traci.lane.getLastStepVehicleNumber("v107_2"),\
                                        traci.lane.getLastStepVehicleNumber("v107_3"),\
                                        traci.lane.getLastStepVehicleNumber("v108_1"),\
                                        traci.lane.getLastStepVehicleNumber("v108_2"),\
                                        traci.lane.getLastStepVehicleNumber("v109_1"),\
                                        traci.lane.getLastStepVehicleNumber("v109_2"),\
                                        traci.lane.getLastStepVehicleNumber("v110_1"),\
                                        traci.lane.getLastStepVehicleNumber("v110_2"),\
                                        traci.lane.getLastStepVehicleNumber("v111_1"),\
                                        traci.lane.getLastStepVehicleNumber("v111_2")])
            return numVeh_L3

        def haltingVeh_section_L3_OnRamp():
            haltingVehs_bottleneck_ramp = np.sum([traci.lane.getLastStepHaltingNumber("18to11_0"),\
                                        traci.lane.getLastStepHaltingNumber("v107_0"),\
                                        traci.lane.getLastStepHaltingNumber("v108_0"),\
                                        traci.lane.getLastStepHaltingNumber("v109_0"),\
                                        traci.lane.getLastStepHaltingNumber("v110_0"),\
                                        traci.lane.getLastStepHaltingNumber("v111_0")])
            return haltingVehs_bottleneck_ramp    
            
        
        def numVehNet():
            numVehNet = traci.vehicle.getIDCount()    
            return numVehNet

    class GetSpeed:

        def section_L0():
            return 0
        
        #=======================A2
        def section_L1():
            speedL1 = np.mean([traci.edge.getLastStepMeanSpeed("v61"),\
                                            traci.edge.getLastStepMeanSpeed("v62"),\
                                            traci.edge.getLastStepMeanSpeed("v63"),\
                                            traci.edge.getLastStepMeanSpeed("v64"),\
                                            traci.edge.getLastStepMeanSpeed("v65"),\
                                            traci.edge.getLastStepMeanSpeed("v66"),\
                                            traci.edge.getLastStepMeanSpeed("v67"),\
                                            traci.edge.getLastStepMeanSpeed("v68"),\
                                            traci.edge.getLastStepMeanSpeed("v69"),\
                                            traci.lane.getLastStepMeanSpeed("v70_1"),\
                                            traci.lane.getLastStepMeanSpeed("v70_2"),\
                                            traci.lane.getLastStepMeanSpeed("v71_1"),\
                                            traci.lane.getLastStepMeanSpeed("v71_2"),\
                                            traci.lane.getLastStepMeanSpeed("v72_1"),\
                                            traci.lane.getLastStepMeanSpeed("v72_2"),\
                                            traci.lane.getLastStepMeanSpeed("v73_1"),\
                                            traci.lane.getLastStepMeanSpeed("v73_2"),\
                                            traci.lane.getLastStepMeanSpeed("v74_1"),\
                                            traci.lane.getLastStepMeanSpeed("v74_2"),\
                                            traci.edge.getLastStepMeanSpeed("v75"),\
                                            traci.edge.getLastStepMeanSpeed("v76"),\
                                            traci.edge.getLastStepMeanSpeed("v77"),\
                                            traci.edge.getLastStepMeanSpeed("v78"),\
                                            traci.edge.getLastStepMeanSpeed("v79"),\
                                            traci.edge.getLastStepMeanSpeed("v80")])
            return speedL1

        #============= A1 and A2
        def section_L2():
            speedL2 = np.mean([traci.edge.getLastStepMeanSpeed("v81"),\
                                traci.edge.getLastStepMeanSpeed("v82"),\
                                traci.edge.getLastStepMeanSpeed("v83"),\
                                traci.edge.getLastStepMeanSpeed("v84"),\
                                traci.edge.getLastStepMeanSpeed("v85"),\
                                traci.edge.getLastStepMeanSpeed("v86"),\
                                traci.edge.getLastStepMeanSpeed("v87"),\
                                traci.edge.getLastStepMeanSpeed("v88"),\
                                traci.edge.getLastStepMeanSpeed("v89"),\
                                traci.edge.getLastStepMeanSpeed("v90"),\
                                traci.edge.getLastStepMeanSpeed("v91"),\
                                traci.edge.getLastStepMeanSpeed("v92"),\
                                traci.edge.getLastStepMeanSpeed("v93"),\
                                traci.edge.getLastStepMeanSpeed("v94"),\
                                traci.edge.getLastStepMeanSpeed("v95"),\
                                traci.edge.getLastStepMeanSpeed("v96"),\
                                traci.edge.getLastStepMeanSpeed("v97"),\
                                traci.edge.getLastStepMeanSpeed("v98"),\
                                traci.edge.getLastStepMeanSpeed("v99"),\
                                traci.edge.getLastStepMeanSpeed("v100")])
            return speedL2

        def section_L3():
            # 12.2.2020 avg. speed Cell 3 (on-ramp area)
            speedL3 = np.mean([traci.edge.getLastStepMeanSpeed("v101"),\
                                traci.edge.getLastStepMeanSpeed("v102"),\
                                traci.edge.getLastStepMeanSpeed("v103"),\
                                traci.edge.getLastStepMeanSpeed("v104"),\
                                traci.edge.getLastStepMeanSpeed("v105"),\
                                traci.edge.getLastStepMeanSpeed("v106"),\
                                traci.lane.getLastStepMeanSpeed("v107_2"),\
                                traci.lane.getLastStepMeanSpeed("v107_3"),\
                                traci.lane.getLastStepMeanSpeed("v108_1"),\
                                traci.lane.getLastStepMeanSpeed("v108_2"),\
                                traci.lane.getLastStepMeanSpeed("v109_1"),\
                                traci.lane.getLastStepMeanSpeed("v109_2"),\
                                traci.lane.getLastStepMeanSpeed("v110_1"),\
                                traci.lane.getLastStepMeanSpeed("v110_2"),\
                                traci.lane.getLastStepMeanSpeed("v111_1"),\
                                traci.lane.getLastStepMeanSpeed("v111_2")])
            return speedL3
    

    
    class GetDensity:

        def section_L0():
            return 0

        def section_L1(numLanes):
            currentDensityL1 = (1/(1000/1000))*(1/numLanes)*np.sum([traci.edge.getLastStepVehicleNumber("v61"),\
                                                                        traci.edge.getLastStepVehicleNumber("v62"),\
                                                                        traci.edge.getLastStepVehicleNumber("v63"),\
                                                                        traci.edge.getLastStepVehicleNumber("v64"),\
                                                                        traci.edge.getLastStepVehicleNumber("v65"),\
                                                                        traci.edge.getLastStepVehicleNumber("v66"),\
                                                                        traci.edge.getLastStepVehicleNumber("v67"),\
                                                                        traci.edge.getLastStepVehicleNumber("v68"),\
                                                                        traci.edge.getLastStepVehicleNumber("v69"),\
                                                                        traci.lane.getLastStepVehicleNumber("v70_1"),\
                                                                        traci.lane.getLastStepVehicleNumber("v70_2"),\
                                                                        traci.lane.getLastStepVehicleNumber("v71_1"),\
                                                                        traci.lane.getLastStepVehicleNumber("v71_2"),\
                                                                        traci.lane.getLastStepVehicleNumber("v72_1"),\
                                                                        traci.lane.getLastStepVehicleNumber("v72_2"),\
                                                                        traci.lane.getLastStepVehicleNumber("v73_1"),\
                                                                        traci.lane.getLastStepVehicleNumber("v73_2"),\
                                                                        traci.lane.getLastStepVehicleNumber("v74_1"),\
                                                                        traci.lane.getLastStepVehicleNumber("v74_2"),\
                                                                        traci.edge.getLastStepVehicleNumber("v75"),\
                                                                        traci.edge.getLastStepVehicleNumber("v76"),\
                                                                        traci.edge.getLastStepVehicleNumber("v77"),\
                                                                        traci.edge.getLastStepVehicleNumber("v78"),\
                                                                        traci.edge.getLastStepVehicleNumber("v79"),\
                                                                        traci.edge.getLastStepVehicleNumber("v80")])
            return currentDensityL1
        
        def section_L2(numLanes):
            currentDensityL2 = (1/(1000/1000))*(1/numLanes)*np.sum([traci.edge.getLastStepVehicleNumber("v81"),\
                                                                    traci.edge.getLastStepVehicleNumber("v82"),\
                                                                    traci.edge.getLastStepVehicleNumber("v83"),\
                                                                    traci.edge.getLastStepVehicleNumber("v84"),\
                                                                    traci.edge.getLastStepVehicleNumber("v85"),\
                                                                    traci.edge.getLastStepVehicleNumber("v86"),\
                                                                    traci.edge.getLastStepVehicleNumber("v87"),\
                                                                    traci.edge.getLastStepVehicleNumber("v88"),\
                                                                    traci.edge.getLastStepVehicleNumber("v89"),\
                                                                    traci.edge.getLastStepVehicleNumber("v90"),\
                                                                    traci.edge.getLastStepVehicleNumber("v91"),\
                                                                    traci.edge.getLastStepVehicleNumber("v92"),\
                                                                    traci.edge.getLastStepVehicleNumber("v93"),\
                                                                    traci.edge.getLastStepVehicleNumber("v94"),\
                                                                    traci.edge.getLastStepVehicleNumber("v95"),\
                                                                    traci.edge.getLastStepVehicleNumber("v96"),\
                                                                    traci.edge.getLastStepVehicleNumber("v97"),\
                                                                    traci.edge.getLastStepVehicleNumber("v98"),\
                                                                    traci.edge.getLastStepVehicleNumber("v99"),\
                                                                    traci.edge.getLastStepVehicleNumber("v100")])
            return currentDensityL2

        def section_L3(numLanes):
            currentDensityL3 = (1/(550/1000))*(1/numLanes)*(np.sum([traci.edge.getLastStepVehicleNumber("v101"),\
                                                                    traci.edge.getLastStepVehicleNumber("v102"),\
                                                                    traci.edge.getLastStepVehicleNumber("v103"),\
                                                                    traci.edge.getLastStepVehicleNumber("v104"),\
                                                                    traci.edge.getLastStepVehicleNumber("v105"),\
                                                                    traci.edge.getLastStepVehicleNumber("v106"),\
                                                                    traci.lane.getLastStepVehicleNumber("v107_2"),\
                                                                    traci.lane.getLastStepVehicleNumber("v107_3"),\
                                                                    traci.lane.getLastStepVehicleNumber("v108_1"),\
                                                                    traci.lane.getLastStepVehicleNumber("v108_2"),\
                                                                    traci.lane.getLastStepVehicleNumber("v109_1"),\
                                                                    traci.lane.getLastStepVehicleNumber("v109_2"),\
                                                                    traci.lane.getLastStepVehicleNumber("v110_1"),\
                                                                    traci.lane.getLastStepVehicleNumber("v110_2"),\
                                                                    traci.lane.getLastStepVehicleNumber("v111_1"),\
                                                                    traci.lane.getLastStepVehicleNumber("v111_2")]))
            return currentDensityL3
        
          
    class HeatMapPlot:
        """
        Entire model is divided into smaller segments in which macroscopic parameter (speed) is used for the heat map plot
        """
        def speedPerSegments():
            speedList=[]
            for i in range(1,161):
                speedList.append(traci.edge.getLastStepMeanSpeed("v"+str(i)))
            return speedList


    class SetSpeedLimit:
        """
        Dynamic VSL zones allocation!
        Different zones their lengths and combinations with different speed limits are
        dynamically activated during the simulation run according to agents' decisions via fn. setSpeedLimit().
        Function setSpeedLimit() is also used to generate list of data for VSL HeatMap plot! 
        """
        def speed_VSLzones(a1, a2, VSL1pos, VSL2pos, plotHeatMap, speedLimitList):

        # ========Initial speed limit for A1 section
            initSpeed = 33.33
            traci.edge.setMaxSpeed("v61", initSpeed)
            traci.edge.setMaxSpeed("v62", initSpeed)
            traci.edge.setMaxSpeed("v63", initSpeed)
            traci.edge.setMaxSpeed("v64", initSpeed)
            traci.edge.setMaxSpeed("v65", initSpeed)
            traci.edge.setMaxSpeed("v66", initSpeed)
            traci.edge.setMaxSpeed("v67", initSpeed)
            traci.edge.setMaxSpeed("v68", initSpeed)
            traci.edge.setMaxSpeed("v69", initSpeed)
            traci.lane.setMaxSpeed("v70_1", initSpeed)
            traci.lane.setMaxSpeed("v70_2", initSpeed)
            traci.lane.setMaxSpeed("v71_1", initSpeed)
            traci.lane.setMaxSpeed("v71_2", initSpeed)
            traci.lane.setMaxSpeed("v72_1", initSpeed)
            traci.lane.setMaxSpeed("v72_2", initSpeed)
            traci.lane.setMaxSpeed("v73_1", initSpeed)
            traci.lane.setMaxSpeed("v73_2", initSpeed)
            traci.lane.setMaxSpeed("v74_1", initSpeed)
            traci.lane.setMaxSpeed("v74_2", initSpeed)
            traci.lane.setMaxSpeed(":n8_1_0", initSpeed)
            traci.lane.setMaxSpeed(":n8_1_1", initSpeed)
            traci.edge.setMaxSpeed("v75", initSpeed)
            traci.edge.setMaxSpeed("v76", initSpeed)
            traci.edge.setMaxSpeed("v77", initSpeed)
            traci.edge.setMaxSpeed("v78", initSpeed)
            traci.edge.setMaxSpeed("v79", initSpeed)
            traci.edge.setMaxSpeed("v80", initSpeed)
            
            
            if plotHeatMap==1:   
                speedLimitList[61]=initSpeed
                speedLimitList[62]=initSpeed
                speedLimitList[63]=initSpeed
                speedLimitList[64]=initSpeed
                speedLimitList[65]=initSpeed
                speedLimitList[66]=initSpeed
                speedLimitList[67]=initSpeed
                speedLimitList[68]=initSpeed
                speedLimitList[69]=initSpeed
                speedLimitList[70]=initSpeed
                speedLimitList[71]=initSpeed
                speedLimitList[72]=initSpeed
                speedLimitList[73]=initSpeed
                speedLimitList[74]=initSpeed
                speedLimitList[75]=initSpeed
                speedLimitList[76]=initSpeed
                speedLimitList[77]=initSpeed
                speedLimitList[78]=initSpeed
                speedLimitList[79]=initSpeed
                speedLimitList[80]=initSpeed

        # ============Initial speed limit for A1's controlled section
            if VSL1pos==1:
                traci.edge.setMaxSpeed("v61", a1)
                traci.edge.setMaxSpeed("v62", a1)
                traci.edge.setMaxSpeed("v63", a1)
                traci.edge.setMaxSpeed("v64", a1)
                traci.edge.setMaxSpeed("v65", a1)
                traci.edge.setMaxSpeed("v66", a1)
                traci.edge.setMaxSpeed("v67", a1)
                traci.edge.setMaxSpeed("v68", a1)
                traci.edge.setMaxSpeed("v69", a1)
                traci.lane.setMaxSpeed("v70_1", a1)
                traci.lane.setMaxSpeed("v70_2", a1)
                traci.lane.setMaxSpeed("v71_1", a1)
                traci.lane.setMaxSpeed("v71_2", a1)
                traci.lane.setMaxSpeed("v72_1", a1)
                traci.lane.setMaxSpeed("v72_2", a1)
                traci.lane.setMaxSpeed("v73_1", a1)
                traci.lane.setMaxSpeed("v73_2", a1)
                traci.lane.setMaxSpeed("v74_1", a1)
                traci.lane.setMaxSpeed("v74_2", a1)
                traci.lane.setMaxSpeed(":n8_1_0", a1)
                traci.lane.setMaxSpeed(":n8_1_1", a1)
                traci.edge.setMaxSpeed("v75", a1)
                traci.edge.setMaxSpeed("v76", a1)
                traci.edge.setMaxSpeed("v77", a1)
                traci.edge.setMaxSpeed("v78", a1)
                traci.edge.setMaxSpeed("v79", a1)
                traci.edge.setMaxSpeed("v80", a1)

                if plotHeatMap==1:
                    speedLimitList[61]=a1
                    speedLimitList[62]=a1
                    speedLimitList[63]=a1
                    speedLimitList[64]=a1
                    speedLimitList[65]=a1
                    speedLimitList[66]=a1
                    speedLimitList[67]=a1
                    speedLimitList[68]=a1
                    speedLimitList[69]=a1
                    speedLimitList[70]=a1
                    speedLimitList[71]=a1
                    speedLimitList[72]=a1
                    speedLimitList[73]=a1
                    speedLimitList[74]=a1
                    speedLimitList[75]=a1
                    speedLimitList[76]=a1
                    speedLimitList[77]=a1
                    speedLimitList[78]=a1
                    speedLimitList[79]=a1
                    speedLimitList[80]=a1
                
                
            elif VSL1pos==2:
                traci.edge.setMaxSpeed("v66", a1)
                traci.edge.setMaxSpeed("v67", a1)
                traci.edge.setMaxSpeed("v68", a1)
                traci.edge.setMaxSpeed("v69", a1)
                traci.lane.setMaxSpeed("v70_1", a1)
                traci.lane.setMaxSpeed("v70_2", a1)
                traci.lane.setMaxSpeed("v71_1", a1)
                traci.lane.setMaxSpeed("v71_2", a1)
                traci.lane.setMaxSpeed("v72_1", a1)
                traci.lane.setMaxSpeed("v72_2", a1)
                traci.lane.setMaxSpeed("v73_1", a1)
                traci.lane.setMaxSpeed("v73_2", a1)
                traci.lane.setMaxSpeed("v74_1", a1)
                traci.lane.setMaxSpeed("v74_2", a1)
                traci.lane.setMaxSpeed(":n8_1_0", a1)
                traci.lane.setMaxSpeed(":n8_1_1", a1)
                traci.edge.setMaxSpeed("v75", a1)
                traci.edge.setMaxSpeed("v76", a1)
                traci.edge.setMaxSpeed("v77", a1)
                traci.edge.setMaxSpeed("v78", a1)
                traci.edge.setMaxSpeed("v79", a1)
                traci.edge.setMaxSpeed("v80", a1)
                
                if plotHeatMap==1: 
                    speedLimitList[66]=a1
                    speedLimitList[67]=a1
                    speedLimitList[68]=a1
                    speedLimitList[69]=a1
                    speedLimitList[70]=a1
                    speedLimitList[71]=a1
                    speedLimitList[72]=a1
                    speedLimitList[73]=a1
                    speedLimitList[74]=a1
                    speedLimitList[75]=a1
                    speedLimitList[76]=a1
                    speedLimitList[77]=a1
                    speedLimitList[78]=a1
                    speedLimitList[79]=a1
                    speedLimitList[80]=a1

                
            elif VSL1pos==3:
                traci.lane.setMaxSpeed("v71_1", a1)
                traci.lane.setMaxSpeed("v71_2", a1)
                traci.lane.setMaxSpeed("v72_1", a1)
                traci.lane.setMaxSpeed("v72_2", a1)
                traci.lane.setMaxSpeed("v73_1", a1)
                traci.lane.setMaxSpeed("v73_2", a1)
                traci.lane.setMaxSpeed("v74_1", a1)
                traci.lane.setMaxSpeed("v74_2", a1)
                traci.lane.setMaxSpeed(":n8_1_0", a1)
                traci.lane.setMaxSpeed(":n8_1_1", a1)
                traci.edge.setMaxSpeed("v75", a1)
                traci.edge.setMaxSpeed("v76", a1)
                traci.edge.setMaxSpeed("v77", a1)
                traci.edge.setMaxSpeed("v78", a1)
                traci.edge.setMaxSpeed("v79", a1)
                traci.edge.setMaxSpeed("v80", a1)

                if plotHeatMap==1:
                    speedLimitList[71]=a1
                    speedLimitList[72]=a1
                    speedLimitList[73]=a1
                    speedLimitList[74]=a1
                    speedLimitList[75]=a1
                    speedLimitList[76]=a1
                    speedLimitList[77]=a1
                    speedLimitList[78]=a1
                    speedLimitList[79]=a1
                    speedLimitList[80]=a1
                
            else:
                traci.edge.setMaxSpeed("v76", a1)
                traci.edge.setMaxSpeed("v77", a1)
                traci.edge.setMaxSpeed("v78", a1)
                traci.edge.setMaxSpeed("v79", a1)
                traci.edge.setMaxSpeed("v80", a1)
                
                if plotHeatMap==1:
                    speedLimitList[76]=a1
                    speedLimitList[77]=a1
                    speedLimitList[78]=a1
                    speedLimitList[79]=a1
                    speedLimitList[80]=a1
                
            
        # ============Initial speed limit for A2's controlled section
            traci.edge.setMaxSpeed("v81", initSpeed)
            traci.edge.setMaxSpeed("v82", initSpeed)
            traci.edge.setMaxSpeed("v83", initSpeed)
            traci.edge.setMaxSpeed("v84", initSpeed)
            traci.edge.setMaxSpeed("v85", initSpeed)
            traci.edge.setMaxSpeed("v86", initSpeed)
            traci.edge.setMaxSpeed("v87", initSpeed)
            traci.edge.setMaxSpeed("v88", initSpeed)
            traci.edge.setMaxSpeed("v89", initSpeed)
            traci.edge.setMaxSpeed("v90", initSpeed)
            traci.edge.setMaxSpeed("v91", initSpeed)
            traci.edge.setMaxSpeed("v92", initSpeed)
            traci.edge.setMaxSpeed("v93", initSpeed)
            traci.edge.setMaxSpeed("v94", initSpeed)
            traci.edge.setMaxSpeed("v95", initSpeed)
            traci.edge.setMaxSpeed("v96", initSpeed)
            traci.edge.setMaxSpeed("v97", initSpeed)
            traci.edge.setMaxSpeed("v98", initSpeed)
            traci.edge.setMaxSpeed("v99", initSpeed)
            traci.edge.setMaxSpeed("v100", initSpeed)
            
            if plotHeatMap==1:     
                speedLimitList[81]=initSpeed
                speedLimitList[82]=initSpeed
                speedLimitList[83]=initSpeed
                speedLimitList[84]=initSpeed
                speedLimitList[85]=initSpeed
                speedLimitList[86]=initSpeed
                speedLimitList[87]=initSpeed
                speedLimitList[88]=initSpeed
                speedLimitList[89]=initSpeed
                speedLimitList[90]=initSpeed
                speedLimitList[91]=initSpeed
                speedLimitList[92]=initSpeed
                speedLimitList[93]=initSpeed
                speedLimitList[94]=initSpeed
                speedLimitList[95]=initSpeed
                speedLimitList[96]=initSpeed
                speedLimitList[97]=initSpeed
                speedLimitList[98]=initSpeed
                speedLimitList[99]=initSpeed
                speedLimitList[100]=initSpeed
            
        # Set speed limit for specific zone A2
            if VSL2pos==1:
                traci.edge.setMaxSpeed("v81", a2)
                traci.edge.setMaxSpeed("v82", a2)
                traci.edge.setMaxSpeed("v83", a2)
                traci.edge.setMaxSpeed("v84", a2)
                traci.edge.setMaxSpeed("v85", a2)
                traci.edge.setMaxSpeed("v86", a2)
                traci.edge.setMaxSpeed("v87", a2)
                traci.edge.setMaxSpeed("v88", a2)
                traci.edge.setMaxSpeed("v89", a2)
                traci.edge.setMaxSpeed("v90", a2)
                traci.edge.setMaxSpeed("v91", a2)
                traci.edge.setMaxSpeed("v92", a2)
                traci.edge.setMaxSpeed("v93", a2)
                traci.edge.setMaxSpeed("v94", a2)
                traci.edge.setMaxSpeed("v95", a2)
                traci.edge.setMaxSpeed("v96", a2)
                traci.edge.setMaxSpeed("v97", a2)
                traci.edge.setMaxSpeed("v98", a2)
                traci.edge.setMaxSpeed("v99", a2)
                traci.edge.setMaxSpeed("v100", a2)
                
                if plotHeatMap==1:     
                    speedLimitList[81]=a2
                    speedLimitList[82]=a2
                    speedLimitList[83]=a2
                    speedLimitList[84]=a2
                    speedLimitList[85]=a2
                    speedLimitList[86]=a2
                    speedLimitList[87]=a2
                    speedLimitList[88]=a2
                    speedLimitList[89]=a2
                    speedLimitList[90]=a2
                    speedLimitList[91]=a2
                    speedLimitList[92]=a2
                    speedLimitList[93]=a2
                    speedLimitList[94]=a2
                    speedLimitList[95]=a2
                    speedLimitList[96]=a2
                    speedLimitList[97]=a2
                    speedLimitList[98]=a2
                    speedLimitList[99]=a2
                    speedLimitList[100]=a2
                
                
                
            elif VSL2pos==2:
                traci.edge.setMaxSpeed("v81", a2)
                traci.edge.setMaxSpeed("v82", a2)
                traci.edge.setMaxSpeed("v83", a2)
                traci.edge.setMaxSpeed("v84", a2)
                traci.edge.setMaxSpeed("v85", a2)
                traci.edge.setMaxSpeed("v86", a2)
                traci.edge.setMaxSpeed("v87", a2)
                traci.edge.setMaxSpeed("v88", a2)
                traci.edge.setMaxSpeed("v89", a2)
                traci.edge.setMaxSpeed("v90", a2)
                traci.edge.setMaxSpeed("v91", a2)
                traci.edge.setMaxSpeed("v92", a2)
                traci.edge.setMaxSpeed("v93", a2)
                traci.edge.setMaxSpeed("v94", a2)
                traci.edge.setMaxSpeed("v95", a2)

                if plotHeatMap==1:     
                    speedLimitList[81]=a2
                    speedLimitList[82]=a2
                    speedLimitList[83]=a2
                    speedLimitList[84]=a2
                    speedLimitList[85]=a2
                    speedLimitList[86]=a2
                    speedLimitList[87]=a2
                    speedLimitList[88]=a2
                    speedLimitList[89]=a2
                    speedLimitList[90]=a2
                    speedLimitList[91]=a2
                    speedLimitList[92]=a2
                    speedLimitList[93]=a2
                    speedLimitList[94]=a2
                    speedLimitList[95]=a2

                
            elif VSL2pos==3:
                traci.edge.setMaxSpeed("v81", a2)
                traci.edge.setMaxSpeed("v82", a2)
                traci.edge.setMaxSpeed("v83", a2)
                traci.edge.setMaxSpeed("v84", a2)
                traci.edge.setMaxSpeed("v85", a2)
                traci.edge.setMaxSpeed("v86", a2)
                traci.edge.setMaxSpeed("v87", a2)
                traci.edge.setMaxSpeed("v88", a2)
                traci.edge.setMaxSpeed("v89", a2)
                traci.edge.setMaxSpeed("v90", a2)
                
                if plotHeatMap==1:     
                    speedLimitList[81]=a2
                    speedLimitList[82]=a2
                    speedLimitList[83]=a2
                    speedLimitList[84]=a2
                    speedLimitList[85]=a2
                    speedLimitList[86]=a2
                    speedLimitList[87]=a2
                    speedLimitList[88]=a2
                    speedLimitList[89]=a2
                    speedLimitList[90]=a2

            else:
                traci.edge.setMaxSpeed("v81", a2)
                traci.edge.setMaxSpeed("v82", a2)
                traci.edge.setMaxSpeed("v83", a2)
                traci.edge.setMaxSpeed("v84", a2)
                traci.edge.setMaxSpeed("v85", a2)

                if plotHeatMap==1:     
                    speedLimitList[81]=a2
                    speedLimitList[82]=a2
                    speedLimitList[83]=a2
                    speedLimitList[84]=a2
                    speedLimitList[85]=a2

            return speedLimitList