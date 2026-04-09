import traci
import numpy as np

class FlowDemand:
    # mainstream flow demand that can be modified
    def set_mainstream_flow(flow, step, step_length):
        traci.calibrator.setFlow("cali_mainstream", step*step_length, (step*step_length)+60, flow, 33.3,\
                                "myVehDist", "routedist_mainstream_flow",\
                                departLane="free", departSpeed="max")

    # constant demand on-ramp1
    def set_flow_on_ramp1(step, step_length):    
        traci.calibrator.setFlow("cali_onramp1", step*step_length, (step*step_length)+60, \
                                    300, 22.2,\
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

class GetNumberOfVeh:
    def numVehNet():
        numberOfVeh = traci.vehicle.getIDCount()
        return numberOfVeh

class GetCurrentDensity:
    def section4():
        # 2 numLanes
        # 550 m section lenght
        den_Sec4 = (1/(550/1000))*(1/2)*(np.sum([traci.edge.getLastStepVehicleNumber("v101"),\
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
        return den_Sec4
    
    
class SetSpeedLimit:
    def section3(a3):
    # Set speed limit for specific zone section3
        traci.edge.setMaxSpeed("v81", a3)
        traci.edge.setMaxSpeed("v82", a3)
        traci.edge.setMaxSpeed("v83", a3)
        traci.edge.setMaxSpeed("v84", a3)
        traci.edge.setMaxSpeed("v85", a3)
        traci.edge.setMaxSpeed("v86", a3)
        traci.edge.setMaxSpeed("v87", a3)
        traci.edge.setMaxSpeed("v88", a3)
        traci.edge.setMaxSpeed("v89", a3)
        traci.edge.setMaxSpeed("v90", a3)
        traci.edge.setMaxSpeed("v91", a3)
        traci.edge.setMaxSpeed("v92", a3)
        traci.edge.setMaxSpeed("v93", a3)
        traci.edge.setMaxSpeed("v94", a3)
        traci.edge.setMaxSpeed("v95", a3)
        traci.edge.setMaxSpeed("v96", a3)
        traci.edge.setMaxSpeed("v97", a3)
        traci.edge.setMaxSpeed("v98", a3)
        traci.edge.setMaxSpeed("v99", a3)
        traci.edge.setMaxSpeed("v100", a3)
