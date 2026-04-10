import numpy as np

class IterAgents(type):
    def __iter__(cls):
        return iter(cls._allAgents)

class CommonAttributes(metaclass=IterAgents):

    _allAgents=[]
    
    __slots__ = ("Q_LPi", "Q_RPi", "W_LPi", "W_RPi", "Num_Visited_LPi_x_ak",\
                "Num_Visited_RPi_x_ak", "Num_Visited_LPi_x_ai", "Num_Visited_RPi_x_ai",\
                "Num_Visited_LPi_x", "Num_Visited_RPi_x",\
                "actions_mps", "actions_kmph", "prevActionWinIndex",\
                "alpha_LPi_ak", "alpha_LPi_ai", "alpha_W_LPi", "alter_LPi_W",\
                "alpha_RPi_ak", "alpha_RPi_ai", "alpha_W_RPi", "alter_RPi_W",\
                "actionSub_y", "x_LPi", "x_old_LPi", "y_LPi", "R", "VSLposition", "VSLspeed",\
                "actionWant", "actionWant_LPi", "actionWant_RPi", "actionControl_LPi",\
                "actionControl_RPi", "LPiWin", "RPiWin", "policyWin", "actionWinIndex",\
                "speedSectionL0", "speedSectionL1", "speedSectionL2", "speedSectionL3",\
                "densitySectionL0", "densitySectionL1", "densitySectionL2", "densitySectionL3",\
                "TTS_sectionL0", "TTS_sectionL1", "TTS_sectionL2", "TTS_sectionL3", "TTS_sectionL3_haltingVeh_ramp", "TTS_Net_Sim",\
                "VSL_beforeA1", "VSL_beforeA2",\
                "TTS_old_sectionL0", "TTS_old_sectionL1", "TTS_old_sectionL2", "TTS_old_sectionL3")
    
    def __init__(self):

        self._allAgents.append(self)

        self.Q_LPi=[np.zeros((4608,8)), np.zeros((4608,8))]
        self.Q_RPi=[np.zeros((4608,8)), np.zeros((4608,8))]
        self.W_LPi=[np.zeros((4608,1)), np.zeros((4608,1))]
        self.W_RPi= [np.zeros((4608,1)), np.zeros((4608,1))]
        self.Num_Visited_LPi_x_ak= [np.zeros((4608,8)), np.zeros((4608,8))]
        self.Num_Visited_RPi_x_ak= [np.zeros((4608,8)), np.zeros((4608,8))]
        self.Num_Visited_LPi_x_ai= [np.zeros((4608,1)), np.zeros((4608,1))] 
        self.Num_Visited_RPi_x_ai= [np.zeros((4608,1)), np.zeros((4608,1))]
        self.Num_Visited_LPi_x= [np.zeros((4608,1)), np.zeros((4608,1))] 
        self.Num_Visited_RPi_x= [np.zeros((4608,1)), np.zeros((4608,1))]
        self.actions_mps= np.array([np.round((np.array([60, 60, 80, 80, 100, 100, 120, 120]))*(1/3.6),2), np.array([1,3,1,3,1,3,1,3])]) 
        self.actions_kmph= np.array([np.array([60, 60, 80, 80, 100, 100, 120, 120]), np.array([1,3,1,3,1,3,1,3])])
        self.prevActionWinIndex = (np.array([60, 60, 80, 80, 100, 100, 120, 120]).size)-1
        self.alpha_LPi_ak= np.array([0.0, 0.0])
        self.alpha_LPi_ai= np.array([0.0, 0.0]) 
        self.alpha_W_LPi= np.array([0.0, 0.0]) 
        self.alter_LPi_W= np.array([0.0, 0.0])
        self.alpha_RPi_ak= np.array([0.0, 0.0]) 
        self.alpha_RPi_ai= np.array([0.0, 0.0]) 
        self.alpha_W_RPi= np.array([0.0, 0.0])
        self.alter_RPi_W= np.array([0.0, 0.0])
        self.actionSub_y= [np.array([0]), np.array([0])]
        self.x_LPi= np.array([0, 0])
        self.x_old_LPi= np.array([0, 0]) 
        self.y_LPi= np.array([0, 0])
        self.R= np.array([0.0, 0.0]) 
        self.VSLposition= 0
        self.VSLspeed= 0
        self.actionWant= 0
        self.actionWant_LPi= [0, 0]
        self.actionWant_RPi= [0, 0]
        self.actionControl_LPi= [0, 0]
        self.actionControl_RPi= [0, 0]
        self.LPiWin= "" 
        self.RPiWin= "" 
        self.policyWin= "" 
        self.actionWinIndex= 0
        self.speedSectionL0= np.array([0.0,0.0,0.0,0.0])
        self.speedSectionL1= np.array([0.0,0.0,0.0,0.0]) 
        self.speedSectionL2= np.array([0.0,0.0,0.0,0.0]) 
        self.speedSectionL3= np.array([0.0,0.0,0.0,0.0])
        self.densitySectionL0= np.array([0.0,0.0,0.0,0.0]) 
        self.densitySectionL1= np.array([0.0,0.0,0.0,0.0]) 
        self.densitySectionL2= np.array([0.0,0.0,0.0,0.0]) 
        self.densitySectionL3= np.array([0.0,0.0,0.0,0.0])
        self.TTS_sectionL0= 0
        self.TTS_sectionL1= 0 
        self.TTS_sectionL2= 0
        self.TTS_sectionL3= 0
        self.TTS_sectionL3_haltingVeh_ramp= 0 
        self.TTS_Net_Sim= 0
        self.VSL_beforeA1= np.round(np.array([120, 120])*(1/(3.6)),2) 
        self.VSL_beforeA2= np.round(np.array([120, 120])*(1/(3.6)),2)
        self.TTS_old_sectionL0= 0
        self.TTS_old_sectionL1= 0 
        self.TTS_old_sectionL2= 0 
        self.TTS_old_sectionL3= 0












# agents_common_attributes={"Q_LPi= [np.zeros((4608,8)), np.zeros((4608,8))
# "Q_RPi= [np.zeros((4608,8)), np.zeros((4608,8)) 
# "W_LPi= [np.zeros((4608,1)), np.zeros((4608,1)) 
# "W_RPi= [np.zeros((4608,1)), np.zeros((4608,1))
# "Num_Visited_LPi_x_ak= [np.zeros((4608,8)), np.zeros((4608,8))
# "Num_Visited_RPi_x_ak= [np.zeros((4608,8)), np.zeros((4608,8))
# "Num_Visited_LPi_x_ai= [np.zeros((4608,1)), np.zeros((4608,1)) 
# "Num_Visited_RPi_x_ai= [np.zeros((4608,1)), np.zeros((4608,1))
# "Num_Visited_LPi_x= [np.zeros((4608,1)), np.zeros((4608,1)) 
# "Num_Visited_RPi_x= [np.zeros((4608,1)), np.zeros((4608,1))
# "actions_mps= np.array([np.round((np.array([60, 60, 80, 80, 100, 100, 120, 120]))*(1/3.6),2), np.array([1,3,1,3,1,3,1,3])]), 
# "actions_kmph= np.array([np.array([60, 60, 80, 80, 100, 100, 120, 120]), np.array([1,3,1,3,1,3,1,3])]), 
# "prevActionWinIndex= (np.array([60, 60, 80, 80, 100, 100, 120, 120]).size)-1,
# "alpha_LPi_ak= np.array([0.0, 0.0]), 
# "alpha_LPi_ai= np.array([0.0, 0.0]), 
# "alpha_W_LPi= np.array([0.0, 0.0]), 
# "alter_LPi_W= np.array([0.0, 0.0]),
# "alpha_RPi_ak= np.array([0.0, 0.0]), 
# "alpha_RPi_ai= np.array([0.0, 0.0]), 
# "alpha_W_RPi= np.array([0.0, 0.0]),
# "alter_RPi_W= np.array([0.0, 0.0]),
# "actionSub_y= [np.array([0]), np.array([0])
# "x_LPi= np.array([0, 0]), 
# "x_old_LPi= np.array([0, 0]), 
# "y_LPi= np.array([0, 0]), 
# "R= np.array([0.0, 0.0]), 
# "VSLposition= 0, 
# "VSLspeed= 0,
# "actionWant= 0,
# "actionWant_LPi= [0, 0
# "actionWant_RPi= [0, 0
# "actionControl_LPi= [0, 0
# "actionControl_RPi= [0, 0 
# "LPiWin= """", 
# "RPiWin= """", 
# "policyWin= """", 
# "actionWinIndex= 0,
# "speedSectionL0= np.array([0.0,0.0,0.0,0.0]), 
# "speedSectionL1= np.array([0.0,0.0,0.0,0.0]), 
# "speedSectionL2= np.array([0.0,0.0,0.0,0.0]), 
# "speedSectionL3= np.array([0.0,0.0,0.0,0.0]),
# "densitySectionL0= np.array([0.0,0.0,0.0,0.0]), 
# "densitySectionL1= np.array([0.0,0.0,0.0,0.0]), 
# "densitySectionL2= np.array([0.0,0.0,0.0,0.0]), 
# "densitySectionL3= np.array([0.0,0.0,0.0,0.0]),
# "TTS_sectionL0= 0,
# "TTS_sectionL1= 0, 
# "TTS_sectionL2= 0,
# "TTS_sectionL3= 0,
# "TTS_sectionL3_haltingVeh_ramp= 0, 
# "TTS_Net_Sim= 0,
# "VSL_beforeA1= np.round(np.array([120, 120])*(1/(3.6)),2), 
# "VSL_beforeA2= np.round(np.array([120, 120])*(1/(3.6)),2),
# "TTS_old_sectionL0= 0,
# "TTS_old_sectionL1= 0, 
# "TTS_old_sectionL2= 0, 
# "TTS_old_sectionL3= 0}

