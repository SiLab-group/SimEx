import random
import numpy as np
import itertools
from numpy import savetxt
import pandas as pd
from copy import deepcopy
from marl.common_attributes import CommonAttributes

# class IterAgents(type):
#     def __iter__(cls):
#         return iter(cls._allAgents)
    
# class Agent(metaclass=IterAgents):
class Agent(CommonAttributes):
    """
    A class to represent a RL-based VSL agent responsible for the
    management of traffic responsive dynamic speed limits.
    Shared attributes and their values are loaded from module agents_common_attributes.py !

    Attributes:
    name (str): Agent's name
    """
    # _allAgents=[]

    # __slots__ = ["name", "Q_LPi", "Q_RPi", "W_LPi", "W_RPi", "Num_Visited_LPi_x_ak",\
    #             "Num_Visited_RPi_x_ak", "Num_Visited_LPi_x_ai", "Num_Visited_RPi_x_ai",\
    #             "Num_Visited_LPi_x", "Num_Visited_RPi_x",\
    #             "actions_mps", "actions_kmph", "prevActionWinIndex",\
    #             "alpha_LPi_ak", "alpha_LPi_ai", "alpha_W_LPi", "alter_LPi_W",\
    #             "alpha_RPi_ak", "alpha_RPi_ai", "alpha_W_RPi", "alter_RPi_W",\
    #             "actionSub_y", "x_LPi", "x_old_LPi", "y_LPi", "R", "VSLposition", "VSLspeed",\
    #             "actionWant", "actionWant_LPi", "actionWant_RPi", "actionControl_LPi",\
    #             "actionControl_RPi", "LPiWin", "RPiWin", "policyWin", "actionWinIndex",\
    #             "speedSectionL0", "speedSectionL1", "speedSectionL2", "speedSectionL3",\
    #             "densitySectionL0", "densitySectionL1", "densitySectionL2", "densitySectionL3",\
    #             "TTS_sectionL0", "TTS_sectionL1", "TTS_sectionL2", "TTS_sectionL3", "TTS_sectionL3_haltingVeh_ramp", "TTS_Net_Sim",\
    #             "VSL_beforeA1", "VSL_beforeA2",\
    #             "TTS_old_sectionL0", "TTS_old_sectionL1", "TTS_old_sectionL2", "TTS_old_sectionL3"]

    __slots__ = ("name",)
    def __init__(self, name):
        super().__init__()
        # self._allAgents.append(self)
        
        # # Assign the specific attribute
        self.name = name
        '''Name of the agent'''
        
        # # Assign shared attributes manually from the dictionary
        # # for attr, value in itertools.islice(agents_common_attributes.items(), 1, None):
        # for attr, value in agents_common_attributes.items():
        #     setattr(self, attr, value) 
   
    def updateQ(self, x, ak, y, action_subset_y, r, alpha_Q_ak, gama, Q):
        """
        Q-Learning update rule - compute Q-values of an agent
            :param x: state of the system, in RL lingo s(t)
            :param ak: winner action
            :param y: new state of the system, s(t+1)
            :param action_subset_y: allowed actions regarding the previously executed action
            :param r: reward
            :param alpha_Q_ak: dynamic learning rate of Q's
            :param gama: discount factor
            :param Q: Q-matrix (Lookap table)
        """
        Q[x,ak] = (1 - alpha_Q_ak)*Q[x, ak] + alpha_Q_ak*(r + gama*(Q[y,action_subset_y].max()))
   
    def updateW(self, x, ai, y, action_subset_y, r, alpha_W, alpha_Q_ai, gama, w, W, Q):
        """
        W-Learning update rule - compute W-values of an agent
            :param x: state of the system, in RL lingo s(t)
            :param ai: action that agent_i wanted to implement but it didn't win
            :param y: new state of the system, s(t+1)
            :param action_subset_y: allowed actions regarding the previously executed action
            :param r: reward
            :param alpha_W: learning rate of W's
            :param alpha_Q_ai: dynamic learning rate of Q's
            :param w: controls convergence rate of confidence for recommended action
            :param Q: Q-matrix (Lookap table)
        """
        W[x] = (1 - alpha_W)*W[x] + alpha_W*((1-alpha_Q_ai)**w)*(Q[x,ai] - \
                (r + gama*(Q[y,action_subset_y].max())))
        
        
    def alpha_Q_ak_Function(self, x, ak, Num_Visited_x_ak, update):
        """
        Dynamic state-action-visits counter that computes learning parameter alpha_Q for action winner

        Parameters:
            x (int): state of the system, in RL lingo s(t)
            ak (int): winner action
            Num_Visited_x_ak (int): number of visits of state-action pairs (x, ak)
            update (0 or 1): prevents dividing by zero first time (x, ak) is visited

        Returns:
            int or float: The learning parameter alpha_Q 
        """
        if update == 1:
            Num_Visited_x_ak[x,ak]+=1
            alpha_Q_ak = 1/Num_Visited_x_ak[x,ak]
        else:
            if Num_Visited_x_ak[x,ak] == 0:
                alpha_Q_ak = 1
            else:
                alpha_Q_ak = 1/Num_Visited_x_ak[x,ak]
        return alpha_Q_ak
    
    def alpha_W_Function(self, x, Num_Visited_x, update):
        if update == 1:
            Num_Visited_x[x,0]+=1  
            alpha_W = 1/Num_Visited_x[x,0] 
        else:
            if Num_Visited_x[x,0] == 0:
                    alpha_W = 1
            else:
                alpha_W = 1/Num_Visited_x[x,0]
        return alpha_W
        
    def alpha_Q_ai_Function(self, x, ai, Num_Visited_x_ai):
        Num_Visited_x_ai[x,ai]+=1
        alpha_Q_ai = 1/Num_Visited_x_ai[x,ai]
        return alpha_Q_ai
        
    def suggestAction(self, epsilon, x, Q, previous_action_index, actions_A, run, numberSaveData):
        random_number = random.random()
        x_random = sum(random_number >= np.cumsum([0,1-epsilon,epsilon]))
            
        x1=np.array(abs(actions_A[:] - actions_A[previous_action_index])<=20)
        x2=np.array(abs(actions_A[:] - actions_A[previous_action_index])<=20)

        intersect_logic = np.logical_and(x1, x2)
        intersect = np.where(intersect_logic == True)
        action_subset = intersect[0]
        
        if x_random == 1: # exploit
            index_max_Q = np.argmax(Q[x,action_subset])
            action_index=action_subset[index_max_Q]
            action_control = 0
            
        else:  # explore
            action_index = np.random.choice(action_subset)
            action_control = 1
            
        return action_index, action_control
    
    def subsetAction_in_y(self, actionWinIndex, actions_A, run, numberSaveData):
        x1=np.array(abs(actions_A[:] - actions_A[actionWinIndex])<=20)
        x2=np.array(abs(actions_A[:] - actions_A[actionWinIndex])<=20)
        intersect_logic = np.logical_and(x1, x2)
        intersect = np.where(intersect_logic == True)
        action_subset_y = intersect[0]
        
        return action_subset_y



def actionWinner(agent, C, x_LP1, x_LP2, x_RP1, x_RP2, epsilon, actionIndexLP1, actionIndexLP2,\
                 actionIndexRP1, actionIndexRP2, W_LP1, W_LP2, W_RP1, W_RP2):
    
    action_index_LP = [actionIndexLP1, actionIndexLP2]
    action_index_RP = [actionIndexRP1, actionIndexRP2]
    Ws_LP = np.array([[W_LP1[x_LP1,0], W_LP2[x_LP2,0]]])

    index_max_W_LP = np.argmax(Ws_LP[0,:]) # to indicate policy "LPi"
    action_win_index_LP = action_index_LP[index_max_W_LP]
    agentAiLPi_win = "LP"+str(index_max_W_LP+1)
    AiLP1want = actionIndexLP1
    AiLP2want = actionIndexLP2

    Ws_RP = np.array([[W_RP1[x_RP1,0], W_RP2[x_RP2,0]]])
    index_max_W_RP = np.argmax(Ws_RP[0,:]) # to indicate policy "LPi"
    action_win_index_RP = action_index_RP[index_max_W_RP]
    agentAiRPj_win = "RP"+str(index_max_W_RP+1)
    AiRP1want = actionIndexRP1
    AiRP2want = actionIndexRP2
    
    W_win = np.array([[Ws_LP[0, index_max_W_LP], C*Ws_RP[0, index_max_W_RP]]])
    
    index_max_W_win = np.argmax(W_win[0,:]) # to indicate policy "LPi"

    if index_max_W_win == 0:
        policyWin = "LP"+str(index_max_W_LP+1)
        action_win_index = action_win_index_LP
    else:
        policyWin = "RP"+str(index_max_W_RP+1)
        action_win_index = action_win_index_RP
    
    return AiLP1want, AiLP2want, agentAiLPi_win, AiRP1want, AiRP2want, agentAiRPj_win,\
        action_win_index, policyWin
  
# Initialization of agents from Class "Agent"
A1 = Agent('A1') # Global instances
A2 = Agent('A2')

# State definition (for details see our papers mentioned in README file)
def states_combinationsAi():
    states_combinationsAi = np.empty([4608, 4], dtype=object)
    k = 0
    for i in itertools.product(['a0','a1','a2','a3','a4','a5','a6','a7'],\
                            ['A1v1','A1v2','A1v3','A1v4'],\
                            ['L3p1','L3p2','L3p3','L3p4','L3p5','L3p6','L3p7','L3p8','L3p9','L3p10','L3p11','L3p12'],\
                            ['L4p1','L4p2','L4p3','L4p4','L4p5','L4p6','L4p7','L4p8','L4p9','L4p10','L4p11','L4p12']):

        states_combinationsAi[k, 0]=i[0]
        states_combinationsAi[k, 1]=i[1]
        states_combinationsAi[k, 2]=i[2]
        states_combinationsAi[k, 3]=i[3]
        k+=1
    return states_combinationsAi
states_combinationsAi = states_combinationsAi()



def AgentStateIndex(a_before, v_in, p_in1, p_in2):
    v1=14
    v2=21
    v3=28

    p1=15
    p2=20
    p3=23
    p4=26
    p5=29
    p6=32
    p7=35
    p8=38
    p9=45
    p10=55
    p11=65

    if a_before==0:
        a_ = 'a0'
    elif a_before==1:
        a_ = 'a1'
    elif a_before==2:
        a_ = 'a2'
    elif a_before==3:
        a_ = 'a3'
    elif a_before==4:
        a_ = 'a4'
    elif a_before==5:
        a_ = 'a5'
    elif a_before==6:
        a_ = 'a6'
    else:
        a_ = 'a7'
        
  
    if v_in<=v1:
        A1v = 'A1v1'
    elif v1<v_in<=v2:
        A1v = 'A1v2'
    elif v2<v_in<=v3:
        A1v = 'A1v3'
    else:
        A1v = 'A1v4'
              
          
    if p_in1<p1:
        L3p = 'L3p1'
    elif p1<=p_in1<p2:
        L3p = 'L3p2'
    elif p2<=p_in1<p3:
        L3p = 'L3p3'
    elif p3<=p_in1<p4:
        L3p = 'L3p4'
    elif p4<=p_in1<p5:
        L3p = 'L3p5'
    elif p5<=p_in1<p6:
        L3p = 'L3p6'
    elif p6<=p_in1<p7:
        L3p = 'L3p7'
    elif p7<=p_in1<p8:
        L3p = 'L3p8'
    elif p8<=p_in1<p9:
        L3p = 'L3p9'  
    elif p9<=p_in1<p10:
        L3p = 'L3p10'  
    elif p10<=p_in1<p11:
        L3p = 'L3p11'  
    else:
        L3p = 'L3p12'
        
        
    if p_in2<p1:
        L4p = 'L4p1'
    elif p1<=p_in2<p2:
        L4p = 'L4p2'
    elif p2<=p_in2<p3:
        L4p = 'L4p3'
    elif p3<=p_in2<p4:
        L4p = 'L4p4'
    elif p4<=p_in2<p5:
        L4p = 'L4p5'
    elif p5<=p_in2<p6:
        L4p = 'L4p6'
    elif p6<=p_in2<p7:
        L4p = 'L4p7'
    elif p7<=p_in2<p8:
        L4p = 'L4p8'
    elif p8<=p_in2<p9:
        L4p = 'L4p9'
    elif p9<=p_in2<p10:
        L4p = 'L4p10'
    elif p10<=p_in2<p11:
        L4p = 'L4p11'
    else:
        L4p = 'L4p12'

    s1 = np.where(states_combinationsAi[:,0] == a_)
    s2 = np.where(states_combinationsAi[:,1] == A1v)
    s3 = np.where(states_combinationsAi[:,2] == L3p)
    s4 = np.where(states_combinationsAi[:,3] == L4p)

    s1_set = set(np.unique(s1))
    s2_set = set(np.unique(s2))
    s3_set = set(np.unique(s3))
    s4_set = set(np.unique(s4))
    row_intersect = set(s1_set).intersection(s2_set, s3_set, s4_set)
    list_row_intersect = list(row_intersect)
    row_state_index = list_row_intersect[0]

    return row_state_index
    

    
def saveData(vsl_on_off, run, numberSaveData, path_results, controlFile,\
             speedHeatMap, speedLimitHeatMap):
    """
    Save data:
    1) Learning parameters (can be loaded again by fn. LoadKnowlede() avoiding learning from scratch)
    2) System performance
    3) Agents behaviors

    Args:
    vsl_on_off (int): 0 or 1, 0==VSL off, 1==VSL on
    run (int): current simulation run
    numberSaveData (int): each #simulations saveData()
    path_results (string): path to store data
    controlFile (string): fiel where agents' behaviors are stored
    speedHeatMap (string): file where spatiotemporal motorway flow speeds are stored
    speedLimitHeatMap (string): file where spatiotemporal motorway speed limits are stored
    """
    if vsl_on_off==1 and (run%numberSaveData==0):

        savetxt(path_results+"speedHeatMap"+str(run)+".csv", speedHeatMap, delimiter=',')
        savetxt(path_results+"speedLimitHeatMap"+str(run)+".csv", speedLimitHeatMap, delimiter=',')        
        
        # save A1 policies LPs and RPs
        savetxt(path_results+"A1LP1"+str(run)+".csv", A1.Q_LPi[0], delimiter=',')
        savetxt(path_results+"A1LP2"+str(run)+".csv", A1.Q_LPi[1], delimiter=',')
        savetxt(path_results+"A1RP1"+str(run)+".csv", A1.Q_RPi[0], delimiter=',')
        savetxt(path_results+"A1RP2"+str(run)+".csv", A1.Q_RPi[1], delimiter=',')
        # save A2 policies LPs and RPs
        savetxt(path_results+"A2LP1"+str(run)+".csv", A2.Q_LPi[0], delimiter=',')
        savetxt(path_results+"A2LP2"+str(run)+".csv", A2.Q_LPi[1], delimiter=',')
        savetxt(path_results+"A2RP1"+str(run)+".csv", A2.Q_RPi[0], delimiter=',')
        savetxt(path_results+"A2RP2"+str(run)+".csv", A2.Q_RPi[1], delimiter=',')
        # save A1 Ws
        savetxt(path_results+"A1W_LP1"+str(run)+".csv", A1.W_LPi[0], delimiter=',')
        savetxt(path_results+"A1W_LP2"+str(run)+".csv", A1.W_LPi[1], delimiter=',')
        savetxt(path_results+"A1W_RP1"+str(run)+".csv", A1.W_RPi[0], delimiter=',')
        savetxt(path_results+"A1W_RP2"+str(run)+".csv", A1.W_RPi[1], delimiter=',')
        # save A2 Ws
        savetxt(path_results+"A2W_LP1"+str(run)+".csv", A2.W_LPi[0], delimiter=',')
        savetxt(path_results+"A2W_LP2"+str(run)+".csv", A2.W_LPi[1], delimiter=',')
        savetxt(path_results+"A2W_RP1"+str(run)+".csv", A2.W_RPi[0], delimiter=',')
        savetxt(path_results+"A2W_RP2"+str(run)+".csv", A2.W_RPi[1], delimiter=',')

        # save n(x,ak) for A1, A2
        savetxt(path_results+"A1NumLP1_X_ak"+str(run)+".csv", A1.Num_Visited_LPi_x_ak[0], delimiter=',')
        savetxt(path_results+"A1NumLP2_X_ak"+str(run)+".csv", A1.Num_Visited_LPi_x_ak[1], delimiter=',')
        savetxt(path_results+"A1NumRP1_X_ak"+str(run)+".csv", A1.Num_Visited_RPi_x_ak[0], delimiter=',')
        savetxt(path_results+"A1NumRP2_X_ak"+str(run)+".csv", A1.Num_Visited_RPi_x_ak[1], delimiter=',')
        
        savetxt(path_results+"A2NumLP1_X_ak"+str(run)+".csv", A2.Num_Visited_LPi_x_ak[0], delimiter=',')
        savetxt(path_results+"A2NumLP2_X_ak"+str(run)+".csv", A2.Num_Visited_LPi_x_ak[1], delimiter=',')
        savetxt(path_results+"A2NumRP1_X_ak"+str(run)+".csv", A2.Num_Visited_RPi_x_ak[0], delimiter=',')
        savetxt(path_results+"A2NumRP2_X_ak"+str(run)+".csv", A2.Num_Visited_RPi_x_ak[1], delimiter=',')
        
        # save n(x,ai) for A1, A2 this is not necessary
        savetxt(path_results+"A1NumLP1_X_ai"+str(run)+".csv", A1.Num_Visited_LPi_x_ai[0], delimiter=',')
        savetxt(path_results+"A1NumLP2_X_ai"+str(run)+".csv", A1.Num_Visited_LPi_x_ai[1], delimiter=',')
        savetxt(path_results+"A1NumRP1_X_ai"+str(run)+".csv", A1.Num_Visited_RPi_x_ai[0], delimiter=',')
        savetxt(path_results+"A1NumRP2_X_ai"+str(run)+".csv", A1.Num_Visited_RPi_x_ai[1], delimiter=',')
        
        savetxt(path_results+"A2NumLP1_X_ai"+str(run)+".csv", A2.Num_Visited_LPi_x_ai[0], delimiter=',')
        savetxt(path_results+"A2NumLP2_X_ai"+str(run)+".csv", A2.Num_Visited_LPi_x_ai[1], delimiter=',')
        savetxt(path_results+"A2NumRP1_X_ai"+str(run)+".csv", A2.Num_Visited_RPi_x_ai[0], delimiter=',')
        savetxt(path_results+"A2NumRP2_X_ai"+str(run)+".csv", A2.Num_Visited_RPi_x_ai[1], delimiter=',')
        
        # save n(x) for A1, A2
        savetxt(path_results+"A1NumLP1_X"+str(run)+".csv", A1.Num_Visited_LPi_x[0], delimiter=',')
        savetxt(path_results+"A1NumLP2_X"+str(run)+".csv", A1.Num_Visited_LPi_x[1], delimiter=',')
        savetxt(path_results+"A1NumRP1_X"+str(run)+".csv", A1.Num_Visited_RPi_x[0], delimiter=',')
        savetxt(path_results+"A1NumRP2_X"+str(run)+".csv", A1.Num_Visited_RPi_x[1], delimiter=',')
        
        savetxt(path_results+"A2NumLP1_X"+str(run)+".csv", A2.Num_Visited_LPi_x[0], delimiter=',')
        savetxt(path_results+"A2NumLP2_X"+str(run)+".csv", A2.Num_Visited_LPi_x[1], delimiter=',')
        savetxt(path_results+"A2NumRP1_X"+str(run)+".csv", A2.Num_Visited_RPi_x[0], delimiter=',')
        savetxt(path_results+"A2NumRP2_X"+str(run)+".csv", A2.Num_Visited_RPi_x[1], delimiter=',')
               
        # optional -> Results and additional analytics for control of the agent's learning process (you can reduce it unnecessary)
        df=pd.DataFrame({'A1X(t-1)': controlFile[1:,0], 'A2X(t-1)': controlFile[1:,1], 'V1(t-1)': controlFile[1:,2],\
                        'V2(t-1)': controlFile[1:,3],'V3(t-1)': controlFile[1:,4],'V4(t-1)': controlFile[1:,5],\
                        'denL1(t-1)': controlFile[1:,6],'denL2(t-1)': controlFile[1:,7], 'denL3(t-1)': controlFile[1:,8],\
                        'denL4(t-1)': controlFile[1:,9], 'VSL1(t-1)': controlFile[1:,10], 'VSL2(t-1)': controlFile[1:,11],\
                        'TTS1(t-1)': controlFile[1:,12], 'TTS2(t-1)': controlFile[1:,13], 'TTS3(t-1)': controlFile[1:,14],\
                        'TTS4(t-1)': controlFile[1:,15], 'TTSW(t-1)': controlFile[1:,16], 'NumVeh_W': controlFile[1:,17],\
                        'A1r1': controlFile[1:,18], 'A1r2': controlFile[1:,19], 'A2r1': controlFile[1:,20],\
                        'A2r2': controlFile[1:,21], 'A1X(t)': controlFile[1:,22], 'A2X(t)': controlFile[1:,23],\
                        'V1(t)': controlFile[1:,24],'V2(t)': controlFile[1:,25],'V3(t)': controlFile[1:,26],\
                        'V4(t)': controlFile[1:,27], 'denL1(t)': controlFile[1:,28],\
                         'denL2(t)': controlFile[1:,29], 'denL3(t)': controlFile[1:,30],\
                         'denL4(t)': controlFile[1:,31], 'VSL1(t)': controlFile[1:,32],\
                        'A1LP1want': controlFile[1:,33], 'A1LP2want': controlFile[1:,34],'A1RP1want': controlFile[1:,35],\
                        'A1RP2want': controlFile[1:,36], 'A1policyWin': controlFile[1:,37],'A1LP1cont': controlFile[1:,38],\
                        'A1LP2cont': controlFile[1:,39], 'A1RP1cont': controlFile[1:,40],'A1RP2cont': controlFile[1:,41],\
                        'VSL2(t)': controlFile[1:,42], 'A2LP1want': controlFile[1:,43],'A2LP2want': controlFile[1:,44],\
                         'A2RP1want': controlFile[1:,45], 'A2RP2want': controlFile[1:,46],'A2policyWin': controlFile[1:,47],\
                         'A2LP1cont': controlFile[1:,48], 'A2LP2cont': controlFile[1:,49], 'A2RP1cont': controlFile[1:,50],\
                         'A2RP2cont': controlFile[1:,51], 'A1W_LP1': controlFile[1:,52],'A1W_LP2': controlFile[1:,53],\
                        'A1W_RP1': controlFile[1:,54],'A1W_RP2': controlFile[1:,55],'A2W_LP1': controlFile[1:,56],\
                         'A2W_LP2': controlFile[1:,57], 'A2W_RP1': controlFile[1:,58],'A2W_RP2': controlFile[1:,59],\
                        'A1alph_LP1_ai': controlFile[1:,60],'A1alph_LP2_ai': controlFile[1:,61],\
                        'A1alph_RP1_ai': controlFile[1:,62],'A1alph_RP2_ai': controlFile[1:,63],\
                         'A2alph_LP1_ai': controlFile[1:,64],'A2alph_LP2_ai': controlFile[1:,65],\
                         'A2alph_RP1_ai': controlFile[1:,66],'A2alph_RP2_ai': controlFile[1:,67],\
                        'A1alph_LP1_ak': controlFile[1:,68],'A1alph_LP2_ak': controlFile[1:,69],\
                        'A1alph_RP1_ak': controlFile[1:,70],'A1alph_RP2_ak': controlFile[1:,71],\
                         'A2alph_LP1_ak': controlFile[1:,72],'A2alph_LP2_ak': controlFile[1:,73],\
                         'A2alph_RP1_ak': controlFile[1:,74],'A2alph_RP2_ak': controlFile[1:,75],\
                        'A1alph_WLP1': controlFile[1:,76],'A1alph_WLP2': controlFile[1:,77],\
                        'A1alph_WRP1': controlFile[1:,78],'A1alph_WRP2': controlFile[1:,79],\
                         'A2alph_WLP1': controlFile[1:,80],'A2alph_WLP2': controlFile[1:,81],\
                         'A2alph_WRP1': controlFile[1:,82],'A2alph_WRP2': controlFile[1:,83],\
                        'A1alt_WLP1': controlFile[1:,84],'A1alt_WLP2': controlFile[1:,85],\
                        'A1alt_WRP1': controlFile[1:,86],'A1alt_WRP2': controlFile[1:,87],\
                         'A2alt_WLP1': controlFile[1:,88],'A2alt_WLP2': controlFile[1:,89],\
                         'A2alt_WRP1': controlFile[1:,90],'A2alt_WRP2': controlFile[1:,91],\
                         'TTSsim': controlFile[1:,92], 'Occ1': controlFile[1:,93], 'Occ2': controlFile[1:,94],\
                        'Occ3': controlFile[1:,95], 'A1win': controlFile[1:,96], 'A1LPwin': controlFile[1:,97],\
                        'A1RPwin': controlFile[1:,98], 'A2win': controlFile[1:,99], 'A2LPwin': controlFile[1:,100],\
                        'A2RPwin': controlFile[1:,101], 'VSL1pos': controlFile[1:,102], 'VSL2pos': controlFile[1:,103]})
        
        df.to_excel(path_results+"controlFile"+str(run)+".xlsx", index = False)

        
    elif vsl_on_off==0:
        
        savetxt(path_results+"NO_VSL_speedHeatMap"+str(run)+".csv", speedHeatMap, delimiter=',')
        savetxt(path_results+"NO_VSL_speedLimitHeatMap"+str(run)+".csv", speedLimitHeatMap, delimiter=',')        
        
        # save A1 policies LPs and RPs
        savetxt(path_results+"NO_VSL_A1LP1"+str(run)+".csv", A1.Q_LPi[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A1LP2"+str(run)+".csv", A1.Q_LPi[1], delimiter=',')
        savetxt(path_results+"NO_VSL_A1RP1"+str(run)+".csv", A1.Q_RPi[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A1RP2"+str(run)+".csv", A1.Q_RPi[1], delimiter=',')
        # save A2 policies LPs and RPs
        savetxt(path_results+"NO_VSL_A2LP1"+str(run)+".csv", A2.Q_LPi[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A2LP2"+str(run)+".csv", A2.Q_LPi[1], delimiter=',')
        savetxt(path_results+"NO_VSL_A2RP1"+str(run)+".csv", A2.Q_RPi[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A2RP2"+str(run)+".csv", A2.Q_RPi[1], delimiter=',')
        # save A1 Ws
        savetxt(path_results+"NO_VSL_A1W_LP1"+str(run)+".csv", A1.W_LPi[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A1W_LP2"+str(run)+".csv", A1.W_LPi[1], delimiter=',')
        savetxt(path_results+"NO_VSL_A1W_RP1"+str(run)+".csv", A1.W_RPi[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A1W_RP2"+str(run)+".csv", A1.W_RPi[1], delimiter=',')
        # save A2 Ws
        savetxt(path_results+"NO_VSL_A2W_LP1"+str(run)+".csv", A2.W_LPi[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A2W_LP2"+str(run)+".csv", A2.W_LPi[1], delimiter=',')
        savetxt(path_results+"NO_VSL_A2W_RP1"+str(run)+".csv", A2.W_RPi[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A2W_RP2"+str(run)+".csv", A2.W_RPi[1], delimiter=',')

        # save n(x,ak) for A1, A2
        savetxt(path_results+"NO_VSL_A1NumLP1_X_ak"+str(run)+".csv", A1.Num_Visited_LPi_x_ak[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A1NumLP2_X_ak"+str(run)+".csv", A1.Num_Visited_LPi_x_ak[1], delimiter=',')
        savetxt(path_results+"NO_VSL_A1NumRP1_X_ak"+str(run)+".csv", A1.Num_Visited_RPi_x_ak[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A1NumRP2_X_ak"+str(run)+".csv", A1.Num_Visited_RPi_x_ak[1], delimiter=',')
        
        savetxt(path_results+"NO_VSL_A2NumLP1_X_ak"+str(run)+".csv", A2.Num_Visited_LPi_x_ak[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A2NumLP2_X_ak"+str(run)+".csv", A2.Num_Visited_LPi_x_ak[1], delimiter=',')
        savetxt(path_results+"NO_VSL_A2NumRP1_X_ak"+str(run)+".csv", A2.Num_Visited_RPi_x_ak[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A2NumRP2_X_ak"+str(run)+".csv", A2.Num_Visited_RPi_x_ak[1], delimiter=',')
        
        # save n(x,ai) for A1, A2 this is not necessary
        savetxt(path_results+"NO_VSL_A1NumLP1_X_ai"+str(run)+".csv", A1.Num_Visited_LPi_x_ai[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A1NumLP2_X_ai"+str(run)+".csv", A1.Num_Visited_LPi_x_ai[1], delimiter=',')
        savetxt(path_results+"NO_VSL_A1NumRP1_X_ai"+str(run)+".csv", A1.Num_Visited_RPi_x_ai[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A1NumRP2_X_ai"+str(run)+".csv", A1.Num_Visited_RPi_x_ai[1], delimiter=',')
        
        savetxt(path_results+"NO_VSL_A2NumLP1_X_ai"+str(run)+".csv", A2.Num_Visited_LPi_x_ai[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A2NumLP2_X_ai"+str(run)+".csv", A2.Num_Visited_LPi_x_ai[1], delimiter=',')
        savetxt(path_results+"NO_VSL_A2NumRP1_X_ai"+str(run)+".csv", A2.Num_Visited_RPi_x_ai[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A2NumRP2_X_ai"+str(run)+".csv", A2.Num_Visited_RPi_x_ai[1], delimiter=',')
        
        # save n(x) for A1, A2
        savetxt(path_results+"NO_VSL_A1NumLP1_X"+str(run)+".csv", A1.Num_Visited_LPi_x[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A1NumLP2_X"+str(run)+".csv", A1.Num_Visited_LPi_x[1], delimiter=',')
        savetxt(path_results+"NO_VSL_A1NumRP1_X"+str(run)+".csv", A1.Num_Visited_RPi_x[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A1NumRP2_X"+str(run)+".csv", A1.Num_Visited_RPi_x[1], delimiter=',')
        
        savetxt(path_results+"NO_VSL_A2NumLP1_X"+str(run)+".csv", A2.Num_Visited_LPi_x[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A2NumLP2_X"+str(run)+".csv", A2.Num_Visited_LPi_x[1], delimiter=',')
        savetxt(path_results+"NO_VSL_A2NumRP1_X"+str(run)+".csv", A2.Num_Visited_RPi_x[0], delimiter=',')
        savetxt(path_results+"NO_VSL_A2NumRP2_X"+str(run)+".csv", A2.Num_Visited_RPi_x[1], delimiter=',')
        
        df=pd.DataFrame({'A1X(t-1)': controlFile[1:,0], 'A2X(t-1)': controlFile[1:,1], 'V1(t-1)': controlFile[1:,2],\
                        'V2(t-1)': controlFile[1:,3],'V3(t-1)': controlFile[1:,4],'V4(t-1)': controlFile[1:,5],\
                        'denL1(t-1)': controlFile[1:,6],'denL2(t-1)': controlFile[1:,7], 'denL3(t-1)': controlFile[1:,8],\
                        'denL4(t-1)': controlFile[1:,9], 'VSL1(t-1)': controlFile[1:,10], 'VSL2(t-1)': controlFile[1:,11],\
                        'TTS1(t-1)': controlFile[1:,12], 'TTS2(t-1)': controlFile[1:,13], 'TTS3(t-1)': controlFile[1:,14],\
                        'TTS4(t-1)': controlFile[1:,15], 'TTSW(t-1)': controlFile[1:,16], 'NumVeh_W': controlFile[1:,17],\
                        'A1r1': controlFile[1:,18], 'A1r2': controlFile[1:,19], 'A2r1': controlFile[1:,20],\
                        'A2r2': controlFile[1:,21], 'A1X(t)': controlFile[1:,22], 'A2X(t)': controlFile[1:,23],\
                        'V1(t)': controlFile[1:,24],'V2(t)': controlFile[1:,25],'V3(t)': controlFile[1:,26],\
                        'V4(t)': controlFile[1:,27], 'denL1(t)': controlFile[1:,28],\
                         'denL2(t)': controlFile[1:,29], 'denL3(t)': controlFile[1:,30],\
                         'denL4(t)': controlFile[1:,31], 'VSL1(t)': controlFile[1:,32],\
                        'A1LP1want': controlFile[1:,33], 'A1LP2want': controlFile[1:,34],'A1RP1want': controlFile[1:,35],\
                        'A1RP2want': controlFile[1:,36], 'A1policyWin': controlFile[1:,37],'A1LP1cont': controlFile[1:,38],\
                        'A1LP2cont': controlFile[1:,39], 'A1RP1cont': controlFile[1:,40],'A1RP2cont': controlFile[1:,41],\
                        'VSL2(t)': controlFile[1:,42], 'A2LP1want': controlFile[1:,43],'A2LP2want': controlFile[1:,44],\
                         'A2RP1want': controlFile[1:,45], 'A2RP2want': controlFile[1:,46],'A2policyWin': controlFile[1:,47],\
                         'A2LP1cont': controlFile[1:,48], 'A2LP2cont': controlFile[1:,49], 'A2RP1cont': controlFile[1:,50],\
                         'A2RP2cont': controlFile[1:,51], 'A1W_LP1': controlFile[1:,52],'A1W_LP2': controlFile[1:,53],\
                        'A1W_RP1': controlFile[1:,54],'A1W_RP2': controlFile[1:,55],'A2W_LP1': controlFile[1:,56],\
                         'A2W_LP2': controlFile[1:,57], 'A2W_RP1': controlFile[1:,58],'A2W_RP2': controlFile[1:,59],\
                        'A1alph_LP1_ai': controlFile[1:,60],'A1alph_LP2_ai': controlFile[1:,61],\
                        'A1alph_RP1_ai': controlFile[1:,62],'A1alph_RP2_ai': controlFile[1:,63],\
                         'A2alph_LP1_ai': controlFile[1:,64],'A2alph_LP2_ai': controlFile[1:,65],\
                         'A2alph_RP1_ai': controlFile[1:,66],'A2alph_RP2_ai': controlFile[1:,67],\
                        'A1alph_LP1_ak': controlFile[1:,68],'A1alph_LP2_ak': controlFile[1:,69],\
                        'A1alph_RP1_ak': controlFile[1:,70],'A1alph_RP2_ak': controlFile[1:,71],\
                         'A2alph_LP1_ak': controlFile[1:,72],'A2alph_LP2_ak': controlFile[1:,73],\
                         'A2alph_RP1_ak': controlFile[1:,74],'A2alph_RP2_ak': controlFile[1:,75],\
                        'A1alph_WLP1': controlFile[1:,76],'A1alph_WLP2': controlFile[1:,77],\
                        'A1alph_WRP1': controlFile[1:,78],'A1alph_WRP2': controlFile[1:,79],\
                         'A2alph_WLP1': controlFile[1:,80],'A2alph_WLP2': controlFile[1:,81],\
                         'A2alph_WRP1': controlFile[1:,82],'A2alph_WRP2': controlFile[1:,83],\
                        'A1alt_WLP1': controlFile[1:,84],'A1alt_WLP2': controlFile[1:,85],\
                        'A1alt_WRP1': controlFile[1:,86],'A1alt_WRP2': controlFile[1:,87],\
                         'A2alt_WLP1': controlFile[1:,88],'A2alt_WLP2': controlFile[1:,89],\
                         'A2alt_WRP1': controlFile[1:,90],'A2alt_WRP2': controlFile[1:,91],\
                        'TTSsim': controlFile[1:,92],'Occ1': controlFile[1:,93], 'Occ2': controlFile[1:,94],\
                        'Occ3': controlFile[1:,95], 'A1win': controlFile[1:,96], 'A1LPwin': controlFile[1:,97],\
                        'A1RPwin': controlFile[1:,98], 'A2win': controlFile[1:,99], 'A2LPwin': controlFile[1:,100],\
                        'A2RPwin': controlFile[1:,101], 'VSL1pos': controlFile[1:,102], 'VSL2pos': controlFile[1:,103]})
        
        df.to_excel(path_results+"NO_VSL_controlFile"+str(run)+".xlsx", index = False)



def LoadKnowlede(path_results, sim_run):    # sim_run old knowledge in case!
        A1.Q_LPi[0] = np.loadtxt(open(path_results+'A1LP1'+str(sim_run)+'.csv', 'rt'), delimiter=",")
        A2.Q_LPi[0] = np.loadtxt(open(path_results+'A2LP1'+str(sim_run)+'.csv', 'rt'), delimiter=",")
        A1.Q_RPi[0] = np.loadtxt(open(path_results+'A1RP1'+str(sim_run)+'.csv', 'rt'), delimiter=",")
        A2.Q_RPi[0] = np.loadtxt(open(path_results+'A2RP1'+str(sim_run)+'.csv', 'rt'), delimiter=",")
        A1.Q_LPi[1] = np.loadtxt(open(path_results+'A1LP2'+str(sim_run)+'.csv', 'rt'), delimiter=",")
        A2.Q_LPi[1] = np.loadtxt(open(path_results+'A2LP2'+str(sim_run)+'.csv', 'rt'), delimiter=",")
        A1.Q_RPi[1] = np.loadtxt(open(path_results+'A1RP2'+str(sim_run)+'.csv', 'rt'), delimiter=",")
        A2.Q_RPi[1] = np.loadtxt(open(path_results+'A2RP2'+str(sim_run)+'.csv', 'rt'), delimiter=",")
        A1.W_LPi[0] = np.resize(np.loadtxt(open(path_results+'A1W_LP1'+str(sim_run)+'.csv', 'rt'), delimiter=","), 4608).reshape(4608,1)
        A2.W_LPi[0] = np.resize(np.loadtxt(open(path_results+'A2W_LP1'+str(sim_run)+'.csv', 'rt'), delimiter=","), 4608).reshape(4608,1)
        A1.W_RPi[0] = np.resize(np.loadtxt(open(path_results+'A1W_RP1'+str(sim_run)+'.csv', 'rt'), delimiter=","), 4608).reshape(4608,1)
        A2.W_RPi[0] = np.resize(np.loadtxt(open(path_results+'A2W_RP1'+str(sim_run)+'.csv', 'rt'), delimiter=","), 4608).reshape(4608,1)
        A1.W_LPi[1] = np.resize(np.loadtxt(open(path_results+'A1W_LP2'+str(sim_run)+'.csv', 'rt'), delimiter=","), 4608).reshape(4608,1)
        A2.W_LPi[1] = np.resize(np.loadtxt(open(path_results+'A2W_LP2'+str(sim_run)+'.csv', 'rt'), delimiter=","), 4608).reshape(4608,1)
        A1.W_RPi[1] = np.resize(np.loadtxt(open(path_results+'A1W_RP2'+str(sim_run)+'.csv', 'rt'), delimiter=","), 4608).reshape(4608,1)
        A2.W_RPi[1] = np.resize(np.loadtxt(open(path_results+'A2W_RP2'+str(sim_run)+'.csv', 'rt'), delimiter=","), 4608).reshape(4608,1)
             
        A1.Num_Visited_LPi_x_ak[0] = np.loadtxt(open(path_results+'A1NumLP1_X_ak'+str(sim_run)+'.csv', 'rt'), delimiter=',')
        A2.Num_Visited_LPi_x_ak[0] = np.loadtxt(open(path_results+'A2NumLP1_X_ak'+str(sim_run)+'.csv', 'rt'), delimiter=',')
        A1.Num_Visited_RPi_x_ak[0] = np.loadtxt(open(path_results+'A1NumRP1_X_ak'+str(sim_run)+'.csv', 'rt'), delimiter=',')
        A2.Num_Visited_RPi_x_ak[0] = np.loadtxt(open(path_results+'A2NumRP1_X_ak'+str(sim_run)+'.csv', 'rt'), delimiter=',')
        A1.Num_Visited_LPi_x_ak[1] = np.loadtxt(open(path_results+'A1NumLP2_X_ak'+str(sim_run)+'.csv', 'rt'), delimiter=',')
        A2.Num_Visited_LPi_x_ak[1] = np.loadtxt(open(path_results+'A2NumLP2_X_ak'+str(sim_run)+'.csv', 'rt'), delimiter=',')
        A1.Num_Visited_RPi_x_ak[1] = np.loadtxt(open(path_results+'A1NumRP2_X_ak'+str(sim_run)+'.csv', 'rt'), delimiter=',')
        A2.Num_Visited_RPi_x_ak[1] = np.loadtxt(open(path_results+'A2NumRP2_X_ak'+str(sim_run)+'.csv', 'rt'), delimiter=',')
       
        A1.Num_Visited_LPi_x[0] = np.resize(np.loadtxt(open(path_results+'A1NumLP1_X'+str(sim_run)+'.csv', 'rt'), delimiter=','), 4608).reshape(4608,1)
        A2.Num_Visited_LPi_x[0] = np.resize(np.loadtxt(open(path_results+'A2NumLP1_X'+str(sim_run)+'.csv', 'rt'), delimiter=','), 4608).reshape(4608,1)
        A1.Num_Visited_RPi_x[0] = np.resize(np.loadtxt(open(path_results+'A1NumRP1_X'+str(sim_run)+'.csv', 'rt'), delimiter=','), 4608).reshape(4608,1)
        A2.Num_Visited_RPi_x[0] = np.resize(np.loadtxt(open(path_results+'A2NumRP1_X'+str(sim_run)+'.csv', 'rt'), delimiter=','), 4608).reshape(4608,1)
        A1.Num_Visited_LPi_x[1] = np.resize(np.loadtxt(open(path_results+'A1NumLP2_X'+str(sim_run)+'.csv', 'rt'), delimiter=','), 4608).reshape(4608,1)
        A2.Num_Visited_LPi_x[1] = np.resize(np.loadtxt(open(path_results+'A2NumLP2_X'+str(sim_run)+'.csv', 'rt'), delimiter=','), 4608).reshape(4608,1)
        A1.Num_Visited_RPi_x[1] = np.resize(np.loadtxt(open(path_results+'A1NumRP2_X'+str(sim_run)+'.csv', 'rt'), delimiter=','), 4608).reshape(4608,1)
        A2.Num_Visited_RPi_x[1] = np.resize(np.loadtxt(open(path_results+'A2NumRP2_X'+str(sim_run)+'.csv', 'rt'), delimiter=','), 4608).reshape(4608,1)
       
        A1.Num_Visited_LPi_x_ai[0] = np.loadtxt(open(path_results+'A1NumLP1_X_ai'+str(sim_run)+'.csv', 'rt'), delimiter=',')
        A2.Num_Visited_LPi_x_ai[0] = np.loadtxt(open(path_results+'A2NumLP1_X_ai'+str(sim_run)+'.csv', 'rt'), delimiter=',')
        A1.Num_Visited_RPi_x_ai[0] = np.loadtxt(open(path_results+'A1NumRP1_X_ai'+str(sim_run)+'.csv', 'rt'), delimiter=',')
        A2.Num_Visited_RPi_x_ai[0] = np.loadtxt(open(path_results+'A2NumRP1_X_ai'+str(sim_run)+'.csv', 'rt'), delimiter=',')
        A1.Num_Visited_LPi_x_ai[1] = np.loadtxt(open(path_results+'A1NumLP2_X_ai'+str(sim_run)+'.csv', 'rt'), delimiter=',')
        A2.Num_Visited_LPi_x_ai[1] = np.loadtxt(open(path_results+'A2NumLP2_X_ai'+str(sim_run)+'.csv', 'rt'), delimiter=',')
        A1.Num_Visited_RPi_x_ai[1] = np.loadtxt(open(path_results+'A1NumRP2_X_ai'+str(sim_run)+'.csv', 'rt'), delimiter=',')
        A2.Num_Visited_RPi_x_ai[1] = np.loadtxt(open(path_results+'A2NumRP2_X_ai'+str(sim_run)+'.csv', 'rt'), delimiter=',')

def AgentsInitSimParam(vsl_on_off):
    """
    Initialisation of some simulation parameters important for agents:
    1) Action set -> speed limit granularity and range
    2) VSL zone index that defines VSL zone configuration
    3) Initial speed limit value and VSL zone configuration at the begining of simulation
    """
    if vsl_on_off==1:
        actions_mps = np.round((np.array([60, 60, 80, 80, 100, 100, 120, 120]))*(1/3.6),2)
        position = np.array([1,3,1,3,1,3,1,3])
        A1.actions_mps = np.array([actions_mps, position])
        A2.actions_mps = np.array([actions_mps, position])
        actions_kmph = np.array([60, 60, 80, 80, 100, 100, 120, 120])
        position = np.array([1,3,1,3,1,3,1,3])
        A1.actions_kmph = np.array([actions_kmph, position])
        A2.actions_kmph = np.array([actions_kmph, position])
    else:
        actions_mps = np.round((np.array([120, 120, 120, 120, 120, 120, 120, 120]))*(1/3.6),2)
        position = np.array([1,3,1,3,1,3,1,3])
        A1.actions_mps = np.array([actions_mps, position])
        A2.actions_mps = np.array([actions_mps, position])
        actions_kmph = np.array([120, 120, 120, 120, 120, 120, 120, 120])
        position = np.array([1,3,1,3,1,3,1,3])
        A1.actions_kmph = np.array([actions_kmph, position])
        A2.actions_kmph = np.array([actions_kmph, position])

            #action_index_min = 0
        action_index_max_A1 = (A1.actions_mps[0,:].size)-1
        action_index_max_A2 = (A2.actions_mps[0,:].size)-1

        A1.prevActionWinIndex = action_index_max_A1
        A2.prevActionWinIndex = action_index_max_A2



def AgentState_x(j):


    A1.x_LPi[0] = AgentStateIndex(A1.prevActionWinIndex,\
                                        A1.speedSectionL2[j], A1.densitySectionL1[j], A1.densitySectionL2[j])

    A1.x_LPi[1] = AgentStateIndex(A1.prevActionWinIndex,\
                                        A1.speedSectionL1[j], A1.densitySectionL1[j], A1.densitySectionL2[j])

    A2.x_LPi[0] = AgentStateIndex(A2.prevActionWinIndex,\
                                    A2.speedSectionL3[j], A2.densitySectionL2[j], A2.densitySectionL3[j])

    A2.x_LPi[1] = AgentStateIndex(A2.prevActionWinIndex,\
                                    A2.speedSectionL2[j], A2.densitySectionL2[j], A2.densitySectionL3[j])

    
def AgentState_y(j):
    A1.y_LPi[0] = AgentStateIndex(A1.prevActionWinIndex,\
                                        A1.speedSectionL2[j], A1.densitySectionL1[j], A1.densitySectionL2[j])

    A1.y_LPi[1] = AgentStateIndex(A1.prevActionWinIndex,\
                                        A1.speedSectionL1[j], A1.densitySectionL1[j], A1.densitySectionL2[j])

    A2.y_LPi[0] = AgentStateIndex(A2.prevActionWinIndex,\
                                    A2.speedSectionL3[j], A2.densitySectionL2[j], A2.densitySectionL3[j])

    A2.y_LPi[1] = AgentStateIndex(A2.prevActionWinIndex,\
                                    A2.speedSectionL2[j], A2.densitySectionL2[j], A2.densitySectionL3[j])

# #=============== A1 Define action                                        
def Agent1Action(epsilon, run, numberSaveData, C):
    """
    Compute action for given state taking into account feasible action space due to previously executed action!

    Args:
    epsilon (float): epsilon-greedy approach for random actions (it controls exploration during learnig phase)
    run (int): current simulation run
    numberSaveData (int): each #numberSaveData simulations saveData() 
    C (float): cooperation coeficinet
    """

    actionIndexA1_LP1, actionControlA1_LP1 = A1.suggestAction(epsilon, A1.x_LPi[0], A1.Q_LPi[0],\
                                    A1.prevActionWinIndex, A1.actions_kmph[0,:], run, numberSaveData)
    actionIndexA1_LP2, actionControlA1_LP2 = A1.suggestAction(epsilon, A1.x_LPi[1], A1.Q_LPi[1],\
                                    A1.prevActionWinIndex, A1.actions_kmph[0,:], run, numberSaveData)
    actionIndexA1_RP1, actionControlA1_RP1 = A1.suggestAction(epsilon, A2.x_LPi[0], A1.Q_RPi[0],\
                                    A1.prevActionWinIndex, A1.actions_kmph[0,:], run, numberSaveData)
    actionIndexA1_RP2, actionControlA1_RP2 = A1.suggestAction(epsilon, A2.x_LPi[1], A1.Q_RPi[1],\
                                    A1.prevActionWinIndex, A1.actions_kmph[0,:], run, numberSaveData)


    A1LP1want, A1LP2want, policyA1LP_win, A1RP1want, A1RP2want,\
    remotePolicyA1RP_win, actionWinIndexA1, policyWinA1 = actionWinner(A1.name, C,\
                                                                    A1.x_LPi[0], A1.x_LPi[1], A2.x_LPi[0],\
                                                                    A2.x_LPi[1], epsilon, actionIndexA1_LP1,\
                                                                    actionIndexA1_LP2, actionIndexA1_RP1,\
                                                                    actionIndexA1_RP2, A1.W_LPi[0],\
                                                                    A1.W_LPi[1], A1.W_RPi[0], A1.W_RPi[1]) 


    A1.actionWinIndex = actionWinIndexA1
    A1.VSLspeed = A1.actions_mps[0, actionWinIndexA1]
    A1.VSLposition = A1.actions_mps[1, actionWinIndexA1]
#                 A1.prevActionWinIndex = A1.actionWinIndex
    A1.actionWant_LPi = [actionIndexA1_LP1, actionIndexA1_LP2]
    A1.actionWant_RPi = [actionIndexA1_RP1, actionIndexA1_RP2]
    A1.actionControl_LPi = [actionControlA1_LP1, actionControlA1_LP2]
    A1.actionControl_RPi = [actionControlA1_RP1, actionControlA1_RP2]
    A1.LPiWin = policyA1LP_win
    A1.RPiWin = remotePolicyA1RP_win
    A1.policyWin = policyWinA1

# #================ A2 Define action
def Agent2Action(epsilon, run, numberSaveData, C):
    """
    Compute action for given state taking into account feasible action space due to previously executed action!

    Args:
    epsilon (float): epsilon-greedy approach for random actions (it controls exploration)
    run (int): current simulation run
    numberSaveData (int): each #numberSaveData simulations saveData() 
    C (float): cooperation coeficinet
    """
    actionIndexA2_LP1, actionControlA2_LP1 = A2.suggestAction(epsilon, A2.x_LPi[0], A2.Q_LPi[0],\
                                    A2.prevActionWinIndex, A2.actions_kmph[0,:], run, numberSaveData)
    actionIndexA2_LP2, actionControlA2_LP2 = A2.suggestAction(epsilon, A2.x_LPi[1], A2.Q_LPi[1],\
                                    A2.prevActionWinIndex, A2.actions_kmph[0,:], run, numberSaveData)

    actionIndexA2_RP1, actionControlA2_RP1 = A2.suggestAction(epsilon, A1.x_LPi[0], A2.Q_RPi[0],\
                                    A2.prevActionWinIndex, A2.actions_kmph[0,:], run, numberSaveData)
    actionIndexA2_RP2, actionControlA2_RP2 = A2.suggestAction(epsilon, A1.x_LPi[1], A2.Q_RPi[1],\
                                    A2.prevActionWinIndex, A2.actions_kmph[0,:], run, numberSaveData)
    
    A2LP1want, A2LP2want, policyA2LP_win, A2RP1want, A2RP2want,\
    remotePolicyA2RP_win, actionWinIndexA2, policyWinA2 = actionWinner(A2.name, C, A2.x_LPi[0], A2.x_LPi[1],\
                                                                    A1.x_LPi[0], A1.x_LPi[1], epsilon,\
                                                                    actionIndexA2_LP1, actionIndexA2_LP2,\
                                                                    actionIndexA2_RP1, actionIndexA2_RP2,\
                                                                    A2.W_LPi[0], A2.W_LPi[1], A2.W_RPi[0],\
                                                                    A2.W_RPi[1])
    

    A2.actionWinIndex = actionWinIndexA2
    A2.VSLspeed = A2.actions_mps[0, actionWinIndexA2]
    A2.VSLposition = A2.actions_mps[1, actionWinIndexA2]
#                 A2.prevActionWinIndex = A2.actionWinIndex
    A2.actionWant_LPi = [actionIndexA2_LP1, actionIndexA2_LP2]
    A2.actionWant_RPi = [actionIndexA2_RP1, actionIndexA2_RP2]
    A2.actionControl_LPi = [actionControlA2_LP1, actionControlA2_LP2]
    A2.actionControl_RPi = [actionControlA2_RP1, actionControlA2_RP2]
    A2.LPiWin = policyA2LP_win
    A2.RPiWin = remotePolicyA2RP_win
    A2.policyWin = policyWinA2
                


def UpdateRewards(j, speed_free_flow, alpha, beta):
     
    #=================================================================
        # A1 policies rewards               
    if A1.speedSectionL1[j]>speed_free_flow and A1.speedSectionL2[j]>speed_free_flow:
        A1.R[0] = 0
#                     r2 = 0
    else:               
        A1.R[0]=-A1.TTS_sectionL1 #TTS2
    
    A1.R[1]=-alpha*A1.TTS_sectionL2 #TTS3
    #==================================================================
    # A2 policies rewards
    if A2.speedSectionL2[j]>speed_free_flow and A2.speedSectionL3[j]>speed_free_flow:
        A2.R[0] = 0
#                     r2 = 0
    else:               
        A2.R[0]=-A2.TTS_sectionL2
    
    A2.R[1]=-beta*A2.TTS_sectionL3 #TTS4
    #==================================================================         

def UpdateLocalpolicies(run, numberSaveData, gama, w):

#=================Update Ai
    for idAi, Ai in enumerate(Agent):
#=================Update Ai's LPi
        for indx, Q_LP in enumerate(Ai.Q_LPi):
            action_subset_y_Ai_LPi = Ai.subsetAction_in_y(Ai.actionWinIndex, Ai.actions_kmph[0,:],\
                                                            run, numberSaveData)
            Ai.actionSub_y[indx]=action_subset_y_Ai_LPi

            alphaQ_AiLPi_ak = Ai.alpha_Q_ak_Function(Ai.x_LPi[indx], Ai.actionWinIndex,\
                                                        Ai.Num_Visited_LPi_x_ak[indx], 1)                  

            Ai.updateQ(Ai.x_LPi[indx], Ai.actionWinIndex, Ai.y_LPi[indx], action_subset_y_Ai_LPi,\
                        Ai.R[indx], alphaQ_AiLPi_ak, gama, Ai.Q_LPi[indx])


        if Ai.policyWin=="RP1" or Ai.policyWin=="RP2":
            alphaQ_Ai_LP1_ai = Ai.alpha_Q_ak_Function(Ai.x_LPi[0], Ai.actionWant_LPi[0],\
                                    Ai.Num_Visited_LPi_x_ak[0], 0)
            alphaAi_W_LP1 = Ai.alpha_W_Function(Ai.x_LPi[0], Ai.Num_Visited_LPi_x[0], 1)
            Ai.updateW(Ai.x_LPi[0], Ai.actionWant_LPi[0], Ai.y_LPi[0], Ai.actionSub_y[0],\
                        Ai.R[0], alphaAi_W_LP1, alphaQ_Ai_LP1_ai, gama, w, Ai.W_LPi[0], Ai.Q_LPi[0])
            alterAi_LP1_W = alphaAi_W_LP1*(1-alphaQ_Ai_LP1_ai)**w

            alphaQ_Ai_LP2_ai = Ai.alpha_Q_ak_Function(Ai.x_LPi[1], Ai.actionWant_LPi[1],\
                                    Ai.Num_Visited_LPi_x_ak[1], 0)
            alphaAi_W_LP2 = Ai.alpha_W_Function(Ai.x_LPi[1], Ai.Num_Visited_LPi_x[1], 1)
            Ai.updateW(Ai.x_LPi[1], Ai.actionWant_LPi[1], Ai.y_LPi[1], Ai.actionSub_y[1],\
                        Ai.R[1], alphaAi_W_LP2, alphaQ_Ai_LP2_ai, gama, w, Ai.W_LPi[1], Ai.Q_LPi[1])
            alterAi_LP2_W = alphaAi_W_LP2*(1-alphaQ_Ai_LP2_ai)**w        
        else:
            if Ai.LPiWin == "LP2":
                alphaQ_Ai_LP1_ai = Ai.alpha_Q_ak_Function(Ai.x_LPi[0], Ai.actionWant_LPi[0],\
                                        Ai.Num_Visited_LPi_x_ak[0], 0)
                alphaAi_W_LP1 = Ai.alpha_W_Function(Ai.x_LPi[0], Ai.Num_Visited_LPi_x[0], 1)
                Ai.updateW(Ai.x_LPi[0], Ai.actionWant_LPi[0], Ai.y_LPi[0], Ai.actionSub_y[0],\
                            Ai.R[0], alphaAi_W_LP1, alphaQ_Ai_LP1_ai, gama, w, Ai.W_LPi[0], Ai.Q_LPi[0])
                alterAi_LP1_W = alphaAi_W_LP1*(1-alphaQ_Ai_LP1_ai)**w
            else:
                alphaQ_Ai_LP2_ai = Ai.alpha_Q_ak_Function(Ai.x_LPi[1], Ai.actionWant_LPi[1],\
                                        Ai.Num_Visited_LPi_x_ak[1], 0)
                alphaAi_W_LP2 = Ai.alpha_W_Function(Ai.x_LPi[1], Ai.Num_Visited_LPi_x[1], 1)
                Ai.updateW(Ai.x_LPi[1], Ai.actionWant_LPi[1], Ai.y_LPi[1], Ai.actionSub_y[1],\
                            Ai.R[1], alphaAi_W_LP2, alphaQ_Ai_LP2_ai, gama, w, Ai.W_LPi[1], Ai.Q_LPi[1])
                alterAi_LP2_W = alphaAi_W_LP2*(1-alphaQ_Ai_LP2_ai)**w
            
        # used for analytics, saved in controlFile
        Ai.alpha_LPi_ak[0] = Ai.alpha_Q_ak_Function(Ai.x_LPi[0], Ai.actionWinIndex,\
                                                        Ai.Num_Visited_LPi_x_ak[0], 0)
        Ai.alpha_LPi_ak[1] = Ai.alpha_Q_ak_Function(Ai.x_LPi[1], Ai.actionWinIndex,\
                                Ai.Num_Visited_LPi_x_ak[1], 0)
        
        Ai.alpha_LPi_ai[0] = Ai.alpha_Q_ak_Function(Ai.x_LPi[0], Ai.actionWant_LPi[0],\
                                    Ai.Num_Visited_LPi_x_ak[0], 0)
        Ai.alpha_LPi_ai[1] = Ai.alpha_Q_ak_Function(Ai.x_LPi[1], Ai.actionWant_LPi[1],\
                                    Ai.Num_Visited_LPi_x_ak[1], 0)

        Ai.alpha_W_LPi[0] = Ai.alpha_W_Function(Ai.x_LPi[0], Ai.Num_Visited_LPi_x[0], 0)
        Ai.alpha_W_LPi[1] = Ai.alpha_W_Function(Ai.x_LPi[1], Ai.Num_Visited_LPi_x[1], 0)

        Ai.alter_LPi_W[0] = Ai.alpha_W_LPi[0]*(1-Ai.alpha_LPi_ai[0])**w
        Ai.alter_LPi_W[1] = Ai.alpha_W_LPi[1]*(1-Ai.alpha_LPi_ai[1])**w

def UpdateRemotepolicies(run, numberSaveData, gama, w):
#================== Update Remote Policies (RP) for Ai
    agentList=[]
    for Ai in Agent:
        agentList.append(Ai)
    for idAi, Ai in enumerate(Agent):
        LPij=1-idAi
#=================Update Ai's RPi
        for indxR, Q_RP in enumerate(Ai.Q_RPi):
            alphaQR_AiRPi_ak = Ai.alpha_Q_ak_Function(agentList[LPij].x_LPi[indxR], Ai.actionWinIndex,\
                                                        Ai.Num_Visited_RPi_x_ak[indxR], 1)

            Ai.updateQ(agentList[LPij].x_LPi[indxR], Ai.actionWinIndex, agentList[LPij].y_LPi[indxR],\
                        Ai.actionSub_y[indxR], agentList[LPij].R[indxR], alphaQR_AiRPi_ak, gama, Ai.Q_RPi[indxR])

        if Ai.policyWin=="LP1" or Ai.policyWin=="LP2":
            alphaQR_Ai_RP1_ai = Ai.alpha_Q_ak_Function(agentList[LPij].x_LPi[0],  Ai.actionWant_RPi[0],\
                                                Ai.Num_Visited_RPi_x_ak[0], 0)
            alphaAi_W_RP1 = Ai.alpha_W_Function(agentList[LPij].x_LPi[0], Ai.Num_Visited_RPi_x[0], 1)
            Ai.updateW(agentList[LPij].x_LPi[0], Ai.actionWant_RPi[0], agentList[LPij].y_LPi[0],\
                        Ai.actionSub_y[0], agentList[LPij].R[0], alphaAi_W_RP1, alphaQR_Ai_RP1_ai,\
                        gama, w, Ai.W_RPi[0], Ai.Q_RPi[0])
            alterAi_RP1_W = alphaAi_W_RP1*(1-alphaQR_Ai_RP1_ai)**w

            alphaQR_Ai_RP2_ai = Ai.alpha_Q_ak_Function(agentList[LPij].x_LPi[1],  Ai.actionWant_RPi[1],\
                        Ai.Num_Visited_RPi_x_ak[1], 0)
            alphaAi_W_RP2 = Ai.alpha_W_Function(agentList[LPij].x_LPi[1], Ai.Num_Visited_RPi_x[1], 1)
            Ai.updateW(agentList[LPij].x_LPi[1], Ai.actionWant_RPi[1], agentList[LPij].y_LPi[1],\
                        Ai.actionSub_y[1], agentList[LPij].R[1], alphaAi_W_RP2, alphaQR_Ai_RP2_ai,\
                        gama, w, Ai.W_RPi[1], Ai.Q_RPi[1])
            alterAi_RP2_W = alphaAi_W_RP2*(1-alphaQR_Ai_RP2_ai)**w

        else:
            if Ai.RPiWin == "RP2":
                alphaQR_Ai_RP1_ai = Ai.alpha_Q_ak_Function(agentList[LPij].x_LPi[0],  Ai.actionWant_RPi[0],\
                                                    Ai.Num_Visited_RPi_x_ak[0], 0)
                alphaAi_W_RP1 = Ai.alpha_W_Function(agentList[LPij].x_LPi[0], Ai.Num_Visited_RPi_x[0], 1)
                Ai.updateW(agentList[LPij].x_LPi[0], Ai.actionWant_RPi[0], agentList[LPij].y_LPi[0],\
                            Ai.actionSub_y[0], agentList[LPij].R[0], alphaAi_W_RP1, alphaQR_Ai_RP1_ai,\
                            gama, w, Ai.W_RPi[0], Ai.Q_RPi[0])
                alterAi_RP1_W = alphaAi_W_RP1*(1-alphaQR_Ai_RP1_ai)**w
            else:
                alphaQR_Ai_RP2_ai = Ai.alpha_Q_ak_Function(agentList[LPij].x_LPi[1],  Ai.actionWant_RPi[1],\
                            Ai.Num_Visited_RPi_x_ak[1], 0)
                alphaAi_W_RP2 = Ai.alpha_W_Function(agentList[LPij].x_LPi[1], Ai.Num_Visited_RPi_x[1], 1)
                Ai.updateW(agentList[LPij].x_LPi[1], Ai.actionWant_RPi[1], agentList[LPij].y_LPi[1],\
                            Ai.actionSub_y[1], agentList[LPij].R[1], alphaAi_W_RP2, alphaQR_Ai_RP2_ai,\
                            gama, w, Ai.W_RPi[1], Ai.Q_RPi[1])
                alterAi_RP2_W = alphaAi_W_RP2*(1-alphaQR_Ai_RP2_ai)**w
            
        # used for analytics, saved in controlFile
        Ai.alpha_RPi_ak[0] = Ai.alpha_Q_ak_Function(agentList[LPij].x_LPi[0], Ai.actionWinIndex,\
                            Ai.Num_Visited_RPi_x_ak[0], 0)
        Ai.alpha_RPi_ak[1] = Ai.alpha_Q_ak_Function(agentList[LPij].x_LPi[1], Ai.actionWinIndex,\
                            Ai.Num_Visited_RPi_x_ak[1], 0)
        
        Ai.alpha_RPi_ai[0] = Ai.alpha_Q_ak_Function(agentList[LPij].x_LPi[0],\
                                                    Ai.actionWant_RPi[0], Ai.Num_Visited_RPi_x_ak[0], 0)
        Ai.alpha_RPi_ai[1] = Ai.alpha_Q_ak_Function(agentList[LPij].x_LPi[1],\
                                                    Ai.actionWant_RPi[1], Ai.Num_Visited_RPi_x_ak[1], 0)

        Ai.alpha_W_RPi[0] = Ai.alpha_W_Function(agentList[LPij].x_LPi[0], Ai.Num_Visited_RPi_x[0], 0)
        Ai.alpha_W_RPi[1] = Ai.alpha_W_Function(agentList[LPij].x_LPi[1], Ai.Num_Visited_RPi_x[1], 0)

        Ai.alter_RPi_W[0] = Ai.alpha_W_RPi[0]*(1-Ai.alpha_RPi_ai[0])**w
        Ai.alter_RPi_W[1] = Ai.alpha_W_RPi[1]*(1-Ai.alpha_RPi_ai[1])**w

def Clean_up_TTS_var():
    """
    Both agents have stored all sections' TTS. So currently A1.TTS_sectionLi = A2.TTS_sectionLi.
    """
    # print(f'agent TTS_L1 id: {id(A1.TTS_sectionL1)},\n agent TTS_L1: {A1.TTS_sectionL1}')
    A1.TTS_old_sectionL0 = A1.TTS_sectionL0    
    A1.TTS_old_sectionL1 = A1.TTS_sectionL1
    A1.TTS_old_sectionL2 = A1.TTS_sectionL2
    A1.TTS_old_sectionL3 = A1.TTS_sectionL3

    A2.TTS_old_sectionL0 = A2.TTS_sectionL0
    A2.TTS_old_sectionL1 = A2.TTS_sectionL1
    A2.TTS_old_sectionL2 = A2.TTS_sectionL2
    A2.TTS_old_sectionL3 = A2.TTS_sectionL3

   
    A1.TTS_sectionL0 = 0
    A1.TTS_sectionL1 = 0
    A1.TTS_sectionL2 = 0
    A1.TTS_sectionL3 = 0

    A2.TTS_sectionL0 = 0
    A2.TTS_sectionL1 = 0
    A2.TTS_sectionL2 = 0
    A2.TTS_sectionL3 = 0

def Clean_up_TTS_Net_Sim_var():
    """
    Clean TTS_Net_Sim variable before the simulation ends!
    This enable new TTS_Net_Sim variables start from 0 in new simulation run!
    (Both agents store the same amount in TTS_Net_Sim)
    """
    A1.TTS_Net_Sim = 0
    A2.TTS_Net_Sim = 0    


def updateCurrentSpeedStateAsPrevious(j):
    """
    Remove the current "last" time step state as the first element in new cycles.
    We measure states each 50 [s] thus 4 times per control time step 150 [s].
    However, we always use last "current" measurements as relevant for states
    (e.g., one can play and try to use average of all measurements)
    """
    A1.speedSectionL0[0] = A1.speedSectionL0[j]
    A1.speedSectionL0[1:] = 0
    A1.speedSectionL1[0] = A1.speedSectionL1[j]
    A1.speedSectionL1[1:] = 0
    A1.speedSectionL2[0] = A1.speedSectionL2[j]
    A1.speedSectionL2[1:] = 0
    A1.speedSectionL3[0] = A1.speedSectionL3[j]
    A1.speedSectionL3[1:] = 0

    A2.speedSectionL0[0] = A2.speedSectionL0[j]
    A2.speedSectionL0[1:] = 0
    A2.speedSectionL1[0] = A2.speedSectionL1[j]
    A2.speedSectionL1[1:] = 0
    A2.speedSectionL2[0] = A2.speedSectionL2[j]
    A2.speedSectionL2[1:] = 0
    A2.speedSectionL3[0] = A2.speedSectionL3[j]
    A2.speedSectionL3[1:] = 0

def updateCurrentDensityStateAsPrevious(j):
    """
    Remove the current time step state as the first element in new control cycles.
    We measure states each 50 [s] thus 4 times per control time step at 0, 50, 100, and 150 [s].
    However, we always use last "current" measurements as relevant for states 
    (one can play and try to use average, rate of state changes, etc.)
    """
    A1.densitySectionL0[0] = A1.densitySectionL0[j]
    A1.densitySectionL0[1:] = 0
    A1.densitySectionL1[0] = A1.densitySectionL1[j]
    A1.densitySectionL1[1:] = 0
    A1.densitySectionL2[0] = A1.densitySectionL2[j]
    A1.densitySectionL2[1:] = 0
    A1.densitySectionL3[0] = A1.densitySectionL3[j]
    A1.densitySectionL3[1:] = 0

    A2.densitySectionL0[0] = A2.densitySectionL0[j]
    A2.densitySectionL0[1:] = 0
    A2.densitySectionL1[0] = A2.densitySectionL1[j]
    A2.densitySectionL1[1:] = 0
    A2.densitySectionL2[0] = A2.densitySectionL2[j]
    A2.densitySectionL2[1:] = 0
    A2.densitySectionL3[0] = A2.densitySectionL3[j]
    A2.densitySectionL3[1:] = 0


def DeepCopyOfStates():
    """
    Set X(t-1) --> X(t)
    Set X(t+1) --> X(t), or in lingo of RL: X' --> X 
    """
    A1.x_old_LPi = deepcopy(A1.x_LPi) 
    A2.x_old_LPi = deepcopy(A2.x_LPi)

    A1.x_LPi=deepcopy(A1.y_LPi)
    A2.x_LPi=deepcopy(A2.y_LPi)

# nebi bilo lose dodati random ispis uz pobjednicku strategiju na kraju file-a
def ControlFileVStack(controlFile, j):
    controlFile = np.vstack([controlFile,[str(A1.x_old_LPi[0])+'-'+str(A1.x_old_LPi[1]),\
                                        str(A2.x_old_LPi[0])+'-'+str(A2.x_old_LPi[1]),\
                                        format(A1.speedSectionL0[0],".1f"),\
                                        format(A1.speedSectionL1[0],".1f"),\
                                        format(A1.speedSectionL2[0],".1f"),\
                                        format(A1.speedSectionL3[0],".1f"),\
                                        format(A1.densitySectionL0[0],".1f"),\
                                        format(A1.densitySectionL1[0],".1f"),\
                                        format(A1.densitySectionL2[0],".1f"),\
                                        format(A1.densitySectionL3[0],".1f"),\
                                        format(A1.actions_mps[0, A1.prevActionWinIndex],".1f"),\
                                        format(A2.actions_mps[0, A2.prevActionWinIndex],".1f"),\
                                        format(A1.TTS_old_sectionL0,".2f"),\
                                        format(A1.TTS_old_sectionL1,".2f"),\
                                        format(A1.TTS_old_sectionL2,".2f"),\
                                        format(A1.TTS_old_sectionL3,".2f"),\
                                        format(A1.TTS_sectionL3_haltingVeh_ramp,".2f"),\
                                        0,\
                                        format(A1.R[0],".2f"),\
                                        format(A1.R[1],".2f"),\
                                        format(A2.R[0],".2f"),\
                                        format(A2.R[1],".2f"),\
                                        str(A1.x_LPi[0])+'-'+str(A1.x_LPi[1]),\
                                        str(A2.x_LPi[0])+'-'+str(A2.x_LPi[1]),\
                                        format(A1.speedSectionL0[j],".1f"),\
                                        format(A1.speedSectionL1[j],".1f"),\
                                        format(A1.speedSectionL2[j],".1f"),\
                                        format(A1.speedSectionL3[j],".1f"),\
                                        format(A1.densitySectionL0[j],".1f"),\
                                        format(A1.densitySectionL1[j],".1f"),\
                                        format(A1.densitySectionL2[j],".1f"),\
                                        format(A1.densitySectionL3[j],".1f"),\
                                        format(A1.actions_mps[0,A1.actionWinIndex],".1f"),\
                                        A1.actionWant_LPi[0],\
                                        A1.actionWant_LPi[1],\
                                        A1.actionWant_RPi[0],\
                                        A1.actionWant_RPi[1],\
                                        A1.policyWin,\
                                        A1.actionControl_LPi[0],\
                                        A1.actionControl_LPi[1],\
                                        A1.actionControl_RPi[0],\
                                        A1.actionControl_RPi[1],\
                                        format(A2.actions_mps[0,A2.actionWinIndex],".1f"),\
                                        A2.actionWant_LPi[0],\
                                        A2.actionWant_LPi[1],\
                                        A2.actionWant_RPi[0],\
                                        A2.actionWant_RPi[1],\
                                        A2.policyWin,\
                                        A2.actionControl_LPi[0],\
                                        A2.actionControl_LPi[1],\
                                        A2.actionControl_RPi[0],\
                                        A2.actionControl_RPi[1],\
                                        format(A1.W_LPi[0][A1.x_LPi[0],0],".6f"),\
                                        format(A1.W_LPi[1][A1.x_LPi[1],0],".6f"),\
                                        format(A1.W_RPi[0][A2.x_LPi[0],0],".6f"),\
                                        format(A1.W_RPi[1][A2.x_LPi[1],0],".6f"),\
                                        format(A2.W_LPi[0][A2.x_LPi[0],0],".6f"),\
                                        format(A2.W_LPi[1][A2.x_LPi[1],0],".6f"),\
                                        format(A2.W_RPi[0][A1.x_LPi[0],0],".6f"),\
                                        format(A2.W_RPi[1][A1.x_LPi[1],0],".6f"),\
                                        format(A1.alpha_LPi_ai[0],".7f"),\
                                        format(A1.alpha_LPi_ai[1],".7f"),\
                                        format(A1.alpha_RPi_ai[0],".7f"),\
                                        format(A1.alpha_RPi_ai[1],".7f"),\
                                        format(A2.alpha_LPi_ai[0],".7f"),\
                                        format(A2.alpha_LPi_ai[1],".7f"),\
                                        format(A2.alpha_RPi_ai[0],".7f"),\
                                        format(A2.alpha_RPi_ai[1],".7f"),\
                                        format(A1.alpha_LPi_ak[0],".7f"),\
                                        format(A1.alpha_LPi_ak[1],".7f"),\
                                        format(A1.alpha_RPi_ak[0],".7f"),\
                                        format(A1.alpha_RPi_ak[1],".7f"),\
                                        format(A2.alpha_LPi_ak[0],".7f"),\
                                        format(A2.alpha_LPi_ak[1],".7f"),\
                                        format(A2.alpha_RPi_ak[0],".7f"),\
                                        format(A2.alpha_RPi_ak[1],".7f"),\
                                        format(A1.alpha_W_LPi[0],".7f"),\
                                        format(A1.alpha_W_LPi[1],".7f"),\
                                        format(A1.alpha_W_RPi[0],".7f"),\
                                        format(A1.alpha_W_RPi[1],".7f"),\
                                        format(A2.alpha_W_LPi[0],".7f"),\
                                        format(A2.alpha_W_LPi[1],".7f"),\
                                        format(A2.alpha_W_RPi[0],".7f"),\
                                        format(A2.alpha_W_RPi[1],".7f"),\
                                        format(A1.alter_LPi_W[0],".7f"),\
                                        format(A1.alter_LPi_W[1],".7f"),\
                                        format(A1.alter_RPi_W[0],".7f"),\
                                        format(A1.alter_RPi_W[1],".7f"),\
                                        format(A2.alter_LPi_W[0],".7f"),\
                                        format(A2.alter_LPi_W[1],".7f"),\
                                        format(A2.alter_RPi_W[0],".7f"),\
                                        format(A2.alter_RPi_W[1],".7f"),\
                                        format(A1.TTS_Net_Sim,".2f"),\
                                        format(0,".4f"),\
                                        format(0,".4f"),\
                                        format(0,".4f"),\
                                        A1.actionWinIndex,\
                                        A1.LPiWin,\
                                        A1.RPiWin,\
                                        A2.actionWinIndex,\
                                        A2.LPiWin,\
                                        A2.RPiWin,\
                                        A1.VSLposition,\
                                        A2.VSLposition]])
    return controlFile


def DeepCopyActionWin():
    A1.prevActionWinIndex = deepcopy(A1.actionWinIndex)     
    A2.prevActionWinIndex = deepcopy(A2.actionWinIndex)


