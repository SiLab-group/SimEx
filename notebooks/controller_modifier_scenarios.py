import random


class Controller:
    def get_list_of_modifiers(number_of_scenarios):
        modifier = []
        # flow_list = []
        for ID in range(1, number_of_scenarios + 1, 1):
            flow_list = []
            print(f"ID is {ID}")
            for i in range(0, 90, 1):
                # Keep it simple for the first implementation, later we can add some function-flow distribution
                flow_list.append(2800 + ID * 100)  #random.randint(2800,3800)
            modifier.append([ID, flow_list])
            print(f"Time is the {len(flow_list)} in min and flow list {flow_list}")
        return modifier
