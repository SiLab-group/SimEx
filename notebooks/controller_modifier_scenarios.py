import random

from typing import List
from modifier import Modifier
from global_settings import sumovsls

class Controller:
    def get_list_of_modifiers(number_of_scenarios):
        modifier = []
        # flow_list = []
        for ID in range(1, number_of_scenarios + 1, 1):
            flow_list = []
            #print(f"ID is {ID}")
            for i in range(0, 90, 1):
                # Keep it simple for the first implementation, later we can add some function-flow distribution
                flow_list.append(2800 + ID * 100)  #random.randint(2800,3800)
            modifier.append([ID, flow_list])
            #print(f"Time is the {len(flow_list)} in min and flow list {flow_list}")
        return modifier

    def modifier_sumo(interval_list, modifier_list):
        # Get interval [1.25, 2.75]
        # Get the values from the modifier array for the interval ID,[value,...]
        # Resample traffic volume with different granularity
        interval_ids = [[int(num) for num in interval] for interval in interval_list]
        mlist = []
        for interval in interval_ids:
            ids = []
            volume = []
            print(f"Print interval {interval}")
            # Get traffic volume for each id
            # We assume, constant volume per interval
            start = modifier_list[interval[0] - 1][1][0]
            end = modifier_list[interval[1] - 1][1][0]
            # Create step for traffic volume
            step = 25
            for value in range(start, end, step):
                volume.append(value)

            ids = Modifier.rescale(volume, interval[0], interval[1])
            print(f"volume {volume}")
            for i in range(0, len(ids) - 1):
                mlist.append([ids[i], [volume[i]] * 90])
            print(f"MOD list {mlist}")
        return mlist
