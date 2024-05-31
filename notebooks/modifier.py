import numpy as np


class Modifier:

    def rescale(old_list, new_min, new_max):
        """
        Rescales a list of values from the original range to a new range.

        Args:
            old_list (list): A list of numeric values to be rescaled.
            new_min (float): The minimum value of the new range.
            new_max (float): The maximum value of the new range.

        Returns:
            list: A new list of rescaled values.

        Example:
            >>> old_values = [10, 20, 30, 40]
            >>> new_min = 0
            >>> new_max = 1
            >>> rescaled_values = rescale(old_values, new_min, new_max)
            >>> print(rescaled_values)
            [0.0, 0.25, 0.5, 1.0]
        """
        # handle empty list case
        if not np.any(old_list):
            return []

        old_min = min(old_list)
        old_max = max(old_list)

        if old_min == old_max:
            # Handle the case when all elements in old_list are the same
            return [new_min] * len(old_list)

        new_values = []
        for old_value in old_list:
            denominator = old_max - old_min
            if denominator != 0:
                scaled_value = (((old_value - old_min) * (new_max - new_min)) / denominator) + new_min
                new_values.append(scaled_value)
            else:
                # Handle the case when the interval is zero
                new_values.append(new_min)
        return new_values

    def modifierA(x, new_min, new_max):
        """
        Applies a rescaling operation to the input value x.

        Args:
            x (float): The input value.
            new_min (float): The desired minimum value after rescaling.
            new_max (float): The desired maximum value after rescaling.

        Returns:
            float: The rescaled value of x.

        Example:
            >>> modifierA(5, 0, 10)
            25.0
        """
        temp = np.array(x) ** 2
        temp = Modifier.rescale(temp, new_min, new_max)
        return temp

    def modifierB(x, new_min, new_max):
        temp = x * 2 / 3
        temp = Modifier.rescale(temp, new_min, new_max)
        return temp

    def sumo_modifier(x, num_scenarios, new_min, new_max):
        modifier = []
        # flow_list = []
        temp = x * 2 / 3
        for ID in range(1, number_of_scenarios + 1, 1):
            flow_list = []
            # print(f"ID is {ID}")
            # for i in range(0, 90, 1):
            #     # Keep it simple for the first implementation, later we can add some function-flow distribution
            #     flow_list.append(new_min + ID * 100)  # random.randint(2800,3800)
            # modifier.append([ID, flow_list])
            # print(f"Time is the {len(flow_list)} in min and flow list {flow_list}")
            # modifier.append(new_min + ID * 100)
        return modifier