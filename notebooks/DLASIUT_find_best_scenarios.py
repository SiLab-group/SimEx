import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# df_baseline = pd.read_csv('simex_output-NOVSL-2024-07-04-14-05-41.csv')
# df_control = pd.read_csv('simex_output-VSL-2024-07-04-14-05-47.csv')
#
# # Main Function recives np.arrays
# dataset_baseline = df_baseline.to_numpy()
# dataset_control = df_control.to_numpy()

def automatic_detection_of__performance(dataset_baseline, dataset_control,  incremnet_step_for_x = 10, max_order_of_polynom = 9, tolerance_in_diffrence=12):

    lower_bound_of_x = int(dataset_baseline[0,0])
    upper_bound_of_x = int(dataset_baseline[dataset_baseline.shape[0]-1,1])

    half_of_interval_step_for_x = int(incremnet_step_for_x/2)

    output_with_intervals_and_results = np.array([]).reshape((0,3))
    result = []
    
    for _lower__value_in_x_interval in range(lower_bound_of_x, upper_bound_of_x , incremnet_step_for_x): #Main interator trouugh x-axis - use case flow [veh/h]
        #Find i in ranges in tables,
        #then load coresponding polyinom
        #Do the same for baseline and control cases
   
        current_x_for_computing_polynom = _lower__value_in_x_interval + half_of_interval_step_for_x

        # Baseline scenario
        for id_of_selected_polynom_interval in range(0, dataset_baseline.shape[0]):

            # Find in which polynom interval is current_x_for_computing_polynom
            if int(dataset_baseline[id_of_selected_polynom_interval,0]) < current_x_for_computing_polynom and current_x_for_computing_polynom < int( dataset_baseline[id_of_selected_polynom_interval,1]):
                
                selcted_polynom_coded_in_list_baseline = dataset_baseline[id_of_selected_polynom_interval, 2 :12]
        
                #Compute values of polynom, first for baseline , than for control case
                result_of_polynom_baseline = np.polyval(np.poly1d(selcted_polynom_coded_in_list_baseline), current_x_for_computing_polynom)

        # Control scenario
        for id_of_selected_polynom_interval1 in range(0, dataset_control.shape[0]):
            selcted_polynom_coded_in_list_control = []
            if dataset_control[id_of_selected_polynom_interval1,0] < current_x_for_computing_polynom and current_x_for_computing_polynom <= dataset_control[id_of_selected_polynom_interval1,1]:
    
                selcted_polynom_coded_in_list_control = dataset_control[id_of_selected_polynom_interval1, 2 : 12]
             
                #Compute values of polynom, first for baseline , than for control case
                result_of_polynom_control = np.polyval(np.poly1d(selcted_polynom_coded_in_list_control), current_x_for_computing_polynom)
            
        #If control case for specific increment_step_for_x lower than baseline mark 1 othervise 0
        # Added_tolerance
        interval_for_exploration = 0
        if result_of_polynom_baseline < result_of_polynom_control:
            if tolerance_in_diffrence < result_of_polynom_control - result_of_polynom_baseline:
                interval_for_exploration = 1
                result.append((int(_lower__value_in_x_interval),int(_lower__value_in_x_interval+ incremnet_step_for_x)))
        else:
            interval_for_exploration = 0
        
        #print(current_x_for_computing_polynom, result_of_polynom_baseline , result_of_polynom_control, interval_for_exploration)

        output_with_intervals_and_results = np.append(output_with_intervals_and_results, [[int(_lower__value_in_x_interval), int(_lower__value_in_x_interval + incremnet_step_for_x), int(interval_for_exploration)]], axis=0)
    
    # output matrice
    # 1 column - start od interval, 2 - column - end of interval 3- column  exploration or not (binary)          
    return output_with_intervals_and_results


# Just for vizualization
def visualization(dataset_baseline, dataset_control,  _incremnet_step_for_x = 10, _max_order_of_polynom = 9, _tolerance_in_diffrence=12):

    lower_bound_of_x = int(dataset_baseline[0,0])
    upper_bound_of_x = int(dataset_baseline[dataset_baseline.shape[0]-1,1])
    
    # Call main function for detection
    results = automatic_detection_of__performance(dataset_baseline, dataset_control, incremnet_step_for_x = _incremnet_step_for_x,  max_order_of_polynom =  _max_order_of_polynom, tolerance_in_diffrence=_tolerance_in_diffrence)
    # print((results))
    # print(results)
    # Result of intersection of two set of polynom --- control and basline
    x_axis = list(range(lower_bound_of_x,upper_bound_of_x))
    
    ploting_polynomal_results_baseline = []
    for i in range(0, dataset_baseline.shape[0]):
        vectorized_polynom = dataset_baseline[i, 2 : 2 + _max_order_of_polynom + 1  ]
        interval_of_plynom = list((range(int(dataset_baseline[i,0]), int(dataset_baseline[i,1])) )) 
        
        resulting_vector_polynom = np.polyval(np.poly1d(vectorized_polynom), interval_of_plynom)
        ploting_polynomal_results_baseline = [*ploting_polynomal_results_baseline, *resulting_vector_polynom] 
        
        colour_code = (np.random.random(), 0, 0)
        plt.plot(interval_of_plynom,resulting_vector_polynom, c = colour_code, label= "No control scenario " + '[' + str(int(dataset_baseline[i,0])) + ',' +  str(int(dataset_baseline[i,1])) + ']')
        
    #plt.plot(x_axis,ploting_polynomal_results_baseline, 'r-', label= 'No VSL scenario')
    
    ploting_polynomal_results_baseline = []
    for i in range(0, dataset_control.shape[0]):
        vectorized_polynom = dataset_control[i, 2 : 2 + _max_order_of_polynom + 1  ]
        interval_of_plynom = list((range(int(dataset_control[i,0]), int(dataset_control[i,1])) )) 
        
        resulting_vector_polynom = np.polyval(np.poly1d(vectorized_polynom), interval_of_plynom)
        ploting_polynomal_results_baseline = [*ploting_polynomal_results_baseline, *resulting_vector_polynom] 
        
        colour_code = (0, np.random.random(), 0)
        plt.plot(interval_of_plynom,resulting_vector_polynom, c= colour_code, linestyle='dashed', label="VSL scenario " + '[' + str(int(dataset_control[i,0])) + ',' +  str(int(dataset_control[i,1])) + ']')
        
    #plt.plot(x_axis,ploting_polynomal_results_baseline, 'g', label='VSL scenario')

    plt.bar(results[:,0]+ int(_incremnet_step_for_x/2), results[:,2]*900, width=_incremnet_step_for_x, alpha=0.25, label= 'Detail exploration needed')
    
    plt.xlabel("Traffic flow [veh/h]")
    plt.ylabel("TTS [veh$\cdot$s]")
    #plt.ylabel("Polynom values")
    #plt.xticks(range(lower_bound_of_x, upper_bound_of_x, 250))
    plt.ylim((0,900))
    plt.xlim((lower_bound_of_x,upper_bound_of_x))
    plt.legend(loc='upper left')
    plt.show()

def automatic_performance(dataset_baseline, dataset_control, incremnet_step_for_x=10,
                                        max_order_of_polynom=9, tolerance_in_diffrence=12):

    lower_bound_of_x = int(dataset_baseline[0, 0])
    upper_bound_of_x = int(dataset_baseline[dataset_baseline.shape[0] - 1, 1])

    half_of_interval_step_for_x = int(incremnet_step_for_x / 2)

    output_with_intervals_and_results = np.array([]).reshape((0, 3))
    result = []

    for _lower__value_in_x_interval in range(lower_bound_of_x, upper_bound_of_x,
                                             incremnet_step_for_x):  # Main interator trouugh x-axis - use case flow [veh/h]
        # Find i in ranges in tables,
        # then load coresponding polyinom
        # Do the same for baseline and control cases

        current_x_for_computing_polynom = _lower__value_in_x_interval + half_of_interval_step_for_x

        # Baseline scenario
        for id_of_selected_polynom_interval in range(0, dataset_baseline.shape[0]):

            # Find in which polynom interval is current_x_for_computing_polynom
            if int(dataset_baseline[
                       id_of_selected_polynom_interval, 0]) < current_x_for_computing_polynom and current_x_for_computing_polynom < int(
                    dataset_baseline[id_of_selected_polynom_interval, 1]):
                selcted_polynom_coded_in_list_baseline = dataset_baseline[id_of_selected_polynom_interval, 2:12]

                # Compute values of polynom, first for baseline , than for control case
                result_of_polynom_baseline = np.polyval(np.poly1d(selcted_polynom_coded_in_list_baseline),
                                                        current_x_for_computing_polynom)

        # Control scenario
        for id_of_selected_polynom_interval1 in range(0, dataset_control.shape[0]):
            selcted_polynom_coded_in_list_control = []
            if dataset_control[
                id_of_selected_polynom_interval1, 0] < current_x_for_computing_polynom and current_x_for_computing_polynom <= \
                    dataset_control[id_of_selected_polynom_interval1, 1]:
                selcted_polynom_coded_in_list_control = dataset_control[id_of_selected_polynom_interval1, 2: 12]

                # Compute values of polynom, first for baseline , than for control case
                result_of_polynom_control = np.polyval(np.poly1d(selcted_polynom_coded_in_list_control),
                                                       current_x_for_computing_polynom)

        # If control case for specific increment_step_for_x lower than baseline mark 1 othervise 0
        # Added_tolerance
        interval_for_exploration = 0
        if result_of_polynom_baseline < result_of_polynom_control:
            if tolerance_in_diffrence < result_of_polynom_control - result_of_polynom_baseline:
                interval_for_exploration = 1
                result.append(
                    (int(_lower__value_in_x_interval), int(_lower__value_in_x_interval + incremnet_step_for_x)))
        else:
            interval_for_exploration = 0

        # print(current_x_for_computing_polynom, result_of_polynom_baseline , result_of_polynom_control, interval_for_exploration)

        output_with_intervals_and_results = np.append(output_with_intervals_and_results, [
            [int(_lower__value_in_x_interval), int(_lower__value_in_x_interval + incremnet_step_for_x),int(interval_for_exploration)]], axis=0)

    # print(f"{output_with_intervals_and_results}")
    # output matrice
    # 1 column - start od interval, 2 - column - end of interval 3- column  exploration or not (binary)
    intervals = []
    for n, el in enumerate(output_with_intervals_and_results):
        if el[2] == 1:
            intervals.append([el[0], el[1]])
    print(f"INTERVALS: {intervals}")
    if intervals:
        intervals.sort(key=lambda interval: interval[0])
        merged = [intervals[0]]
        for current in intervals:
            previous = merged[-1]
            if current[0] <= previous[1]:
                previous[1] = max(previous[1], current[1])
            else:
                merged.append(current)
            # print(intervals)
            #print(f"MERGED: {merged})
    else:
         merged = None
    return merged



# # Call visulasiation with main function in it
# visualization(dataset_baseline, dataset_control, _incremnet_step_for_x = 10, _tolerance_in_diffrence  = 12)
# _incremnet_step_for_x = 10
# _max_order_of_polynom = 9
# _tolerance_in_diffrence=12
# results = automatic_performance(dataset_baseline, dataset_control, incremnet_step_for_x = _incremnet_step_for_x,  max_order_of_polynom =  _max_order_of_polynom, tolerance_in_diffrence=_tolerance_in_diffrence)
# print(results)
