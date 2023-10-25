import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from global_settings import fitting_threshold, mdv


# import global here: threshold

class Validator:
    def __init__(self): #i.e. if first iteration
        self.iterations = 1
        self.total_points = 0
        self.total_bad_points = 0
        self.range = (mdv["domain_min_range"],mdv["domain_max_range"])  # Initialize the range as a tuple
        self.num_points_evaluated = 0  # Initialize num_points_evaluated

    def update_history(self, new_iterations, new_total_points, new_range):
        self.iterations += new_iterations
        self.total_points += new_total_points
        self.range = new_range

    def update_num_points_evaluated(self, points, min_x, max_x):
        self.num_points_evaluated = len([(x, y) for x, y in points if min_x <= x <= max_x])
    
    def local_exploration_validator_A():
        # generate fit function 
        fit_curve()
        # find misfit points from mod_x_list and sim_y_list (outliers) using threshold from fit function
        get_unfitting_point()
        # generate ranges of misfit points (make it fancy threshold)
        get_unfitting_ranges()
        ## update your history (for each iteration: 

        # Update history with new values
        Validator.update_history(5, 100, (10, 20))  # For example, 5 new iterations, 100 new total points, and new range

        # Update num_points_evaluated (self, new_iterations, new_total_points, new_range)
        Validator.update_num_points_evaluated(points, min_x, max_x)

        # Print updated values
        print(f"Iterations: {Validator.iterations}")
        print(f"Total Points: {Validator.total_points}")
        print(f"Range: {Validator.range}")
        print(f"Num Points Evaluated: {Validator.num_points_evaluated}")

        # iteration, 
        # total points evaluated (good and misfit), 
        # points evaluated this iteration (good and misfit), 
        # number of misfit ranges)

        # num_points_input = len(mod_x_list)
        # total_points.append(mod_x_list)

        # # Get range from previous iteration range generation
        # min_x,max_x = range_iteration 
        # num_points_evaluated = [(x, y) for x, y in total_points if min_x <= x <= max_x]


    def validator_controller(mod_x_list,sim_y_list, global_range=[mdv["domain_min_range"],mdv["domain_max_range"]],threshold=fitting_threshold, local_validator=local_exploration_validator_A, do_plot=False):
        # gets points mod_x_list, sim_y_list
        validator_ranges=local_exploration_validator_A()

        # if not first time accessing validator, merge old points with new
            # i.e. merge all data points together
        return validator_ranges
        

    def collect_data(self, sym, mod, ranges, unfit_points):
        self.archive_sym.append(sym)
        self.archive_mod.append(mod)
        # track number of iterations
        self.iterations = self.iterations+1
        self.total_points = self.total_points+len(mod)
        # track number of good/bad points
        self.total_unfit_points = self.total_unfit_points+len(mod)
    
    def update_statistics(self, new_sym, new_mod):
        # track number of points generated from input to get_unfitting_ranges
        self.collect_data(new_sym,new_mod)
        # track number of unfit intervals i.e. append length of ranges each itteration)
        self.history.extend(new_sym)
    
    def fit_curve(x_values,y_values):
        # Assuming you have a function to fit a curve to the data
        # Replace the placeholder code below with your curve fitting logic

        # standardize    
        x_scaler, y_scaler = StandardScaler(), StandardScaler()
        x_train = x_scaler.fit_transform(x_values)
        y_train = y_scaler.fit_transform(y_values)

        # fit model
        model = HuberRegressor(epsilon=1)
        model.fit(x_train, y_train.ravel())

        # Fit curve to the data
        fitted_curve = np.polyfit(x_values, y_values, 1)
        return fitted_curve
    
    def thresholding(self, threshold):
        # Assuming you want to return x values above the threshold
        x_values = np.arange(len(self.history))
        y_values = np.array(self.history)
        above_threshold = x_values[y_values > threshold]
        return above_threshold
    
    def get_unfitting_ranges(mod_x_list,sim_y_list,threshold=fitting_threshold):
        # apply curve fit to new data
        fitted_curve = Validator.fit_curve(mod_x_list,sim_y_list)
        # get points of unfit
        unfit_points, fit_points = Validator.thresholding(fitted_curve,threshold)

        # create ranges from continuous unfit points
        if Validator.iterations == 0:
            unfit_ranges = [[0,4],[40,50]]

        Validator.collect_data(sim_y_list,mod_x_list,unfit_ranges,unfit_points=unfit_points)
        # return unfit ranges
        Validator.collect_data()
        return unfit_ranges

# # Example usage
# analyzer = Validator()
# data = [1, 2, 3, 4]
# analyzer.collect_data(data)
# new_data = [5, 6, 7]
# analyzer.update_history(new_data)
# print(analyzer.history)  # Output: [1, 2, 3, 4, 5, 6, 7]

# fitted_curve = analyzer.fit_curve()
# print(fitted_curve)  # Output: [slope, intercept]

# threshold = 3
# above_threshold = analyzer.thresholding(threshold)
# print(above_threshold)  # Output: [3, 4, 5, 6]
