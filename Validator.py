import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

class Validator:
    def __init__(self):
        self.archive_sym = []
        self.archive_mod = []
        self.iterations = 0
        self.total_points = 0
        self.total_bad_points = 0
    
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
    
    def fit_curve(self,x_values,y_values):
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
    
    def get_unfitting_ranges(self,mod_x_list,sim_y_list,threshold=0.9):
        # apply curve fit to new data
        fitted_curve = self.fit_curve(mod_x_list,sim_y_list)
        # get points of unfit
        unfit_points, fit_points = self.thresholding(fitted_curve,threshold)

        # create ranges from continuous unfit points
        if self.iterations == 0:
            unfit_ranges = [[0,4],[40,50]]

        self.collect_data(sim_y_list,mod_x_list,unfit_ranges,unfit_points=unfit_points)
        # return unfit ranges
        self.collect_data()
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
