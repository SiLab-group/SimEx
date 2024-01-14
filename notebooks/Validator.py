## VALIDATOR FILE: :
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from global_settings import mdv,vls,vlv
from itertools import compress
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

class Validator:
    def __init__(self):
        self.iterations = 1
        self.total_points = 0
        self.total_bad_points = 0
        self.range = (mdv["domain_min_range"], mdv["domain_max_range"])
        self.num_points_evaluated = 0
        
        self.least_fit_intercept = None
        self.least_fit_y_pred = None
        self.least_fit_x_range = None
        self.least_fit_points = None


    def fit_curve(self,x_values, y_values, global_range=0):
        print('       *** USING fit_curve')
        
        x_values = np.array(x_values)  # Convert to numpy array
        y_values = np.array(y_values)  # Convert to numpy array

        # Set the degree of the polynomial (you can adjust this)
        degree = 2

        # Create a polynomial regression model
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

        # Fit the model
        model.fit(x_values.reshape(-1, 1), y_values)

        # Generate a range of x values for plotting
        # x_range = np.linspace(min(x_values), max(x_values), 10).reshape(-1, 1)


        # Get the coefficients of the fitted line
        intercept = model.named_steps['linearregression'].intercept_
        coefficients = model.named_steps['linearregression'].coef_
        print('       *** OUTPUT fit_curve slope, intercept',coefficients, intercept,'\n')
        y_pred = model.predict(x_values.reshape(-1, 1))

        return intercept,y_pred,x_values


    def find_unfit_points(self, x_values, y_values, fitted_curve):
        # Fit a curve using HuberRegressor
        intercept,y_pred,x_range = fitted_curve
        # Calculate the predicted y-values using the curve
        # predicted_y_values = slope * x_values + intercept
        self.least_fit_intercept = intercept
        # Calculate the residuals (the differences between predicted and actual y-values)
        residuals = np.round(y_values,2) - np.round(y_pred,2)
        print('\n\n\nResiduals: ',residuals)
        # Get all indeces where residual is higher than threshold*y_predict value
        print(vlv["threshold_y_fitting"])
        least_fit_indices =  np.where(np.abs(residuals) > vlv["threshold_y_fitting"])[0]
        print('least_fit_indices' ,least_fit_indices)

        # Create a list of points with the residuals higher than threshold
        self.least_fit_points = [[round(x_values[i],2), round(y_values[i],2)] for i in least_fit_indices]
        print('least_fit_points', self.least_fit_points, '\n\n\n')

        # print('LEAST FIT POINTS: ',self.least_fit_points)

        return self.least_fit_points, y_pred

    def generate_ranges_from_unfit_points(self,unfit_points,x_values):
 
        # Sort least-fit points based on x-values
        # unfit_points.sort(key=lambda point: point[0])
        
        # Calculate the interpoint interval
        # interpoint_interval = np.diff(x_values).mean()
        
        # Calculate the continuous ranges around least-fit points
        current_range = []
        unfit_point_x = [couple[0] for couple in unfit_points]

        listofranges = []
        for i,point in enumerate(x_values):
            # print('\nthis is i:',i)
            # print('this is listofranges:',listofranges,'\n')
            if np.round(point,2) not in np.round(unfit_point_x,2):
                if len(current_range)==0:
                    # print('this is len(current_range)==0')
                    continue
                else:
                    # print('\nthis is len(current_range)==0 else')
                    # close the range with point[-1]+threshold
                    interpoint_interval = point - x_values[i-1]
                    current_range.append(x_values[i-1]+vlv["threshold_x_interval"]*interpoint_interval)
                    listofranges.append(current_range)
                    current_range = []
            else:                    
                if len(current_range)==0 and 0<i<len(x_values)-1:
                    # print('\nthis is len(current_range)==0 and 0<i<len(x_values)')
                    interpoint_interval = point - x_values[i-1]
                    current_range.append(point-vlv["threshold_x_interval"]*interpoint_interval)
                elif len(current_range)==0 and i==0:
                    # print('\nthis is len(current_range)==0 and i==0')
                    current_range.append(point)
                elif len(current_range)==0 and i==len(x_values)-1:
                    # print('\nthis is len(current_range)==0 and i==len(x_values)')
                    interpoint_interval= point - x_values[i-1]
                    current_range.append(point-vlv["threshold_x_interval"]*interpoint_interval)
                    current_range.append(point)
                    listofranges.append(current_range)
                    current_range = []
                elif len(current_range)>0 and i==len(x_values)-1:
                    # print('\nthis is len(current_range)>0 and i==len(x_values)')
                    current_range.append(point)
                    listofranges.append(current_range)
                    current_range = []


        print('\n\n\n list of ranges: ',listofranges,'\n\n\n')
        return listofranges
        
    def local_exploration_validator_A(self,x_values, y_values, selected_range=0):
        print('       *** USING local_exploration_validator_A')
      
        fitted_curve = self.fit_curve(x_values, y_values, selected_range)
        least_fit_points,predicted_values = self.find_unfit_points(x_values, y_values,fitted_curve=fitted_curve)
        # least_fit_ranges = self.generate_ranges_from_unfit_points(least_fit_points,threshold=0.75 )        # unfit_points = self.find_least_fit_points(x_values, y_values, fitted_curve, threshold=threshold)
        unfitting_ranges = self.generate_ranges_from_unfit_points(least_fit_points,x_values)
        # print(unfitting_ranges)
        # self.update_num_points_evaluated([x_values,y_values], min=global_range[0], max=global_range[1])
        # self.save_to_text_file('output.txt', least_fit_points, unfitting_ranges,x_values)
        self.plot_curve(x_values, y_values, fitted_curve, unfitting_ranges,predicted_values)

        print('       *** OUTPUT unfitting_ranges',unfitting_ranges,'\n')
        return unfitting_ranges

    # def save_to_text_file(self,filename, least_fit_points, unfitting_ranges,x_values):
    #     # Modifier: Domain information: min max 
    #     #           increment unit
    #     #           number of points generated this cycle by modifier
    #     # Validator: Fit and unfit points (each itteration)
    #     #            unfit ranges
    #     with open(filename, 'w') as file:
    #         file.write("Modifier: Unfit Points (y,x):\n")
    #         for point in least_fit_points:
    #             file.write(f"{point[0]}, {point[1]}\n")
    #         file.write("\nModifier: MDV Information:\n")
    #         for val in mdv:
    #             file.write(f"{val}: {mdv[val]} \n")
    #         file.write("\nValidator:Totals:\n")
    #         file.write(f"Original mod points: {len(x_values)}\n")
    #         file.write(f"Unfit points: {len(least_fit_points)}\n")
            

    #         file.write("\nValidator:Unfit Ranges:\n")
    #         for start, end in unfitting_ranges:
    #             file.write(f"[{start},{end}]\n")
                

    def validator_controller(self, mod_x_list, sim_y_list, global_range=[mdv["domain_min_range"], mdv["domain_max_range"]],
                             local_validator=None, do_plot=False):
        print('       *** USING validator_controller')

        if local_validator is None:
            local_validator = self.local_exploration_validator_A  # Set default if not provided
        
        # TODO add to memory the new generated points in bad intervals
        
        # TODO check if unfit ranges exist and replace global range with loop of unfit ranges
        
        if np.any(self.least_fit_x_range): # if self.least_fit_x_range is not empty
            points = list(zip(mod_x_list, sim_y_list))
            points = [list(point) for point in points]
            points.extend(self.least_fit_points)
            points = sorted(points, key=lambda point: point[0])
            print("THIS IS PONITS ",points)

            validator_ranges=[]
            for each_range in self.least_fit_x_range: 
                #Calcualte bad points in each range
                print("THIS IS self.ranges ",self.least_fit_x_range)
                print("THIS IS RANGE ",each_range[0]," ",each_range[1])
                unfit_points = [(x, y) for x, y in points if each_range[0] <= x <= each_range[1]]
                if np.any(unfit_points):
                    unfit_x_values, unfit_y_values = zip(*unfit_points)
                    local_ranges = local_validator(unfit_x_values, unfit_y_values, selected_range=each_range)
                    validator_ranges.append(local_ranges)
            validator_ranges = [item for sublist in validator_ranges for item in sublist]
        else: 
            validator_ranges = local_validator(mod_x_list, sim_y_list, selected_range=global_range)
            self.least_fit_x_range = validator_ranges
        print('       *** OUTPUT validator_ranges', validator_ranges, '\n')
        return validator_ranges


    def plot_curve(self, x_values, y_values, fitted_curve, unfitting_ranges, predicted_values):  # Add self
        print('       *** USING plot_curve')
        plt.figure(figsize=(10, 6))

        plt.scatter(x_values, y_values, label='Original Data')
        plt.scatter(x_values, predicted_values, label='Predicted y Data', marker='x')

        plt.plot(fitted_curve[2], fitted_curve[1], color='red', label='Polynomial Regression')
        plt.plot(fitted_curve[2], fitted_curve[1] + vlv["threshold_y_fitting"], color='black', label='threshold ')
        plt.plot(fitted_curve[2], fitted_curve[1] - vlv["threshold_y_fitting"], color='black', label='threshold ')

        for start, end in unfitting_ranges:
            plt.axvspan(start, end, color='orange', alpha=0.3, label='Unfitting Range')

        plt.xlabel('X Values')
        plt.ylabel('Y Values')
        plt.title('Fitted Curve with Unfitting Ranges')
        plt.legend()
        plt.show()
        
### OLD CLASS BELOW: 
# # import global here: threshold

# class Validator:
#     def __init__(self): #i.e. if first iteration
#         self.iterations = 1
#         self.total_points = 0
#         self.total_bad_points = 0
#         self.range = (mdv["domain_min_range"],mdv["domain_max_range"])  # Initialize the range as a tuple
#         self.num_points_evaluated = 0  # Initialize num_points_evaluated


#     def fit_curve(x_values,y_values,global_range):
#         # Assuming you have a function to fit a curve to the data
#         # Replace the placeholder code below with your curve fitting logic

#         # standardize    
#         x_scaler, y_scaler = StandardScaler(), StandardScaler()
#         x_train = x_scaler.fit_transform(x_values)
#         y_train = y_scaler.fit_transform(y_values)

#         # fit model
#         model = HuberRegressor(epsilon=1)
#         model.fit(x_train, y_train.ravel())

#         # Fit curve to the data
#         fitted_curve = np.polyfit(x_values, y_values, 1)
#         return fitted_curve

#     def get_unfitting_point(x_values, y_values,fitted_curve,threshold = 0.9):
#         # extract y_value points at the mod_x_values, given that y_values are futher than threshold (euclidean distance)
#         # TODO: apply curve fit to x values, returns y_values
#         # TODO: if diff between y_values and true_y_values is greater than threshold, save point as unfit point
#         temp_soln = [1,4,5,6] 
#         unfit_points = [x_values[temp_soln],y_values[temp_soln]]

#         #returns all unfit y_value at mod_x points
#         return unfit_points

#     def generate_unfitting_ranges(x_values,sim_y_list,threshold=fitting_threshold):
#         # apply curve fit to new data
#         fitted_curve = Validator.fit_curve(x_values,sim_y_list)
#         # get points of unfit
#         unfit_points, fit_points = Validator.thresholding(fitted_curve,threshold)

#         # create ranges from continuous unfit points
#         if Validator.iterations == 0:
#             unfit_ranges = [[0,4],[40,50]]

#         Validator.collect_data(sim_y_list,x_values,unfit_ranges,unfit_points=unfit_points)
#         # return unfit ranges
#         Validator.collect_data()
#         return unfit_ranges

#     def update_history(self, new_total_points, new_range):
#         self.iterations += 1
#         self.total_points += new_total_points
#         self.range = new_range

#     def generate_report(self):
#         print('\nThe iteration is: ',self.iterations)
#         print('The range is: ', self.range)
#         print('The total point count is: ', self.total_points)

#         # # Print updated values
#         # print(f"Iterations: {Validator.iterations}")
#         # print(f"Total Points: {Validator.total_points}")
#         # print(f"Range: {Validator.range}")
#         # print(f"Num Points Evaluated: {Validator.num_points_evaluated}")

#     def update_num_points_evaluated(self, points, min_x, max_x):
#         self.num_points_evaluated = len([(x, y) for x, y in points if min_x <= x <= max_x])
    
#     def local_exploration_validator_A(x_values, y_values, global_range=[mdv["domain_min_range"],mdv["domain_max_range"]],threshold=fitting_threshold):
#         # generate fit function 
#         fitted_curve = Validator.fit_curve(x_values, y_values,global_range)
#         # find misfit points from mod_x_list and sim_y_list (outliers) using threshold from fit function
#         unfitting_points = Validator.get_unfitting_point(x_values, y_values,fitted_curve,threshold = threshold)
#         # generate ranges of misfit points (make it fancy threshold)
#         unfitting_ranges = Validator.generate_unfitting_ranges(unfitting_points)
#         ## update your history (for each iteration: 

#         # Update history with new values
#         Validator.update_history(5, 100, (10, 20))  # For example, 5 new iterations, 100 new total points, and new range
#         Validator.generate_report()
#         # Generate report

#         # Update num_points_evaluated (self, new_iterations, new_total_points, new_range)
#         Validator.update_num_points_evaluated(len(x_values), min=global_range[0], max=global_range[1])



#         # iteration, 
#         # total points evaluated (good and misfit), 
#         # points evaluated this iteration (good and misfit), 
#         # number of misfit ranges)

#         # num_points_input = len(mod_x_list)
#         # total_points.append(mod_x_list)

#         # # Get range from previous iteration range generation
#         # min_x,max_x = range_iteration 
#         # num_points_evaluated = [(x, y) for x, y in total_points if min_x <= x <= max_x]
#         return unfitting_ranges


#     def validator_controller(mod_x_list,sim_y_list, global_range=[mdv["domain_min_range"],mdv["domain_max_range"]],threshold=fitting_threshold, local_validator=local_exploration_validator_A, do_plot=False):
#         # gets points mod_x_list, sim_y_list
#         validator_ranges=Validator.local_exploration_validator_A(mod_x_list,sim_y_list, global_range=[mdv["domain_min_range"],mdv["domain_max_range"]],threshold=fitting_threshold)

#         # if not first time accessing validator, merge old points with new
#             # i.e. merge all data points together
#         return validator_ranges


#     def collect_data(self, sym, mod, ranges, unfit_points):
#         self.archive_sym.append(sym)
#         self.archive_mod.append(mod)
#         # track number of iterations
#         self.iterations = self.iterations+1
#         self.total_points = self.total_points+len(mod)
#         # track number of good/bad points
#         self.total_unfit_points = self.total_unfit_points+len(mod)
    
#     def update_statistics(self, new_sym, new_mod):
#         # track number of points generated from input to get_unfitting_ranges
#         self.collect_data(new_sym,new_mod)
#         # track number of unfit intervals i.e. append length of ranges each itteration)
#         self.history.extend(new_sym)
    

    
#     def thresholding(self, threshold):
#         # Assuming you want to return x values above the threshold
#         x_values = np.arange(len(self.history))
#         y_values = np.array(self.history)
#         above_threshold = x_values[y_values > threshold]
#         return above_threshold
    

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
