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
from Logger import Logger

logger = Logger() 

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



    def fit_curve(self, x_values, y_values, fit_type='polynomial', global_range=0):
       
        x_values = np.array(x_values)  # Convert to numpy array
        y_values = np.array(y_values)  # Convert to numpy array

        if fit_type == 'polynomial':

            '''using both Polynomial Features and LinearRegression to create a polynomial regression model. 
            The PolynomialFeatures is used to generate polynomial features, and then 
            LinearRegression is applied to these features.'''

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

            equation = f'y = {coefficients[0]:.2f}x^2 + {coefficients[1]:.2f}x + {intercept:.2f}'
            print('       *** OUTPUT fit_curve equation:', equation, '\n')

            y_pred = model.predict(x_values.reshape(-1, 1))

        elif fit_type == 'exponential':
            # Exponential fit
            model = LinearRegression()

            # Transform x_values to the logarithmic scale for exponential fit
            x_values_log = np.log(x_values)

            # Fit the model
            model.fit(x_values_log.reshape(-1, 1), y_values)

            # Get the coefficients of the fitted line
            intercept = model.intercept_
            slope = model.coef_[0]

            # Print the equation of the fitted curve
            equation = f'y = e^({intercept:.2f} + {slope:.2f} * log(x))'
            print('       *** OUTPUT fit_curve exponential equation:', equation, '\n')

            # Predict y values using the fitted model
            y_pred = np.exp(intercept + slope * np.log(x_values))

        elif fit_type == 'linear':
            # Linear fit
            model = LinearRegression()

            # Fit the model
            model.fit(x_values.reshape(-1, 1), y_values)

            # Get the coefficients of the fitted line
            intercept = model.intercept_
            slope = model.coef_[0]

            # Print the equation of the fitted line
            equation = f'y = {slope:.2f}x + {intercept:.2f}'
            print('       *** OUTPUT fit_curve linear equation:', equation, '\n')

            # Predict y values using the fitted model
            y_pred = model.predict(x_values.reshape(-1, 1))

        else:
            raise ValueError("Invalid fit_type. Use 'polynomial', 'exponential', or 'linear'.")

        return intercept, y_pred, x_values



    def find_unfit_points(self, x_values, y_values, fitted_curve):
        # Fit a curve using HuberRegressor
        intercept,y_pred,x_range = fitted_curve
        # Calculate the predicted y-values using the curve
        # predicted_y_values = slope * x_values + intercept
        self.least_fit_intercept = intercept
        # Calculate the residuals (the differences between predicted and actual y-values)
        residuals = np.round(y_values,4) - np.round(y_pred,4)
        print('\n\n\nResiduals: ',residuals)
        # Get all indeces where residual is higher than threshold*y_predict value
        print(vlv["threshold_y_fitting"])
        least_fit_indices =  np.where(np.abs(residuals) > vlv["threshold_y_fitting"])[0]
        print('least_fit_indices' ,least_fit_indices)

        # Create a list of points with the residuals higher than threshold
        self.least_fit_points = [[round(x_values[i],4), round(y_values[i],4)] for i in least_fit_indices]
        print('least_fit_points', self.least_fit_points, '\n\n\n')

        # print('LEAST FIT POINTS: ',self.least_fit_points)

        return self.least_fit_points, y_pred

    def generate_ranges_from_unfit_points(self,unfit_points,x_values):
 
        # Calculate the continuous ranges around least-fit points
        current_range = []
        unfit_point_x = [couple[0] for couple in unfit_points]

        listofranges = []
        for i,point in enumerate(x_values):
            # print('\nthis is i:',i)
            # print('this is listofranges:',listofranges,'\n')
            if np.round(point,4) not in np.round(unfit_point_x,4):
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
        # Fitted curve fit_type options include fit_type='polynomial', 'exponential', 'linear'
        fitted_curve = self.fit_curve(x_values, y_values, fit_type='polynomial',global_range=selected_range)
        least_fit_points,predicted_values = self.find_unfit_points(x_values, y_values,fitted_curve=fitted_curve)
        # least_fit_ranges = self.generate_ranges_from_unfit_points(least_fit_points,threshold=0.75 )        # unfit_points = self.find_least_fit_points(x_values, y_values, fitted_curve, threshold=threshold)
        unfitting_ranges = self.generate_ranges_from_unfit_points(least_fit_points,x_values)
        # print(unfitting_ranges)
        # self.update_num_points_evaluated([x_values,y_values], min=global_range[0], max=global_range[1])
        # self.save_to_text_file('output.txt', least_fit_points, unfitting_ranges,x_values)
        self.plot_curve(x_values, y_values, fitted_curve, unfitting_ranges,predicted_values)

        print('       *** OUTPUT unfitting_ranges',unfitting_ranges,'\n')
       
        
        return unfitting_ranges
                

    def validator_controller(self, mod_x_list, sim_y_list, global_range=[mdv["domain_min_range"], mdv["domain_max_range"]],
                             local_validator=None, do_plot=False):
        print('       *** USING validator_controller')
        if local_validator is None:
            local_validator = self.local_exploration_validator_A  # Set default if not provided
                        
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
                    local_unfit_range = local_validator(unfit_x_values, unfit_y_values, selected_range=each_range)
                    validator_ranges.append(local_unfit_range)
                    print("local_unfit_range ",local_unfit_range)
                    logger_validator_arguments = {}
                    logger_validator_arguments["log_contex"] = "internal VAL stats"
                    logger_validator_arguments["local_unfit_range"] = each_range
                    logger_validator_arguments["unfit_points"] = unfit_points
                    logger.log_validator(logger_validator_arguments)
            
            validator_ranges = [item for sublist in validator_ranges for item in sublist]
            self.least_fit_x_range = validator_ranges
        else: 
            validator_ranges = local_validator(mod_x_list, sim_y_list, selected_range=global_range)
            self.least_fit_x_range = validator_ranges
        print('       *** OUTPUT validator_ranges', validator_ranges, '\n')
        
        logger_validator_arguments = {}
        logger_validator_arguments["log_contex"] = "internal VAL stats"
        logger_validator_arguments["validator_ranges"] = validator_ranges
        logger.log_validator(logger_validator_arguments)
        
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
