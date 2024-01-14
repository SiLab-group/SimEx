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
        self.least_fit_y_pred = y_pred
        self.least_fit_x_range = x_range 
        # Calculate the residuals (the differences between predicted and actual y-values)
        residuals = np.round(y_values,2) - np.round(self.least_fit_y_pred,2)
        print('\n\n\nResiduals: ',residuals)
        # Get all indeces where residual is higher than threshold*y_predict value
        print(vlv["threshold_y_fitting"])
        least_fit_indices =  np.where(np.abs(residuals) > vlv["threshold_y_fitting"])[0]
        print('least_fit_indices' ,least_fit_indices)

        # Create a list of points with the residuals higher than threshold
        self.least_fit_points = [[round(x_values[i],2), round(y_values[i],2)] for i in least_fit_indices]
        print('least_fit_points', self.least_fit_points, '\n\n\n')

        # print('LEAST FIT POINTS: ',self.least_fit_points)

        return self.least_fit_points, self.least_fit_y_pred

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
        
    def local_exploration_validator_A(self,x_values, y_values, global_range=[mdv["domain_min_range"], mdv["domain_max_range"]]):
        print('       *** USING local_exploration_validator_A')
        fitted_curve = Validator.fit_curve(x_values, y_values, global_range)
        least_fit_points,predicted_values = self.find_unfit_points(x_values, y_values,fitted_curve=fitted_curve)
        # least_fit_ranges = Validator.generate_ranges_from_unfit_points(least_fit_points,threshold=0.75 )        # unfit_points = Validator.find_least_fit_points(x_values, y_values, fitted_curve, threshold=threshold)
        unfitting_ranges = Validator.generate_ranges_from_unfit_points(least_fit_points,x_values)
        # print(unfitting_ranges)
        # Validator.update_num_points_evaluated([x_values,y_values], min=global_range[0], max=global_range[1])
        # Validator.save_to_text_file('output.txt', least_fit_points, unfitting_ranges,x_values)
        Validator.plot_curve(x_values, y_values, fitted_curve, unfitting_ranges,predicted_values)

        print('       *** OUTPUT unfitting_ranges',unfitting_ranges,'\n')
        return unfitting_ranges

    def validator_controller(self,mod_x_list, sim_y_list, global_range=[mdv["domain_min_range"], mdv["domain_max_range"]], local_validator=self.local_exploration_validator_A, do_plot=False):
        print('       *** USING validator_controller')

        validator_ranges = local_validator(mod_x_list, sim_y_list, global_range=[mdv["domain_min_range"], mdv["domain_max_range"]])
        print('       *** OUTPUT validator_ranges',validator_ranges,'\n')
        return validator_ranges


    def plot_curve(x_values, y_values, fitted_curve, unfitting_ranges,predicted_values):
        print('       *** USING plot_curve')
        plt.figure(figsize=(10, 6))

        # Plot the original x_values, y_values data
        plt.scatter(x_values, y_values, label='Original Data')
        plt.scatter(x_values, predicted_values, label='Predicted y Data',marker='x')

        # Plot the fitted curve
        # plt.plot(x_values, fitted_curve[0] * x_values + fitted_curve[1], color='red', label='Fitted Curve')
        
        plt.plot(fitted_curve[2], fitted_curve[1], color='red', label='Polynomial Regression')
        plt.plot(fitted_curve[2], fitted_curve[1]+vlv["threshold_y_fitting"], color='black', label='threshold ')
        plt.plot(fitted_curve[2], fitted_curve[1]-vlv["threshold_y_fitting"], color='black', label='threshold ')

        # Highlight the unfitting ranges
        for start, end in unfitting_ranges:
            plt.axvspan(start, end, color='orange', alpha=0.3, label='Unfitting Range')

        plt.xlabel('X Values')
        plt.ylabel('Y Values')
        plt.title('Fitted Curve with Unfitting Ranges')
        plt.legend()
        plt.show()
