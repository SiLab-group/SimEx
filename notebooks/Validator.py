## VALIDATOR FILE: :
import numpy as np
import matplotlib.pyplot as plt
from global_settings import mds,vfs
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from Logger import Logger

logger = Logger() 

class Validator:
    def __init__(self):
        self.iterations = 1
        self.total_points = 0
        self.total_bad_points = 0
        self.interval = (mds["domain_min_interval"], mds["domain_max_interval"])
        self.num_points_evaluated = 0
        
        self.least_fit_intercept = None
        self.least_fit_y_pred = None
        self.least_fit_x_interval = None
        self.least_fit_points = None
        self.fit_x_interval = None
        self.fit_points = None
        self.equation = None    
    
    def build_equation_string(self, coefficients:list):
        equation = 'y = '
        highest_degree = len(coefficients) -1
        for idx, coeff in enumerate(coefficients):
            degree = highest_degree - idx
            sign = '+' if coeff >= 0 and idx !=0 else ''
            if degree == 0:
                equation += str(coeff)
                break
            equation += f'{sign} {coeff}x^{degree} '
        return equation
        
    def fit_curve(self, x_values, y_values, max_deg=vfs['max_deg'], improvement_threshold=vfs['improvement_threshold'], penality_weight=vfs['penality_weight']):
        x_values = np.array(x_values)  # Convert to numpy array
        y_values = np.array(y_values)  # Convert to numpy array
        mse = np.Infinity
        coeff = None
        intersect = None
        y_pred = None

        degree = vfs['degree']

        while degree <= max_deg:
            current_coeff = np.polyfit(x_values, y_values, deg=degree)
            p = np.poly1d(current_coeff)
            current_intersect = current_coeff[-1]
            current_y_pred = p(x_values.reshape(-1,1))
            # Add penality to MSE to avoid overfitting with high dimension polynomial
            current_mse = mean_squared_error(y_values, current_y_pred) + penality_weight * np.sum(current_coeff[:-1] ** 2)
            has_mse_improved: bool  = current_mse <= mse
            is_acceptable_improvement: bool = (mse - current_mse) >= improvement_threshold
           
            if not has_mse_improved or not is_acceptable_improvement:
                break

            mse = current_mse 
            coeff = current_coeff
            intersect = current_intersect
            y_pred = current_y_pred
            degree += 1
        equation = self.build_equation_string(coeff)
        print("\n\nCALLED FIT_CURVE")
        print("Y_PRED"+str(y_pred.flatten()))
        print("X_VALUES"+str(x_values))
        print("EQUATION"+str(equation))

        return intersect, y_pred.flatten(), x_values, equation



    def find_unfit_points(self, x_values, y_values, fitted_curve):
        # Fit a curve using HuberRegressor
        intercept,y_pred,x_interval,equation = fitted_curve
        # Calculate the predicted y-values using the curve
        # predicted_y_values = slope * x_values + intercept
        self.least_fit_intercept = intercept
        # Calculate the residuals (the differences between predicted and actual y-values)
        residuals = np.round(y_values,4) - np.round(y_pred,4)
        # Get all indeces where residual is higher than threshold*y_predict value
        # print(vlv["threshold_y_fitting"])
        least_fit_indices =  np.where(np.abs(residuals) > vfs["threshold_y_fitting"])[0]
        # print('least_fit_indices' ,least_fit_indices)

        # Create a list of points with the residuals higher than threshold
        self.least_fit_points = [[x_values[i], y_values[i]] for i in least_fit_indices]
        # print('least_fit_points', self.least_fit_points, '\n\n\n')

        print('LEAST FIT POINTS: ',self.least_fit_points)
        
        return self.least_fit_points, y_pred

    def generate_intervals_from_unfit_points(self,unfit_points,x_values):
 
        # Calculate the continuous intervals around least-fit points
        current_interval = []
        unfit_point_x = [couple[0] for couple in unfit_points]

        list_of_intervals = []
        for i,point in enumerate(x_values):
            # print('\nthis is i:',i)
            # print('this is list_of_intervals:',list_of_intervals,'\n')
            if np.round(point,4) not in np.round(unfit_point_x,4):
                if len(current_interval)==0:
                    # print('this is len(current_interval)==0')
                    continue
                else:
                    # print('\nthis is len(current_interval)==0 else')
                    # close the interval with point[-1]+threshold
                    interpoint_interval = point - x_values[i-1]
                    current_interval.append(x_values[i-1]+vfs["threshold_x_interval"]*interpoint_interval)
                    list_of_intervals.append(current_interval)
                    current_interval = []
            else:                    
                if len(current_interval)==0 and 0<i<len(x_values)-1:
                    # print('\nthis is len(current_interval)==0 and 0<i<len(x_values)')
                    interpoint_interval = point - x_values[i-1]
                    current_interval.append(point-vfs["threshold_x_interval"]*interpoint_interval)
                elif len(current_interval)==0 and i==0:
                    # print('\nthis is len(current_interval)==0 and i==0')
                    current_interval.append(point)
                elif len(current_interval)==0 and i==len(x_values)-1:
                    # print('\nthis is len(current_interval)==0 and i==len(x_values)')
                    interpoint_interval= point - x_values[i-1]
                    current_interval.append(point-vfs["threshold_x_interval"]*interpoint_interval)
                    current_interval.append(point)
                    list_of_intervals.append(current_interval)
                    current_interval = []
                elif len(current_interval)>0 and i==len(x_values)-1:
                    # print('\nthis is len(current_interval)>0 and i==len(x_values)')
                    current_interval.append(point)
                    list_of_intervals.append(current_interval)
                    current_interval = []


        # print('\n\n\n list of intervals: ',list_of_intervals,'\n\n\n')
        return list_of_intervals
    
    def find_fit_points(self, x_values_all, y_values_all, least_fit_points, tolerance=1e-5):
        # Find the rest of the points
        rest_of_points = [(x, y) for x, y in zip(x_values_all, y_values_all) if all(abs(x - xp) > tolerance or abs(y - yp) > tolerance for xp, yp in least_fit_points)]
        print('LF... rest_of_points:      ',rest_of_points)
        # Convert the result to a list of lists
        rest_of_points_list = [list(point) for point in rest_of_points]

        return rest_of_points_list

        

    # def get_fit_intervals(self,least_fit_x_interval, domain_min_interval, domain_max_interval):
    #     # Initialize fit_x_intervals with the gap between the minimum domain value and the start of the first interval
    #     fit_x_intervals = [[domain_min_interval, least_fit_x_interval[0][0]]]
    #     print('       *** USING get_fit_intervals:  ',fit_x_intervals)
    #     # Iterate through the given intervals and fill the gaps
    #     for current_interval, next_interval in zip(least_fit_x_interval, least_fit_x_interval[1:]):
    #         gap_interval = [current_interval[1], next_interval[0]]
    #         fit_x_intervals.append(gap_interval)

    #     # Add the last interval if there is any gap to fill
    #     if least_fit_x_interval[-1][1] < domain_max_interval:
    #         fit_x_intervals.append([least_fit_x_interval[-1][1], domain_max_interval])

    #     # Ensure fit_x_intervals are within the specified domain boundaries
    #     fit_x_intervals = [
    #         [max(interval_start, domain_min_interval), min(rinterval_end, domain_max_interval)]
    #         for interval_start, interval_end in fit_x_intervals
    #     ]

    #     return fit_x_intervals

    def get_fit_intervals(self, least_fit_x_interval, domain_min_interval, domain_max_interval):
        # Convert a single interval to a list of intervals
        if least_fit_x_interval==[]:
            return [[domain_min_interval,domain_max_interval]]

        if not isinstance(least_fit_x_interval[0], list):
            least_fit_x_interval = [least_fit_x_interval]

        # Initialize fit_x_intervals with the gap between the minimum domain value and the start of the first interval
        fit_x_intervals = [[domain_min_interval, least_fit_x_interval[0][0]]]
        print('       *** USING get_fit_intervals:  ', fit_x_intervals)

        # Iterate through the given intervals and fill the gaps
        for current_interval, next_interval in zip(least_fit_x_interval, least_fit_x_interval[1:]):
            gap_interval = [current_interval[1], next_interval[0]]
            fit_x_intervals.append(gap_interval)

        # Add the last interval if there is any gap to fill
        if least_fit_x_interval[-1][1] < domain_max_interval:
            fit_x_intervals.append([least_fit_x_interval[-1][1], domain_max_interval])

        # Ensure fit_x_intervals are within the specified domain boundaries
        fit_x_intervals = [
            [max(interval_start, domain_min_interval), min(interval_end, domain_max_interval)]
            for interval_start, interval_end in fit_x_intervals
        ]

        return fit_x_intervals



        
    def local_exploration_validator_A(self,x_values, y_values, selected_interval=0):
        
        print('       *** USING local_exploration_validator_A')
        # Fitted curve fit_type options include fit_type='polynomial', 'exponential', 'linear'
        #fitted_curve = self.fit_curve(x_values, y_values, fit_type='polynomial',global_interval=selected_interval)
        fitted_curve = self.fit_curve(x_values, y_values)
        equation = fitted_curve[3]
        least_fit_points,predicted_values = self.find_unfit_points(x_values, y_values,fitted_curve=fitted_curve)
        # least_fit_interval = self.generate_intervals_from_unfit_points(least_fit_points,threshold=0.75 )        # unfit_points = self.find_least_fit_points(x_values, y_values, fitted_curve, threshold=threshold)
        unfit_interval = self.generate_intervals_from_unfit_points(least_fit_points,x_values)
        print('unfit_interval',unfit_interval)
        print('least_fit_points',least_fit_points)
        
        fit_points = self.find_fit_points(x_values,y_values, least_fit_points)
        fit_interval = self.get_fit_intervals(unfit_interval, domain_min_interval =selected_interval[0], domain_max_interval=selected_interval[1])

        for i, interval in enumerate(fit_interval):
            # Round the interval values to 2 decimal places
            interval = [round(val, 2) for val in interval]

            # Filter fit_points for the current interval and round the points to 2 decimal places
            filtered_fit_points = [(round(point[0], 2), round(point[1], 2)) for point in fit_points if interval[0] <= point[0] <= interval[1]]

            logger_validator_arguments = {}
            logger_validator_arguments["log_contex"] = "fit_VAL_stats"
            logger_validator_arguments["fit_interval"] = interval
            logger_validator_arguments["fitting_function"] = equation
            logger_validator_arguments["fit_points"] = filtered_fit_points
            logger.log_validator(logger_validator_arguments)
        
        # print(unfit_interval)
        # self.update_num_points_evaluated([x_values,y_values], min=global_interval[0], max=global_interval[1])
        # self.save_to_text_file('output.txt', least_fit_points, unfit_interval,x_values)
        self.plot_curve(x_values, y_values, fitted_curve, unfit_interval,predicted_values)

        print('       *** OUTPUT unfit_interval',unfit_interval,'\n')

        return equation,least_fit_points,unfit_interval,fit_points,fit_interval
                

    def validator_controller(self, mod_x_list, sim_y_list, global_interval=[mds["domain_min_interval"], mds["domain_max_interval"]],
                             local_validator=None, do_plot=False):
        print('Validator...')
        if local_validator is None:
            local_validator = self.local_exploration_validator_A  # Set default if not provided
                        
        if np.any(self.least_fit_x_interval): # if self.least_fit_x_interval is not empty
            # Add all new points to oldl unfit points
            points = list(zip(mod_x_list, sim_y_list))
            points = [list(point) for point in points]
            points.extend(self.least_fit_points)
            points = sorted(points, key=lambda point: point[0])
            print("THIS IS POINTS ",points)

            validator_intervals=[]

            # enter each interval couple
            for each_interval in self.least_fit_x_interval:
                #Calcualte bad points in each interval
                print("\n\nTHIS IS self.least_fit_x_interval ",self.least_fit_x_interval)
                print("THIS IS EACH INTERVAL ",each_interval[0]," ",each_interval[1])

                #Select unfit points ONLY withing each_interval
                unfit_points = [(x, y) for x, y in points if each_interval[0] <= x <= each_interval[1]]
                if np.any(unfit_points):
                    unfit_x_values, unfit_y_values = zip(*unfit_points)
                    
                    #Return NEW unfit interval(s) withing each_interval
                    equation,least_fit_points,local_unfit_interval,fit_points,fit_interval = local_validator(unfit_x_values, unfit_y_values, selected_interval=each_interval)
                    validator_intervals.append(local_unfit_interval)
                    print('equation,least_fit_points,local_unfit_interval,fit_points,fit_interval\n',equation,'\n',fit_points,'\n',fit_interval)
                    # print("local_unfit_interval ",local_unfit_interval)
                    logger_validator_arguments = {}
                    logger_validator_arguments["log_contex"] = "internal VAL stats"
                    logger_validator_arguments["local_unfit_interval"] = each_interval
                    logger_validator_arguments["unfit_points"] = unfit_points
                    logger.log_validator(logger_validator_arguments)

            validator_intervals = [item for sublist in validator_intervals for item in sublist]
            self.least_fit_x_interval = validator_intervals

        else: 
            #TODO: whi is least_fit_points not used?
            equation,least_fit_points,validator_intervals,fit_points,fit_interval  = local_validator(mod_x_list, sim_y_list, selected_interval=global_interval)
            print('equation,fit_points,fit_interval\n',equation,'\n',fit_points,'\n\n',fit_interval)

            self.least_fit_x_interval = validator_intervals
            # self.fit_x_intervals = self.get_fit_intervals(validator_intervals, domain_min_interval =mdv['domain_min_interval'], domain_max_interval=mdv['domain_max_interval']) #global minus self.least_fit_x_interval     
            # Log the equation
        print('       *** OUTPUT validator_intervals', validator_intervals, '\n')
        
        logger_validator_arguments = {}
        logger_validator_arguments["log_contex"] = "internal VAL stats"
        logger_validator_arguments["validator_intervals"] = validator_intervals
        logger.log_validator(logger_validator_arguments)
        
        return validator_intervals


    def plot_curve(self, x_values, y_values, fitted_curve, unfit_interval, predicted_values):  # Add self
        print('       *** USING plot_curve')
        plt.figure(figsize=(10, 6))

        plt.scatter(x_values, y_values, label='Original Data')
        plt.scatter(x_values, predicted_values, label='Predicted y Data', marker='x')

        plt.plot(fitted_curve[2], fitted_curve[1], color='red', label='Polynomial Regression')
        plt.plot(fitted_curve[2], fitted_curve[1] + vfs["threshold_y_fitting"], color='black', label='threshold ')
        plt.plot(fitted_curve[2], fitted_curve[1] - vfs["threshold_y_fitting"], color='black', label='threshold ')

        for start, end in unfit_interval:
            plt.axvspan(start, end, color='orange', alpha=0.3, label='unfit Interval')

        plt.xlabel('X Values')
        plt.ylabel('Y Values')
        plt.title('Fitted Curve with unfit Intervals')
        plt.legend()
        plt.show()
