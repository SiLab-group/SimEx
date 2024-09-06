import os

import matplotlib.pyplot as plt
import numpy as np
from global_settings import Vfs, SimexSettings, timestamp
from logger_utils import Logger
from sklearn.metrics import mean_squared_error

logger = Logger()


class Validator:
    def __init__(self):
        self.unfit_intercept = None
        self.predicted_values = None
        self.fitted_curve = None
        self.unfit_interval = None

    def build_equation_string(self, coefficients: list):
        equation = 'y = '
        highest_degree = len(coefficients) - 1
        for idx, coeff in enumerate(coefficients):
            degree = highest_degree - idx
            sign = '+' if coeff >= 0 and idx != 0 else ''
            if degree == 0:
                equation += f'{sign} {str(coeff)}'
                break
            equation += f'{sign} {coeff}x^{degree} '
        return equation

    def fit_curve(self, x_values, y_values, max_deg=Vfs.max_deg, improvement_threshold=Vfs.improvement_threshold,
                  penality_weight=Vfs.penality_weight):
        x_values = np.array(x_values)  # Convert to numpy array
        y_values = np.array(y_values)  # Convert to numpy array
        mse = np.Infinity
        coeff = None
        intersect = None
        y_pred = None   

        degree = Vfs.degree
        is_early_stop = Vfs.early_stop

        while degree <= max_deg:
            current_coeff = np.polyfit(x_values, y_values, deg=degree)
            p = np.poly1d(current_coeff)
            current_intersect = current_coeff[-1]
            current_y_pred = p(x_values.reshape(-1, 1))
            # Add penality to MSE to avoid overfitting with high dimension polynomial
            current_mse = mean_squared_error(
                y_values, current_y_pred) + penality_weight * np.sum(current_coeff[:-1] ** 2)
            has_mse_improved: bool = current_mse <= mse
            is_acceptable_improvement: bool = (
                                                      mse - current_mse) >= improvement_threshold

            
            if is_early_stop:
                # If early stop flag is set to True and we do not have a sufficient improvement by increasing dimension, we stop
                if not has_mse_improved or not is_acceptable_improvement:
                    break
                mse = current_mse
                coeff = current_coeff
                intersect = current_intersect
                y_pred = current_y_pred
            elif has_mse_improved and is_acceptable_improvement:
                # If no early stop, we go trough all dimensions and we keep the best approximation
                    mse = current_mse
                    coeff = current_coeff
                    intersect = current_intersect
                    y_pred = current_y_pred
            
            degree += 1
        equation = self.build_equation_string(coeff)
        # print("\n\nCALLED FIT_CURVE")
        # print("Y_PRED"+str(y_pred.flatten()))
        # print("X_VALUES"+str(x_values))
        # print("EQUATION"+str(equation))

        return intersect, y_pred.flatten(), x_values, equation

    def find_unfit_points(self, x_values, y_values, fitted_curve):
        # Fit a curve using HuberRegressor
        intercept, y_pred, _, _ = fitted_curve
        # Calculate the predicted y-values using the curve
        # predicted_y_values = slope * x_values + intercept
        self.unfit_intercept = intercept
        # Calculate the residuals (the differences between predicted and actual y-values)
        residuals = np.round(y_values, 4) - np.round(y_pred, 4)
        # Get all indeces where residual is higher than threshold*y_predict value
        # print(vlv["threshold_y_fitting"])
        unfit_indices = np.where(
            np.abs(residuals) > Vfs.threshold_y_fitting)[0]
        # print('unfit_indices' ,unfit_indices)

        # Create a list of points with the residuals higher than threshold
        unfit_points = [[x_values[i], y_values[i]] for i in unfit_indices]
        # print('unfit_points', unfit_points, '\n\n\n')

        # print('LEAST FIT POINTS: ',unfit_points)

        return unfit_points, y_pred

    def generate_intervals_from_unfit_points(self, unfit_points, x_values):
        # IF all points are not fit, then keep the same interval as the bad interval because otherwise it will shrink.
        # i.e., interval 1-5 has all points 2, 3 and 4, unfit then the new interval is 2-4, which is incorrect.

        # Calculate the continuous intervals around least-fit points
        current_interval = []
        unfit_point_x = [couple[0] for couple in unfit_points]

        list_of_intervals = []
        for i, point in enumerate(x_values):
            # print('\nthis is i:',i)
            # print('this is list_of_intervals:',list_of_intervals,'\n')
            if np.round(point, 4) not in np.round(unfit_point_x, 4):
                if len(current_interval) == 0:
                    # print('this is len(current_interval)==0')
                    continue
                # print('\nthis is len(current_interval)==0 else')
                # close the interval with point[-1]+threshold
                interpoint_interval = point - x_values[i - 1]
                current_interval.append(
                    x_values[i - 1] + Vfs.threshold_x_interval * interpoint_interval)
                list_of_intervals.append(current_interval)
                current_interval = []
            else:
                if len(current_interval) == 0 and 0 < i < len(x_values) - 1:
                    # print('\nthis is len(current_interval)==0 and 0<i<len(x_values)')
                    interpoint_interval = point - x_values[i - 1]
                    current_interval.append(
                        point - Vfs.threshold_x_interval * interpoint_interval)
                elif len(current_interval) == 0 and i == 0:
                    # print('\nthis is len(current_interval)==0 and i==0')
                    current_interval.append(point)
                elif len(current_interval) == 0 and i == len(x_values) - 1:
                    # print('\nthis is len(current_interval)==0 and i==len(x_values)')
                    interpoint_interval = point - x_values[i - 1]
                    current_interval.append(
                        point - Vfs.threshold_x_interval * interpoint_interval)
                    current_interval.append(point)
                    list_of_intervals.append(current_interval)
                    current_interval = []
                elif len(current_interval) > 0 and i == len(x_values) - 1:
                    # print('\nthis is len(current_interval)>0 and i==len(x_values)')
                    current_interval.append(point)
                    list_of_intervals.append(current_interval)
                    current_interval = []

        # print('\n\n\n list of intervals: ',list_of_intervals,'\n\n\n')
        return list_of_intervals

    def find_fit_points(self, x_values_all, y_values_all, unfit_points, tolerance=1e-5):
        # Find the rest of the points
        rest_of_points = [(x, y) for x, y in zip(x_values_all, y_values_all) if all(
            abs(x - xp) > tolerance or abs(y - yp) > tolerance for xp, yp in unfit_points)]
        # print('LF... rest_of_points:      ',rest_of_points)
        # Convert the result to a list of lists
        rest_of_points_list = [list(point) for point in rest_of_points]
        return rest_of_points_list

    def get_fit_intervals(self, unfit_x_interval, domain_min_interval, domain_max_interval):
        # Convert a single interval to a list of intervals
        if not unfit_x_interval:
            return [[domain_min_interval, domain_max_interval]]

        if not isinstance(unfit_x_interval[0], list):
            unfit_x_interval = [unfit_x_interval]

        # Initialize fit_x_intervals with the gap between the minimum domain value and the start of the first interval
        fit_x_intervals = [[domain_min_interval, unfit_x_interval[0][0]]]
        print('       *** USING get_fit_intervals:  ', fit_x_intervals)

        # Iterate through the given intervals and fill the gaps
        for current_interval, next_interval in zip(unfit_x_interval, unfit_x_interval[1:]):
            gap_interval = [current_interval[1], next_interval[0]]
            fit_x_intervals.append(gap_interval)

        # Add the last interval if there is any gap to fill
        if unfit_x_interval[-1][1] < domain_max_interval:
            fit_x_intervals.append(
                [unfit_x_interval[-1][1], domain_max_interval])

        # Ensure fit_x_intervals are within the specified domain boundaries
        fit_x_intervals = [
            [max(interval_start, domain_min_interval),
             min(interval_end, domain_max_interval)]
            for interval_start, interval_end in fit_x_intervals
        ]

        return fit_x_intervals

    def local_exploration_validator_A(self, x_values, y_values, selected_interval=0):

        print('       *** USING local_exploration_validator_A')
        fitted_curve = self.fit_curve(x_values, y_values)
        equation = fitted_curve[3]
        unfit_points, predicted_values = self.find_unfit_points(
            x_values, y_values, fitted_curve=fitted_curve)
        unfit_interval = self.generate_intervals_from_unfit_points(
            unfit_points, x_values)
        # print('unfit_interval',unfit_interval)
        # print('unfit_points',unfit_points)
        # print('x_values',x_values)

        fit_points = self.find_fit_points(x_values, y_values, unfit_points)
        fit_interval = self.get_fit_intervals(
            unfit_interval, domain_min_interval=selected_interval[0], domain_max_interval=selected_interval[1])

        for _, interval in enumerate(fit_interval):
            # Round the interval values to 2 decimal places
            interval = [round(val, 5) for val in interval]

            # Filter fit_points for the current interval and round the points to 2 decimal places
            filtered_fit_points = [(round(point[0], 5), round(
                point[1], 5)) for point in fit_points if interval[0] <= point[0] <= interval[1]]

            logger_validator_arguments = {"log_contex": "fit_VAL_stats", "fit_interval": interval,
                                          "fitting_function": equation, "fit_points": filtered_fit_points}
            logger.log_validator(logger_validator_arguments)

        # print(unfit_interval)
        self.plot_curve(x_values, y_values, fitted_curve,
                        unfit_interval, predicted_values)

        # print('       *** OUTPUT unfit_interval',unfit_interval,'\n')
        self.fitted_curve = fitted_curve
        self.predicted_values = predicted_values
        return equation, unfit_points, unfit_interval, fit_points, fit_interval

    def plot_curve(self, x_values, y_values, fitted_curve, unfit_interval, predicted_values):  # Add self
        import datetime
        self.unfit_interval = unfit_interval
        plt.rcParams.update({'font.size': Vfs.font_size})

        plt.figure(figsize=(Vfs.figsize_x, Vfs.figsize_y))
        plt.scatter(x_values, y_values, label='Original Data')
        plt.scatter(x_values, predicted_values,
                    label='Predicted y Data', marker='x')

        plt.plot(fitted_curve[2], fitted_curve[1],
                 color='red', label='Polynomial Regression')
        plt.plot(fitted_curve[2], fitted_curve[1] +
                 Vfs.threshold_y_fitting, color='black', label='threshold ')
        plt.plot(fitted_curve[2], fitted_curve[1] -
                 Vfs.threshold_y_fitting, color='black', label='threshold ')
        count = 0
        for start, end in unfit_interval:
            count += 1
            plt.axvspan(start, end, color='orange',
                        alpha=0.3, label=f'Unfit Interval {count}: [{round(start)},{round(end)}]')

        plt.xlabel(Vfs.x_labels)
        plt.ylabel(Vfs.y_labels)
        plt.title(Vfs.title)
        plt.legend()
        plt.savefig(os.path.join(SimexSettings.results_dir, f"TTS_vs_Volume_{SimexSettings.instance_name}-{timestamp}.pdf"), format='pdf')
        plt.show()

    def get_curve_values(self):
        return self.fitted_curve, self.predicted_values, self.unfit_interval

