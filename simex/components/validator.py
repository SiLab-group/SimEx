import os

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from simex.config.settings import timestamp
from simex.utils.logger import Logger
from sklearn.metrics import mean_squared_error


class Validator:
    def __init__(self, logger, settings):
        self.unfit_intercept = None
        self.predicted_values = None
        self.fitted_curve = None
        self.unfit_interval = None
        self.logger = logger
        self.settings = settings
        self.figure_count = 1

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

    def fit_curve(self, x_values, y_values):
        max_deg = self.settings.vfs_max_deg
        improvement_threshold = self.settings.vfs_improvement_threshold
        penality_weight = self.settings.vfs_penality_weight
        x_values = np.asarray(x_values, dtype=float).flatten()
        y_values = np.asarray(y_values, dtype=float).flatten()
        if len(x_values) == 0:
            raise ValueError("fit_curve needs at least one point")
        # A polynomial of degree >= number of points is underdetermined (it interpolates, MSE = 0)
        max_deg = min(max_deg, len(np.unique(x_values)) - 1)
        degree = min(self.settings.vfs_degree, max_deg)
        is_early_stop = self.settings.vfs_early_stop

        best_mse = np.inf
        coeff = None
        intersect = None
        y_pred = None

        while degree <= max_deg:
            # Minimize least-square error on scaled domain
            p_fitted = Polynomial.fit(x_values, y_values, deg=degree)
            # Original coeff highest degree first for by build_equation_string and Logger.get_coefficients
            current_coeff = p_fitted.convert().coef[::-1]
            current_intersect = current_coeff[-1]
            current_y_pred = p_fitted(x_values)
            # Add penality to MSE to avoid overfitting with high dimension polynomial (intercept excluded)
            current_mse = mean_squared_error(
                y_values, current_y_pred) + penality_weight * np.sum(current_coeff[:-1] ** 2)
            has_mse_improved = current_mse < best_mse
            is_acceptable_improvement = np.isinf(best_mse) or (best_mse - current_mse) >= improvement_threshold

            if has_mse_improved and is_acceptable_improvement:
                best_mse = current_mse
                coeff = current_coeff
                intersect = current_intersect
                y_pred = current_y_pred
            elif is_early_stop:
                # Not a sufficient improvement by increasing dimension, we stop
                break

            degree += 1
        equation = self.build_equation_string(coeff)

        return intersect, y_pred, x_values, equation

    def y_tolerance(self, y_pred):
        # Allowed |residual| per point: max(absolute threshold, relative threshold * |y_pred|)
        # getattr keeps settings objects created before vfs_threshold_y_relative existed working
        relative = getattr(self.settings, 'vfs_threshold_y_relative', 0.0)
        return np.maximum(self.settings.vfs_threshold_y_fitting, relative * np.abs(y_pred))

    def find_unfit_points(self, x_values, y_values, fitted_curve):
        # Use the polynomial predictions computed by fit_curve
        intercept, y_pred, _, _ = fitted_curve
        self.unfit_intercept = intercept
        # Flatten so an (n, 1) column does not broadcast against y_pred (n,) into an (n, n) matrix
        x_values = np.asarray(x_values, dtype=float).flatten()
        y_values = np.asarray(y_values, dtype=float).flatten()
        # Calculate the residuals (the differences between actual and predicted y-values)
        residuals = y_values - y_pred
        # Get all indices where the absolute residual is higher than the (absolute or relative) threshold
        unfit_indices = np.where(np.abs(residuals) > self.y_tolerance(y_pred))[0]

        # Create a list of points with the residuals higher than threshold
        unfit_points = [[x_values[i], y_values[i]] for i in unfit_indices]

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
                    x_values[i - 1] + self.settings.vfs_threshold_x_interval * interpoint_interval)
                list_of_intervals.append(current_interval)
                current_interval = []
            else:
                if len(current_interval) == 0 and 0 < i < len(x_values) - 1:
                    # print('\nthis is len(current_interval)==0 and 0<i<len(x_values)')
                    interpoint_interval = point - x_values[i - 1]
                    current_interval.append(
                        point - self.settings.vfs_threshold_x_interval * interpoint_interval)
                elif len(current_interval) == 0 and i == 0:
                    # print('\nthis is len(current_interval)==0 and i==0')
                    current_interval.append(point)
                    if len(x_values) == 1:
                        # Single unfit point: close the interval, otherwise it is lost
                        current_interval.append(point)
                        list_of_intervals.append(current_interval)
                        current_interval = []
                elif len(current_interval) == 0 and i == len(x_values) - 1:
                    # print('\nthis is len(current_interval)==0 and i==len(x_values)')
                    interpoint_interval = point - x_values[i - 1]
                    current_interval.append(
                        point - self.settings.vfs_threshold_x_interval * interpoint_interval)
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

        # Gap before the first unfit interval, if any
        fit_x_intervals = []
        if unfit_x_interval[0][0] > domain_min_interval:
            fit_x_intervals.append([domain_min_interval, unfit_x_interval[0][0]])

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

    def local_exploration_validator_A(self, x_values, y_values, selected_interval=None):

        print('       *** USING local_exploration_validator_A')
        if selected_interval is None:
            # Default to the range covered by the data
            selected_interval = [min(x_values), max(x_values)]
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
            self.logger.log_validator(logger_validator_arguments)

        # print(unfit_interval)
        self.plot_curve(x_values, y_values, fitted_curve,
                        unfit_interval, predicted_values)

        # print('       *** OUTPUT unfit_interval',unfit_interval,'\n')
        self.fitted_curve = fitted_curve
        self.predicted_values = predicted_values
        return equation, unfit_points, unfit_interval, fit_points, fit_interval

    def plot_curve(self, x_values, y_values, fitted_curve, unfit_interval, predicted_values):  # Add self
        self.unfit_interval = unfit_interval
        plt.rcParams.update({'font.size': self.settings.vfs_font_size})

        plt.figure(figsize=(self.settings.vfs_figsize_x, self.settings.vfs_figsize_y))
        plt.scatter(x_values, y_values, label='Original Data')
        plt.scatter(x_values, predicted_values,
                    label='Predicted y Data', marker='x')

        plt.plot(fitted_curve[2], fitted_curve[1],
                 color='red', label='Polynomial Regression')
        tolerance = self.y_tolerance(fitted_curve[1])
        plt.plot(fitted_curve[2], fitted_curve[1] + tolerance,
                 color='black', label='threshold ')
        plt.plot(fitted_curve[2], fitted_curve[1] - tolerance,
                 color='black', label='threshold ')
        count = 0
        for start, end in unfit_interval:
            count += 1
            plt.axvspan(start, end, color='orange',
                        alpha=0.3, label=f'Unfit Interval {count}: [{round(start)},{round(end)}]')

        plt.xlabel(self.settings.vfs_x_labels)
        plt.ylabel(self.settings.vfs_y_labels)
        plt.title(self.settings.vfs_title)
        plt.legend()
        plt.savefig(os.path.join(self.settings.results_dir, f"TTS_vs_Volume_{self.settings.instance_name}-{timestamp}-{self.figure_count}.pdf"), format='pdf')
        self.figure_count = self.figure_count + 1
        plt.show()
        # Release the figure
        plt.close()

    def get_curve_values(self):
        return self.fitted_curve, self.predicted_values, self.unfit_interval

