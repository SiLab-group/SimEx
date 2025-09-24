import numpy as np

from ..components import Validator


class ValidatorController:
    def __init__(self, logger, settings):
        self.unfit_x_interval = None
        self.unfit_points = []
        self.fit_x_interval = None
        self.fit_points = None
        self.equation = None
        self.predicted_values = None
        self.fitted_curve = None
        self.x_values = None
        self.y_values = None
        self.unfit_interval = None
        self.logger = logger
        self.settings = settings

    def validate(self, mod_x_list, sim_y_list, selected_validator, global_interval):
        validator = Validator(self.logger, self.settings)
        print("Validator...")

        if np.any(self.unfit_x_interval):  # if self.unfit_x_interval is not empty
            # Add all new points to old unfit points
            points = list(zip(mod_x_list, sim_y_list))
            points = [list(point) for point in points]
            points.extend(self.unfit_points)
            print("What are POINTS ", points)
            points = sorted(points, key=lambda point: point[0])

            validator_unfit_intervals = []
            validator_unfit_points = []

            # enter each interval couple
            for each_interval in self.unfit_x_interval:
                # Calcualte bad points in each interval
                # print("THIS IS EACH INTERVAL ",each_interval[0]," ",each_interval[1])

                # Select unfit points ONLY withing each_interval
                unfit_points = [
                    (x, y)
                    for x, y in points
                    if each_interval[0] <= x <= each_interval[1]
                ]
                # print("47 THIS IS unfit_points ",unfit_points)

                if len(unfit_points) < 2:
                    print("This is UNFIT POINTS ", unfit_points)
                    # validator_unfit_intervals.append(each_interval)
                    logger_validator_arguments = {
                        "log_contex": "internal VAL stats",
                        "new_unfit_interval": each_interval,
                        "unfit_points": unfit_points,
                    }
                    self.logger.log_validator(logger_validator_arguments)

                    validator_unfit_intervals.append([each_interval])
                    validator_unfit_points.append(unfit_points)
                else:
                    logger_validator_arguments = {
                        "log_contex": "internal VAL stats",
                        "new_unfit_interval": each_interval,
                        "unfit_points": unfit_points,
                    }
                    self.logger.log_validator(logger_validator_arguments)

                    unfit_x_values, unfit_y_values = zip(*unfit_points)
                    (
                        equation,
                        new_unfit_points,
                        new_unfit_interval,
                        _,
                        fit_interval,
                    ) = getattr(validator, selected_validator.__name__)(
                        unfit_x_values,
                        unfit_y_values,
                        selected_interval=each_interval,
                    )
                    validator_unfit_intervals.append(new_unfit_interval)
                    validator_unfit_points.append(new_unfit_points)
                    print(
                        "equation,\n",
                        equation,
                        "\nunfit_points\n",
                        unfit_points,
                        "\nlocal_unfit_interval\n,",
                        fit_interval,
                    )
            print(
                "DIFFERENCES \neach_interval ",
                each_interval,
                "\nnew_unfit_interval  ",
                new_unfit_interval,
            )

            validator_unfit_intervals = [
                item for sublist in validator_unfit_intervals for item in sublist
            ]
            validator_unfit_points = [
                item for sublist in validator_unfit_points for item in sublist
            ]
            self.unfit_x_interval = validator_unfit_intervals
            self.unfit_points = validator_unfit_points

        else:
            (
                equation,
                new_unfit_points,
                validator_unfit_intervals,
                fit_points,
                fit_interval,
            ) = getattr(validator, selected_validator.__name__)(
                x_values=mod_x_list,
                y_values=sim_y_list,
                selected_interval=global_interval,
            )
            # print('equation,fit_points,fit_interval\n',equation,'\n',fit_points,'\n\n',fit_interval)
            self.unfit_x_interval = validator_unfit_intervals
            self.unfit_points = new_unfit_points
            # Log the equation
        # print('       *** OUTPUT validator_intervals', validator_unfit_intervals, '\n')

        logger_validator_arguments = {
            "log_contex": "internal VAL stats",
            "validator_intervals": validator_unfit_intervals,
        }
        self.logger.log_validator(logger_validator_arguments)
        self.fitted_curve, self.predicted_values, self.unfit_interval = (
            validator.get_curve_values()
        )
        self.x_values = mod_x_list
        self.y_values = sim_y_list
        return validator_unfit_intervals
