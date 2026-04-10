class VSLController:
    def __init__(self, name):
        self.name = name
        # self.Kv_gain = Kv_gain
        # self.density_now = density_now
        # self.density_previous = density_previous
        # self.VSL_before = VSL_before
        # self.vsl_cv = vsl_cv

    def P(self, Kv_gain, density_now, density_previous, VSL_before, vsl_cv, VSLmax, VSLmin):
        density_error = density_previous - density_now
        VSL = VSL_before + Kv_gain*density_error
        
        # must be truncated to the respective bound and used as VSL(k−1) for the next control
        # period to avoid the wind-up effect
        # bounds on final speed limit
        if VSL <= VSL_before-vsl_cv:
            new_VSL_Speed = VSL_before-vsl_cv
#             aktivan = 11;
        elif VSL >= VSL_before+vsl_cv:
            new_VSL_Speed = VSL_before+vsl_cv
#             aktivan = 12;
        else:
            new_VSL_Speed = VSL

        # speed limit check min max
        if new_VSL_Speed >= VSLmax:
            new_VSL_Speed = VSLmax
        elif new_VSL_Speed <= VSLmin:
            new_VSL_Speed = VSLmin
        
        # round VSL
        lst = [60,80,100,120]
        target = new_VSL_Speed
        res = min(enumerate(lst), key=lambda x: abs(target - x[1]))
        new_VSL_Speed = lst[res[0]]

        return new_VSL_Speed