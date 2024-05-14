
import os
import shutil
from simulator_controller import SimulatorController
from simulator import Simulator
from modifier_controller import ModifierController
from modifier import Modifier
from logger_utils import Logger
from global_settings import SimexSettings, timestamp


def test_modifier():
    sim = SimexSettings(instance_name="test", ops_sigmoid_tailing=True)
    resultdir = f"results_dir_{sim.instance_name}-{timestamp}"
    # Create directory for the results
    script_dir = os.path.abspath('')
    results_dir = os.path.join(script_dir, resultdir)
    sim.results_dir = results_dir

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
        print(f"Results dir {results_dir}")

    logger = Logger(filename=os.path.join(results_dir, f"{sim.log_filename}-{timestamp}.txt"), simex_settings=sim)
    modifier_controller = ModifierController(logger, settings=sim)
    intervals_list = [[sim.domain_min_interval, sim.domain_max_interval]]
    mod_outcome = modifier_controller.control(intervals_list=intervals_list, selected_modifier=Modifier.modifierA,
                                              do_plot=False)
    shutil.rmtree(results_dir)
    assert mod_outcome == ([[2600.0, 2681.5384615384614, 2766.153846153846, 2853.846153846154, 2944.6153846153848,
                             3038.4615384615386, 3135.3846153846152, 3235.3846153846152, 3338.4615384615386,
                             3444.6153846153848, 3553.846153846154, 3666.153846153846, 3781.5384615384614, 3900.0]],
                           [[2500, 4000]])


def test_simulator_parallel():
    mod_x_list = [[2600.0, 2681.5384615384614, 2766.153846153846, 2853.846153846154, 2944.6153846153848,
                   3038.4615384615386, 3135.3846153846152, 3235.3846153846152, 3338.4615384615386, 3444.6153846153848,
                   3553.846153846154, 3666.153846153846, 3781.5384615384614, 3900.0]]
    mod_x, sim_y_list = SimulatorController.simulate_parallel(mod_x_list, Simulator.sim_func_C_no_noise, workers=8)
    assert len(mod_x) == len(sim_y_list)
    assert sim_y_list == [1733.8407448206267, 1788.6915714961476, 1844.6524070340233, 1902.280968464465,
                          1964.0628908009996, 2025.5652217048346, 2090.725008592008, 2157.67808255228,
                          2224.8837026313304, 2297.0302510929305, 2368.328923165324, 2445.051851790707,
                          2520.776874248403, 2599.083679325016]


def test_simulator():
    mod_x_list = [[2600.0, 2681.5384615384614, 2766.153846153846, 2853.846153846154, 2944.6153846153848,
                   3038.4615384615386, 3135.3846153846152, 3235.3846153846152, 3338.4615384615386, 3444.6153846153848,
                   3553.846153846154, 3666.153846153846, 3781.5384615384614, 3900.0]]
    mod_x, sim_y_list = SimulatorController.simulate(mod_x_list, Simulator.sim_func_C_no_noise)
    assert len(mod_x) == len(sim_y_list)
    assert sim_y_list == [1733.8407448206267, 1788.6915714961476, 1844.6524070340233, 1902.280968464465,
                          1964.0628908009996, 2025.5652217048346, 2090.725008592008, 2157.67808255228,
                          2224.8837026313304, 2297.0302510929305, 2368.328923165324, 2445.051851790707,
                          2520.776874248403, 2599.083679325016]
