import os

import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples', 'marl_vsl'))

from simex.controllers.simulator_controller import SimulatorController
from marl_simulator import Simulator


mod_x_list = [
    [2600.0, 2681.5384615384614, 2766.153846153846, 2853.846153846154, 2944.6153846153848,
     3038.4615384615386, 3135.3846153846152, 3235.3846153846152, 3338.4615384615386,
     3444.6153846153848, 3553.846153846154, 3666.153846153846, 3781.5384615384614, 3900.0]
]


def test_simulator_marl_novsl():
    os.environ['MARL_MODEL_PATH'] = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'marl_model_MD', ''
    )
    mod_x, sim_y_list = SimulatorController.simulate(mod_x_list, Simulator.marl_novsl_simulator)
    assert sim_y_list == [
        355.41666666666646, 366.8333333333335, 379.9861111111111, 396.18055555555566,
        406.18055555555543, 424.8333333333332, 445.5972222222221, 471.97222222222223,
        552.0277777777778, 542.2222222222223, 639.8611111111112, 706.9583333333335,
        764.5555555555557, 797.1805555555555
    ]


def test_simulator_marl_novsl_parallel():
    os.environ['MARL_MODEL_PATH'] = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'marl_model_MD', ''
    )
    mod_x, sim_y_list = SimulatorController.simulate_parallel(
        mod_x_list, Simulator.marl_novsl_simulator, workers=14
    )
    assert sim_y_list == [
        355.41666666666646, 366.8333333333335, 379.9861111111111, 396.18055555555566,
        406.18055555555543, 424.8333333333332, 445.5972222222221, 471.97222222222223,
        552.0277777777778, 542.2222222222223, 639.8611111111112, 706.9583333333335,
        764.5555555555557, 797.1805555555555
    ]


def test_simulator_marl_vsl():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ['MARL_PATH_TRAIN'] = os.path.join(root, 'marl_training_file', '')
    os.environ['MARL_MODEL_PATH'] = os.path.join(root, 'marl_model_MD', '')
    mod_x, sim_y_list = SimulatorController.simulate(mod_x_list, Simulator.marl_vsl_simulator)
    assert sim_y_list == [
        354.94444444444434, 369.861111111111, 380.611111111111, 397.9583333333334,
        411.0694444444444, 432.7916666666666, 452.5000000000001, 470.00000000000006,
        539.0694444444447, 553.7916666666665, 667.1944444444443, 736.1111111111111,
        848.791666666667, 870.7916666666669
    ]


@pytest.mark.xfail
def test_simulator_marl_vsl_parallel():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ['MARL_PATH_TRAIN'] = os.path.join(root, 'marl_training_file', '')
    os.environ['MARL_MODEL_PATH'] = os.path.join(root, 'marl_model_MD', '')
    mod_x, sim_y_list = SimulatorController.simulate_parallel(
        mod_x_list, Simulator.marl_vsl_simulator, workers=14
    )
    assert sim_y_list == [
        357.1388888888888, 369.861111111111, 380.611111111111, 397.9583333333334,
        411.0694444444444, 432.7916666666666, 452.5000000000001, 470.00000000000006,
        539.0694444444447, 553.7916666666665, 615.7222222222224, 716.3333333333336,
        813.9027777777779, 806.6805555555557
    ]
