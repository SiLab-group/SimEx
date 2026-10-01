import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — no display needed, no figures shown

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples', 'marl_vsl'))

from simex import Simex
from simex.components.simulator import Simulator
from simex.components.modifier import Modifier
from simex.components.validator import Validator
from simex.config.settings import SimexSettings, SumoVsl, get_path, load_sumo_config
from simex.controllers.simulator_controller import SimulatorController
from simex.controllers.modifier_controller import ModifierController
from simex.utils.logger import Logger
from marl_simulator import Simulator as MarlSimulator




@pytest.fixture
def settings(tmp_path):
    s = SimexSettings(instance_name='test')
    s.results_dir = str(tmp_path)
    return s


@pytest.fixture
def logger(settings):
    return Logger(
        filename=os.path.join(settings.results_dir, 'test.log'),
        simex_settings=settings
    )

# Modifier


class TestModifier:

    def test_rescale_basic(self):
        result = Modifier.rescale([0, 50, 100], 0, 1)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.5)
        assert result[2] == pytest.approx(1.0)

    def test_rescale_constant(self):
        result = Modifier.rescale([5, 5, 5], 0, 10)
        assert result == [0, 0, 0]

    def test_rescale_empty(self):
        assert Modifier.rescale([], 0, 1) == []

    def test_modifierA_output_length(self):
        x = [2500, 3000, 3500, 4000]
        result = Modifier.modifierA(x, new_min=2500, new_max=4000)
        assert len(result) == len(x)

    def test_modifierA_bounds(self):
        x = [2500, 3000, 3500, 4000]
        result = Modifier.modifierA(x, new_min=2500, new_max=4000)
        assert min(result) == pytest.approx(2500.0)
        assert max(result) == pytest.approx(4000.0)

    def test_modifierB_output_length(self):
        x = [2500, 3000, 3500, 4000]
        result = Modifier.modifierB(x, new_min=2500, new_max=4000)
        assert len(result) == len(x)

    def test_modifierC_output_length(self):
        x = [2500, 3000, 3500, 4000]
        result = Modifier.modifierC(x, new_min=2500, new_max=4000)
        assert len(result) == len(x)

# Simulator (built-in mock simulators)

class TestSimulator:

    def test_sim_func_a_returns_float(self):
        result = Simulator.sim_func_A(3.0)
        assert isinstance(result, float)

    def test_sim_func_b_returns_float(self):
        result = Simulator.sim_func_B(3.0)
        assert isinstance(result, float)

    def test_sim_func_c_returns_float(self):
        result = Simulator.sim_func_C(3.0)
        assert isinstance(result, float)



# SimulatorController
class TestSimulatorController:

    def test_simulate_returns_correct_length(self):
        mod_x = [[1.0, 2.0, 3.0]]
        flat_x, sim_y = SimulatorController.simulate(mod_x, Simulator.sim_func_A)
        assert len(flat_x) == 3
        assert len(sim_y) == 3

    def test_simulate_false_input(self):
        result = SimulatorController.simulate(False, Simulator.sim_func_A)
        assert result is False

    def test_simulate_parallel_returns_correct_length(self):
        mod_x = [[1.0, 2.0, 3.0]]
        flat_x, sim_y = SimulatorController.simulate_parallel(mod_x, Simulator.sim_func_A, workers=2)
        assert len(flat_x) == 3
        assert len(sim_y) == 3

    def test_simulate_parallel_false_input(self):
        result = SimulatorController.simulate_parallel(False, Simulator.sim_func_A, workers=2)
        assert result is False


# ModifierController

class TestModifierController:

    def test_control_returns_points(self, logger, settings):
        mc = ModifierController(logger, settings)
        result, intervals = mc.control(
            intervals_list=[[2500, 4000]],
            selected_modifier=Modifier.modifierA,
            do_plot=False
        )
        assert result is not False
        assert len(result) > 0

    def test_control_stops_when_granularity_exhausted(self, logger, settings):
        settings.modifier_data_point = 1
        settings.modifier_incremental_unit = 25
        mc = ModifierController(logger, settings)
        result, _ = mc.control(
            intervals_list=[[2500, 4000]],
            selected_modifier=Modifier.modifierA,
            do_plot=False
        )
        assert result is False


# Validator (unit tests - no plotting)

class TestValidator:

    def test_build_equation_string(self, logger, settings):
        v = Validator(logger, settings)
        eq = v.build_equation_string([1.0, 0.0, -1.0])
        assert 'y =' in eq

    def test_fit_curve_returns_values(self, logger, settings):
        v = Validator(logger, settings)
        x = np.linspace(2500, 4000, 20)
        y = x ** 2 * 0.001 + np.random.normal(0, 1, 20)
        intersect, y_pred, x_out, equation = v.fit_curve(x, y)
        assert len(y_pred) == len(x)
        assert 'y =' in equation

    def test_find_unfit_points(self, logger, settings):
        v = Validator(logger, settings)
        x = np.linspace(2500, 4000, 20)
        y = x ** 2 * 0.001
        fitted = v.fit_curve(x, y)
        unfit, y_pred = v.find_unfit_points(x, y, fitted)
        assert isinstance(unfit, list)

    def test_find_unfit_points_relative_threshold(self, logger, settings):
        v = Validator(logger, settings)
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([100.0, 1000.0, 10.0])
        # Fake fit: every point is off by 20 from its prediction
        fitted = (0.0, y - 20, x, 'y = 0')

        # Absolute only: |20| > 15, all points unfit
        settings.vfs_threshold_y_fitting = 15
        settings.vfs_threshold_y_relative = 0.0
        unfit, _ = v.find_unfit_points(x, y, fitted)
        assert len(unfit) == 3

        # Relative 5%: tolerance = max(15, 0.05 * |y_pred|) = [15, 49, 15], the point at x=2 becomes fit
        settings.vfs_threshold_y_relative = 0.05
        unfit, _ = v.find_unfit_points(x, y, fitted)
        assert [p[0] for p in unfit] == [1.0, 3.0]

    def test_get_fit_intervals_no_unfit(self, logger, settings):
        v = Validator(logger, settings)
        result = v.get_fit_intervals([], 2500, 4000)
        assert result == [[2500, 4000]]

    def test_get_fit_intervals_with_unfit(self, logger, settings):
        v = Validator(logger, settings)
        result = v.get_fit_intervals([[3000, 3500]], 2500, 4000)
        assert [2500, 3000] in result
        assert [3500, 4000] in result


# Validator regression (exact values)

class TestValidatorRegression:

    def test_fit_curve_recovers_quadratic(self, logger, settings):
        v = Validator(logger, settings)
        x = np.linspace(0, 10, 30)
        y = 2 * x ** 2 - 3 * x + 5
        intercept, y_pred, _, equation = v.fit_curve(x, y)
        coeffs = logger.get_coefficients({'fitting_function': equation})
        assert intercept == pytest.approx(5)
        assert coeffs[:3] == pytest.approx([5, -3, 2])
        assert y_pred == pytest.approx(y)

    def test_fit_curve_matches_polyfit(self, logger, settings):
        settings.vfs_max_deg = 2
        v = Validator(logger, settings)
        rng = np.random.default_rng(0)
        x = np.linspace(2500, 4000, 20)
        y = 0.001 * x ** 2 + rng.normal(0, 1, 20)
        _, y_pred, _, equation = v.fit_curve(x, y)
        coeffs = logger.get_coefficients({'fitting_function': equation})
        expected = np.polyfit(x, y, 2)[::-1]
        assert coeffs[:3] == pytest.approx(expected, rel=1e-5)
        assert y_pred == pytest.approx(np.polyval(np.polyfit(x, y, 2), x))

    def test_equation_roundtrip_matches_prediction(self, logger, settings):
        settings.vfs_max_deg = 3
        v = Validator(logger, settings)
        rng = np.random.default_rng(1)
        x = np.linspace(2500, 4000, 30)
        y = 1e-6 * x ** 3 - 0.005 * x ** 2 + rng.normal(0, 5, 30)
        _, y_pred, _, equation = v.fit_curve(x, y)
        coeffs = logger.get_coefficients({'fitting_function': equation})
        assert np.poly1d(coeffs[::-1])(x) == pytest.approx(y_pred, rel=1e-6)

    def test_fit_curve_caps_degree_with_few_points(self, logger, settings):
        settings.vfs_early_stop = False
        v = Validator(logger, settings)
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 4.0, 9.0])
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            _, y_pred, _, equation = v.fit_curve(x, y)
        assert 'x^3' not in equation
        assert len(y_pred) == 3

    def test_find_unfit_points_column_input(self, logger, settings):
        v = Validator(logger, settings)
        x = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([[10.0], [20.0], [100.0], [40.0]])
        fitted = (0.0, np.array([10.0, 20.0, 30.0, 40.0]), x.flatten(), 'y = 0')
        unfit, _ = v.find_unfit_points(x, y, fitted)
        assert unfit == [[3.0, 100.0]]

    def test_generate_intervals_middle_point(self, logger, settings):
        v = Validator(logger, settings)
        x = [0.0, 10.0, 20.0, 30.0, 40.0]
        result = v.generate_intervals_from_unfit_points([[20.0, 0.0]], x)
        assert result == [pytest.approx([12.0, 28.0])]

    def test_generate_intervals_first_and_last_point(self, logger, settings):
        v = Validator(logger, settings)
        x = [0.0, 10.0, 20.0, 30.0, 40.0]
        result = v.generate_intervals_from_unfit_points([[0.0, 0.0], [40.0, 0.0]], x)
        assert result == [pytest.approx([0.0, 8.0]), pytest.approx([32.0, 40.0])]

    def test_generate_intervals_single_point(self, logger, settings):
        v = Validator(logger, settings)
        assert v.generate_intervals_from_unfit_points([[5.0, 0.0]], [5.0]) == [[5.0, 5.0]]

    def test_get_fit_intervals_unfit_at_start(self, logger, settings):
        v = Validator(logger, settings)
        assert v.get_fit_intervals([[2500, 3000]], 2500, 4000) == [[3000, 4000]]

    def test_get_fit_intervals_all_unfit(self, logger, settings):
        v = Validator(logger, settings)
        assert v.get_fit_intervals([[2500, 4000]], 2500, 4000) == []


# Simulator regression

class TestSimulatorRegression:

    def test_sim_func_a_formula(self):
        np.random.seed(0)
        noise = np.random.normal(-10, 10)
        np.random.seed(0)
        assert Simulator.sim_func_A(3.0) == pytest.approx(3.0 ** 3 - 3.0 ** 2 + noise)

    def test_sim_func_c_formula(self):
        np.random.seed(0)
        noise = np.random.normal(-1, 1)
        np.random.seed(0)
        assert Simulator.sim_func_C(2.0) == pytest.approx(np.sin(4.0) + 2.0 * 2 / 3 + noise)


# SUMO config loader

class TestSumoConfig:

    def test_get_path_expands_and_adds_slash(self, tmp_path, monkeypatch):
        monkeypatch.setenv('HOME', str(tmp_path))
        config = tmp_path / 'sumo_config.ini'
        config.write_text(
            "[SUMO]\nMODEL_PATH = ~/model\nSUMO_PATH = ~/bin/sumo\n"
            "[MARL]\nMODEL_PATH = ~/marl_model\nRESULTS_PATH = ~/results\n"
        )
        paths = get_path(str(config))
        assert paths['model_path'] == f"{tmp_path}/model/"
        assert paths['sumo_path'] == f"{tmp_path}/bin/sumo"
        assert paths['marl_model_path'] == f"{tmp_path}/marl_model/"
        assert paths['marl_results_path'] == f"{tmp_path}/results/"
        assert paths['marl_end'] == '6000'
        assert paths['config_file'] == str(config)

    def test_get_path_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            get_path(str(tmp_path / 'missing.ini'))

    def test_get_path_env_var(self, tmp_path, monkeypatch):
        config = tmp_path / 'other.ini'
        config.write_text("[SUMO]\nSUMO_PATH = /opt/sumo\n")
        monkeypatch.setenv('SIMEX_SUMO_CONFIG', str(config))
        assert get_path()['sumo_path'] == '/opt/sumo'

    def test_load_sumo_config_updates_sumovsl(self, tmp_path, monkeypatch):
        for name in ['model_path', 'sumo_path', 'marl_root_path', 'marl_run', 'marl_end',
                     'marl_vsl', 'marl_model_path', 'marl_results_path']:
            monkeypatch.setattr(SumoVsl, name, getattr(SumoVsl, name))
        config = tmp_path / 'sumo_config.ini'
        config.write_text("[SUMO]\nSUMO_PATH = /opt/sumo\n[MARL]\nEND = 7000\n")
        load_sumo_config(str(config), verbose=False)
        assert SumoVsl.sumo_path == '/opt/sumo'
        assert SumoVsl.marl_end == 7000


# SimexSettings

class TestSimexSettings:

    def test_defaults(self):
        s = SimexSettings(instance_name='foo')
        assert s.domain_min_interval == 2500
        assert s.domain_max_interval == 4000
        assert s.modifier_incremental_unit == 25

    def test_post_init_names(self):
        s = SimexSettings(instance_name='mytest')
        assert 'mytest' in s.log_filename
        assert 'mytest' in s.csv_filename
        assert 'mytest' in s.vfs_title
        assert 'mytest' in s.ops_title

    def test_custom_domain(self):
        s = SimexSettings(instance_name='x', domain_min_interval=1000, domain_max_interval=5000)
        assert s.domain_min_interval == 1000
        assert s.domain_max_interval == 5000


# Full SimEx end-to-end (mock simulator without SUMO)

class TestSimexEndToEnd:

    def test_run_simex_produces_csv(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sim = Simex(instance_name='e2e_test', smoothen=False)
        result_file = sim.run_simex(
            simulator_function=Simulator.sim_func_A,
            modifier=Modifier.modifierA,
            validator=Validator.local_exploration_validator_A,
            parallel=False
        )
        assert os.path.exists(result_file)
        assert result_file.endswith('.csv')

    def test_run_simex_parallel_produces_csv(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sim = Simex(instance_name='e2e_parallel', smoothen=False)
        result_file = sim.run_simex(
            simulator_function=Simulator.sim_func_A,
            modifier=Modifier.modifierA,
            validator=Validator.local_exploration_validator_A,
            parallel=True
        )
        assert os.path.exists(result_file)


# SUMO-dependent tests (skipped in CI)

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_marl_vsl_dir = os.path.join(_root, 'examples', 'marl_vsl')

# Check that the sumo binary is actually available (settings reads sumo_config.ini from cwd at import
# time — if pytest runs from repo root the binary path will be empty even if the config file exists)
from simex.config.settings import SumoVsl
_has_sumo = bool(SumoVsl.sumo_path) and os.path.isfile(SumoVsl.sumo_path)
_sumo_reason = "requires SUMO — run pytest from examples/marl_vsl/ with sumo_config.ini present"

_mod_x_list = [
    [2600.0, 2681.5384615384614, 2766.153846153846, 2853.846153846154, 2944.6153846153848,
     3038.4615384615386, 3135.3846153846152, 3235.3846153846152, 3338.4615384615386,
     3444.6153846153848, 3553.846153846154, 3666.153846153846, 3781.5384615384614, 3900.0]
]


@pytest.mark.skipif(not _has_sumo, reason=_sumo_reason)
def test_simulator_marl_novsl():
    os.environ['MARL_MODEL_PATH'] = os.path.join(_root, 'marl_model_MD', '')
    mod_x, sim_y_list = SimulatorController.simulate(_mod_x_list, MarlSimulator.marl_novsl_simulator)
    assert sim_y_list == [
        355.41666666666646, 366.8333333333335, 379.9861111111111, 396.18055555555566,
        406.18055555555543, 424.8333333333332, 445.5972222222221, 471.97222222222223,
        552.0277777777778, 542.2222222222223, 639.8611111111112, 706.9583333333335,
        764.5555555555557, 797.1805555555555
    ]


@pytest.mark.skipif(not _has_sumo, reason=_sumo_reason)
def test_simulator_marl_novsl_parallel():
    os.environ['MARL_MODEL_PATH'] = os.path.join(_root, 'marl_model_MD', '')
    mod_x, sim_y_list = SimulatorController.simulate_parallel(
        _mod_x_list, MarlSimulator.marl_novsl_simulator, workers=14
    )
    assert sim_y_list == [
        355.41666666666646, 366.8333333333335, 379.9861111111111, 396.18055555555566,
        406.18055555555543, 424.8333333333332, 445.5972222222221, 471.97222222222223,
        552.0277777777778, 542.2222222222223, 639.8611111111112, 706.9583333333335,
        764.5555555555557, 797.1805555555555
    ]


@pytest.mark.skipif(not _has_sumo, reason=_sumo_reason)
def test_simulator_marl_vsl():
    os.environ['MARL_PATH_TRAIN'] = os.path.join(_root, 'marl_training_file', '')
    os.environ['MARL_MODEL_PATH'] = os.path.join(_root, 'marl_model_MD', '')
    mod_x, sim_y_list = SimulatorController.simulate(_mod_x_list, MarlSimulator.marl_vsl_simulator)
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
        _mod_x_list, MarlSimulator.marl_vsl_simulator, workers=14
    )
    assert sim_y_list == [
        357.1388888888888, 369.861111111111, 380.611111111111, 397.9583333333334,
        411.0694444444444, 432.7916666666666, 452.5000000000001, 470.00000000000006,
        539.0694444444447, 553.7916666666665, 615.7222222222224, 716.3333333333336,
        813.9027777777779, 806.6805555555557
    ]
