"""Test module for SimEx package."""

import pytest
import sys
import os
import tempfile
import shutil

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import SimEx components
from simex import Simex, Simulator, Modifier, Validator
from simex.core.settings import SimexSettings, timestamp
from simex.controllers.simulator_controller import SimulatorController
from simex.controllers.modifier_controller import ModifierController
from simex.utils.logger_utils import Logger


def test_basic_imports():
    """Test that basic imports work."""
    assert Simex is not None
    assert Simulator is not None
    assert Modifier is not None
    assert Validator is not None


def test_simex_settings():
    """Test SimexSettings creation."""
    settings = SimexSettings(instance_name="test")
    assert settings.instance_name == "test"
    assert hasattr(settings, 'domain_min_interval')
    assert hasattr(settings, 'domain_max_interval')


def test_simex_creation():
    """Test Simex object creation."""
    sim = Simex(instance_name='test', smoothen=False)
    assert sim.instance_name == 'test'
    assert sim.smoothen is False
    assert isinstance(sim.settings, SimexSettings)


def test_modifier():
    """Test modifier functionality."""
    # Create temporary settings and logger
    with tempfile.TemporaryDirectory() as temp_dir:
        settings = SimexSettings(instance_name="test")
        settings.results_dir = temp_dir
        
        logger = Logger(
            filename=os.path.join(temp_dir, "test.log"), 
            simex_settings=settings
        )
        
        modifier_controller = ModifierController(logger, settings)
        intervals_list = [[settings.domain_min_interval, settings.domain_max_interval]]
        
        result = modifier_controller.control(
            intervals_list=intervals_list, 
            selected_modifier=Modifier.modifierA,
            do_plot=False
        )
        
        assert result is not False
        assert len(result) == 2  # mod_x_list, intervals_list


def test_simulator():
    """Test simulator functionality."""
    # Test individual simulator functions
    test_x = 2500.0
    result_a = Simulator.sim_func_A(test_x)
    result_b = Simulator.sim_func_B(test_x)
    result_c = Simulator.sim_func_C(test_x)
    
    assert isinstance(result_a, float)
    assert isinstance(result_b, float)
    assert isinstance(result_c, float)


def test_simulator_controller():
    """Test simulator controller."""
    mod_x_list = [[2600.0, 2700.0, 2800.0]]
    
    # Test regular simulation
    mod_x, sim_y_list = SimulatorController.simulate(
        mod_x_list, Simulator.sim_func_A
    )
    
    assert len(mod_x) == len(sim_y_list) == 3
    assert all(isinstance(y, float) for y in sim_y_list)


def test_modifier_values():
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


def test_simulator_values():
    mod_x_list = [[2600.0, 2681.5384615384614, 2766.153846153846, 2853.846153846154, 2944.6153846153848,
                   3038.4615384615386, 3135.3846153846152, 3235.3846153846152, 3338.4615384615386, 3444.6153846153848,
                   3553.846153846154, 3666.153846153846, 3781.5384615384614, 3900.0]]
    mod_x, sim_y_list = SimulatorController.simulate(mod_x_list, Simulator.sim_func_C_no_noise)
    assert len(mod_x) == len(sim_y_list)
    assert sim_y_list == [1733.8407448206267, 1788.6915714961476, 1844.6524070340233, 1902.280968464465,
                          1964.0628908009996, 2025.5652217048346, 2090.725008592008, 2157.67808255228,
                          2224.8837026313304, 2297.0302510929305, 2368.328923165324, 2445.051851790707,
                          2520.776874248403, 2599.083679325016]




if __name__ == "__main__":
    # Run basic tests
    test_basic_imports()
    print("✓ Basic imports work")
    
    test_simex_settings()
    print("✓ SimexSettings works")
    
    test_simex_creation() 
    print("✓ Simex creation works")
    
    test_simulator()
    print("✓ Simulator functions work")
    
    test_simulator_controller()
    print("✓ SimulatorController works")

    test_simulator_values()
    print("✓ SimulatorController values works")

    test_simulator_parallel()
    print("✓ SimulatorController parallel works")

    test_modifier_values()
    print("✓ ModifierController works")
    
    print("\nAll tests passed!")
