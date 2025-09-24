#!/usr/bin/env python3
"""
Example script showing how to use the SimEx package.
"""

import time
from simex import Simex, Simulator, Modifier, Validator


def main():
    """Main function to run SimEx simulation."""
    before = time.time()

    # Run simex
    print("Running simex.")
    sim = Simex(instance_name="Func_A", smoothen=False)
    file = sim.run_simex(
        simulator_function=Simulator.sim_func_A,
        modifier=Modifier.modifierA,
        validator=Validator.local_exploration_validator_A,
    )

    print(f"Run finished. CSV file is {file}")
    now = time.time()
    print(f"Run time: {(now-before)/60} minutes")


def example_custom_settings():
    """Example showing custom settings usage."""
    from simex import SimexSettings

    # Custom settings
    settings = SimexSettings(
        instance_name="custom_example",
        domain_min_interval=1000,
        domain_max_interval=5000,
        modifier_incremental_unit=10,
        vfs_threshold_y_fitting=20,
        ops_sigmoid_tailing=True,
    )

    sim = Simex(instance_name="custom_example", smoothen=True)
    sim.settings = settings  # Override default settings

    file = sim.run_simex(
        simulator_function=Simulator.sim_func_B,
        modifier=Modifier.modifierB,
        validator=Validator.local_exploration_validator_A,
    )

    print(f"Custom run finished. CSV file is {file}")


def example_parallel_vs_sequential():
    """Example comparing parallel vs sequential execution."""
    import time

    print("Comparing parallel vs sequential execution...")

    # Sequential execution
    print("\n1. Running sequential simulation...")
    start = time.time()
    sim_seq = Simex(instance_name="sequential", smoothen=False)
    file_seq = sim_seq.run_simex(
        simulator_function=Simulator.sim_func_A,
        modifier=Modifier.modifierA,
        validator=Validator.local_exploration_validator_A,
        parallel=False,
    )
    seq_time = time.time() - start

    # Parallel execution
    print("\n2. Running parallel simulation...")
    start = time.time()
    sim_par = Simex(instance_name="parallel", smoothen=False)
    file_par = sim_par.run_simex(
        simulator_function=Simulator.sim_func_A,
        modifier=Modifier.modifierA,
        validator=Validator.local_exploration_validator_A,
        parallel=True,
    )
    par_time = time.time() - start

    print(f"\nResults:")
    print(f"Sequential: {seq_time:.2f}s - {file_seq}")
    print(f"Parallel:   {par_time:.2f}s - {file_par}")
    print(f"Speedup:    {seq_time/par_time:.2f}x")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "custom":
            example_custom_settings()
        elif sys.argv[1] == "compare":
            example_parallel_vs_sequential()
        else:
            print("Usage: python simex_run.py [custom|compare]")
    else:
        main()
