"""Driver for executing PERFORM simulations.

Initializes all necessary constructs for executing a PERFORM simulation,
namely a SystemSolver, a SolutionDomain, a VisualizationGroup, and a RomDomain (if running a ROM simulation).
Advances through time steps, calls visualization and output routines, and handles solver blowup if it occurs.

After installing PERFORM, the terminal command "perform" will execute main() and take the first command line
argument as the working directory
"""

import os
from time import time
import argparse
import traceback
import warnings

from perform.system_solver import SystemSolver
from perform.solution.solution_domain import SolutionDomain
from perform.visualization.visualization_group import VisualizationGroup
from perform.rom.rom_domain import RomDomain

warnings.filterwarnings("error")
# dt, latent_dims, initbasis_snapIterEnd, adaptiveROMUpdateFreq, adaptiveROMWindowSize

def main():
    """Main driver function which initializes all necessary constructs and advances the solution in time"""

    # ----- Start setup -----

    # Read working directory input
    parser = argparse.ArgumentParser(description="Read working directory and FOM/ROM input parameters")
    parser.add_argument("working_dir", type=str, default="./", help="runtime working directory")
    # Read additional parameters for the FOM/ROM
    parser.add_argument("--calc_rom", type=int, default=None, help="calculate ROM")
    parser.add_argument("--dt", type=float, help="time step size", default=None)
    parser.add_argument("--nrsteps", type=int, help="number of time steps", default=None)
    parser.add_argument("--latent_dims", type=int, help="basis dimension", default=None)
    parser.add_argument("--init_window_size", type=int, help="initial window size", default=None)
    parser.add_argument("--adapt_window_size", type=int, help="adaptive window size", default=None)
    parser.add_argument("--adapt_update_freq", type=int, help="adaptive update frequency", default=None)
    parser.add_argument("--out_skip", type=int, help="skip interval in saving output", default=None)
    
    args = parser.parse_args()
    
    working_dir = os.path.expanduser(parser.parse_args().working_dir)
    assert os.path.isdir(working_dir), "Given working directory does not exist"

    # Retrieve global solver parameters
    # TODO: multi-domain solvers
    solver = SystemSolver(working_dir, args.dt, args.calc_rom, args.nrsteps, args.out_skip)

    # Initialize physical and ROM solutions
    sol_domain = SolutionDomain(solver)
    if solver.calc_rom:
        rom_domain = RomDomain(sol_domain, solver, args.latent_dims, args.init_window_size, args.adapt_window_size, args.adapt_update_freq)
    else:
        rom_domain = None

    # Initialize plots
    visGroup = VisualizationGroup(sol_domain, solver)
    
    # ----- End setup -----

    # ----- Start unsteady solution -----

    try:
        # Loop over time iterations
        time_start = time()
        for solver.iter in range(1, solver.num_steps + 1):

            # edit for adaptive basis. first run FOM to generate window
            # if rom_domain.adaptiveROM and solver.time_iter < rom_domain.adaptiveROMInitTime + 1:
            #         sol_domain.advance_iter(solver)
            #     else:
            #         if rom_domain.adaptiveROM and solver.time_iter == rom_domain.adaptiveROMInitTime + 1:
            #             rom_domain.compute_cellidx_hyper_reduc(sol_domain)
            
            # Advance one physical time step
            if solver.calc_rom:
                rom_domain.advance_iter(sol_domain, solver)
            else:
                sol_domain.advance_iter(solver)
            
            solver.time_iter += 1
            solver.sol_time += solver.dt

            # Write unsteady solution outputs
            sol_domain.write_iter_outputs(solver)

            # Check "steady" solve
            if solver.run_steady:
                break_flag = sol_domain.write_steady_outputs(solver)
                if break_flag:
                    break

            # Visualization
            visGroup.draw_plots(sol_domain, solver)

        runtime = time() - time_start
        print("Solve finished in %.8f seconds, writing to disk" % runtime)

    except RuntimeWarning:
        solver.solve_failed = True
        print(traceback.format_exc())
        print("Solve failed, dumping solution so far to disk")

    # ----- End unsteady solution -----

    # ----- Start post-processing -----

    if rom_domain == None:
        sol_domain.write_final_outputs(solver)
    else:
        sol_domain.write_final_outputs(solver, rom_domain.param_string)

    # ----- End post-processing -----


if __name__ == "__main__":
    try:
        main()
    except:
        print(traceback.format_exc())
        print("Execution failed")
