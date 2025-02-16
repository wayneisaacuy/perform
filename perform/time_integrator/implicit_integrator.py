import numpy as np

import perform.constants as const
from perform.constants import REAL_TYPE
from perform.input_funcs import catch_input
from perform.time_integrator.time_integrator import TimeIntegrator

import copy

class ImplicitIntegrator(TimeIntegrator):
    """Base class for implicit time integrators.

    Solves implicit system via Newton's method.

    Each child class must implement a calc_residual() member function.

    Please refer to the solver theory documentation for details on each method.

    Args:
        param_dict: Dictionary of parameters read from the solver parameters input file.

    Attributes:
        time_type: Set to "implicit".
        subiter_max:
            Maximum number of subiterations to execute before terminating the iterative solve,
            regardless of convergence.
        res_tol:
            Residual norm tolerance, below which Newton's method for a physical time step is considered
            to be converged and subiteration is terminated.
        dual_time: Boolean flag indicating whether to use dual time-stepping.
        dtau: Dual time step size, in seconds.
        adapt_dtau: Boolean flag to indicate whether to adapt dtau, if dual_time == True.
        cfl: Courant–Friedrichs–Lewy number to adapt dtau based on maximum wave speed in each cell.
        vnn: von Neumann number to adapt dtau based on the mixture kinematic viscosity in each cell.
    """

    def __init__(self, param_dict, solver):

        super().__init__(param_dict, solver)

        self.time_type = "implicit"
        self.subiter_max = catch_input(param_dict, "subiter_max", const.SUBITER_MAX_IMP_DEFAULT)
        self.res_tol = catch_input(param_dict, "res_tol", const.L2_RES_TOL_DEFAULT)

        # Dual time-stepping, robustness controls
        self.dual_time = catch_input(param_dict, "dual_time", True)
        self.dtau = catch_input(param_dict, "dtau", const.DTAU_DEFAULT)
        if self.dual_time:
            self.adapt_dtau = catch_input(param_dict, "adapt_dtau", False)
        else:
            self.adapt_dtau = False
        self.cfl = catch_input(param_dict, "cfl", const.CFL_DEFAULT)
        self.vnn = catch_input(param_dict, "vnn", const.VNN_DEFAULT)


class BDF(ImplicitIntegrator):
    """Backwards differentiation formula.

    Supports up to fourth-order accuracy, though anything greater than second-order is generally not stable.

    Args:
        param_dict: Dictionary of parameters read from the solver parameters input file.

    Attributes:
        coeffs: List of NumPy arrays of the time derivative discretization coefficients for each order of accuracy.
    """

    def __init__(self, param_dict, solver):
        super().__init__(param_dict, solver)

        self.coeffs = [None] * 4
        self.coeffs[0] = np.array([1.0, -1.0], dtype=REAL_TYPE)
        self.coeffs[1] = np.array([1.5, -2.0, 0.5], dtype=REAL_TYPE)
        self.coeffs[2] = np.array([11.0 / 16.0, -3.0, 1.5, -1.0 / 3.0], dtype=REAL_TYPE)
        self.coeffs[3] = np.array([25.0 / 12.0, -4.0, 3.0, -4.0 / 3.0, 0.25], dtype=REAL_TYPE)
        assert self.time_order <= 4, (
            str(self.time_order) + "th-order accurate scheme not implemented for " + self.time_scheme + " scheme"
        )

    def calc_residual(self, sol_hist, rhs, solver, samp_idxs=np.s_[:]):
        """Compute fully-discrete residual.

        Args:
            sol_hist:
                List of NumPy arrays representing the recent conservative state solution history,
                as many as are required to compute the maximum order of accuracy requested.
            rhs: NumPy array of the semi-discrete governing ODE right-hand side function evaluation.
            solver: SystemSolver containing global simulation parameters.
            samp_idxs:
                Either a NumPy slice or NumPy array for selecting sampled cells to compute the residual at.
                Used for hyper-reduction of projection-based reduced-order models.

        Returns:
            NumPy array of the fully-discrete residual profiles.
        """

        # Account for cold start
        time_order = min(solver.iter, self.time_order)

        coeffs = self.coeffs[time_order - 1]

        # Compute time derivative component
        residual = coeffs[0] * sol_hist[0][:, samp_idxs]
        for iter_idx in range(1, time_order + 1):
            residual += coeffs[iter_idx] * sol_hist[iter_idx][:, samp_idxs]

        # Add RHS
        # NOTE: Negative convention here is for use with Newton's method
        residual = -(residual / self.dt) + rhs[:, samp_idxs]

        return residual

    def calc_fullydiscrhs(self, sol_domain, stateArg, solver, rom_domain, samp_idxs=np.s_[:]):
        """Compute fully discrete rhs
        
            stateArg is a column vector. also returns a column vector
        """
        
        assert self.time_order < 2, "BDF order has to be 1, backward Euler only"
        
        # make a deep copy of sol_domain
        
        copy_sol_domain = copy.deepcopy(sol_domain)
        
        # reshape stateArg
        
        stateArg_reshape = stateArg.reshape((sol_domain.gas_model.num_eqs, sol_domain.mesh.num_cells), order="C")
        
        # update sol_cons and sol_prim
        
        copy_sol_domain.sol_int.sol_cons = stateArg_reshape
        copy_sol_domain.sol_int.update_state(from_cons=True)
        
        # update deim indices
        copy_sol_domain.direct_samp_idxs = np.arange(0, sol_domain.mesh.num_cells)
        copy_sol_domain.num_samp_cells = len(copy_sol_domain.direct_samp_idxs)
        rom_domain.compute_cellidx_hyper_reduc(copy_sol_domain)
        
        # compute rhs 
        
        copy_sol_domain.calc_rhs(solver)
        
        rhs = copy_sol_domain.sol_int.rhs # shape will be num_eqs x num_cells
        
        # reshape rhs to be a column vector
        
        rhs = rhs.reshape((-1,1))
        
        # calculate semi-discrete rhs here
        
        # Account for cold start
        time_order = min(solver.iter, self.time_order)

        coeffs = self.coeffs[time_order - 1]
        
        # fullydiscrhs = coeffs[0] * stateArg[:, samp_idxs] - self.dt * rhs[:, samp_idxs]

        fullydiscrhs = coeffs[0] * stateArg[samp_idxs, :] - self.dt * rhs[samp_idxs, :]
        
        return fullydiscrhs
    
    def calc_fullydisc_residual(self, sol_hist, sol_domain, stateArg, solver, rom_domain, samp_idxs=np.s_[:]):
        
        assert self.time_order < 2, "BDF order has to be 1, backward Euler only"
        
        # make a deep copy of sol_domain
        
        copy_sol_domain = copy.deepcopy(sol_domain)
        
        # reshape stateArg
        
        stateArg_reshape = stateArg.reshape((sol_domain.gas_model.num_eqs, sol_domain.mesh.num_cells), order="C")
        
        # update sol_cons and sol_prim
        
        copy_sol_domain.sol_int.sol_cons = stateArg_reshape
        copy_sol_domain.sol_int.update_state(from_cons=True)
        
        # # update deim indices
        # copy_sol_domain.direct_samp_idxs = np.arange(0, sol_domain.mesh.num_cells)
        # copy_sol_domain.num_samp_cells = len(copy_sol_domain.direct_samp_idxs)
        # rom_domain.compute_cellidx_hyper_reduc(copy_sol_domain)
        
        # compute rhs 
        
        copy_sol_domain.calc_rhs(solver)
        
        rhs = copy_sol_domain.sol_int.rhs # shape will be num_eqs x num_cells
        
        time_order = min(solver.iter, self.time_order)

        coeffs = self.coeffs[time_order - 1]
        
        residual = coeffs[0] * stateArg_reshape[:, samp_idxs] + coeffs[1] * sol_hist[1][:, samp_idxs]
        
        residual = -(residual / self.dt) + rhs[:, samp_idxs]
        
        # reshape residual
        
        return residual