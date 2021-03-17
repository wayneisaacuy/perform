import numpy as np

from perform.constants import REAL_TYPE
from perform.higher_order_funcs import calc_cell_gradients


def calc_rhs(sol_domain, solver):
	"""
	Compute rhs function
	"""

	sol_int = sol_domain.sol_int
	sol_inlet = sol_domain.sol_inlet
	sol_outlet = sol_domain.sol_outlet
	sol_prim_full = sol_domain.sol_prim_full
	sol_cons_full = sol_domain.sol_cons_full
	direct_samp_idxs = sol_domain.direct_samp_idxs
	gas = sol_domain.gas_model

	# compute ghost cell state (if adjacent cell is sampled)
	# TODO: update this after higher-order contribution?
	# TODO: adapt pass to calc_boundary_state() depending on space scheme
	# TODO: assign more than just one ghost cell for higher-order schemes
	if (direct_samp_idxs[0] == 0):
		sol_inlet.calc_boundary_state(solver.sol_time, sol_domain.space_order,
										sol_prim=sol_int.sol_prim[:, :2],
										sol_cons=sol_int.sol_cons[:, :2])
	if (direct_samp_idxs[-1] == (sol_domain.mesh.num_cells - 1)):
		sol_outlet.calc_boundary_state(solver.sol_time, sol_domain.space_order,
										sol_prim=sol_int.sol_prim[:, -2:],
										sol_cons=sol_int.sol_cons[:, -2:])

	sol_domain.fill_sol_full()  # fill sol_prim_full and sol_cons_full

	# first-order approx at faces
	sol_left = sol_domain.sol_left
	sol_right = sol_domain.sol_right
	sol_left.sol_prim = sol_prim_full[:, sol_domain.flux_samp_left_idxs]
	sol_left.sol_cons = sol_cons_full[:, sol_domain.flux_samp_left_idxs]
	sol_right.sol_prim = sol_prim_full[:, sol_domain.flux_samp_right_idxs]
	sol_right.sol_cons = sol_cons_full[:, sol_domain.flux_samp_right_idxs]

	# add higher-order contribution
	if (sol_domain.space_order > 1):
		sol_prim_grad = calc_cell_gradients(sol_domain)
		sol_left.sol_prim[:, sol_domain.flux_left_extract] += \
			(sol_domain.mesh.dx / 2.0) * sol_prim_grad[:, sol_domain.grad_left_extract]
		sol_right.sol_prim[:, sol_domain.flux_right_extract] -= \
			(sol_domain.mesh.dx / 2.0) * sol_prim_grad[:, sol_domain.grad_right_extract]
		sol_left.calc_state_from_prim(calc_r=True, calc_cp=True)
		sol_right.calc_state_from_prim(calc_r=True, calc_cp=True)

	# compute fluxes
	flux = sol_domain.calc_flux()

	# compute rhs
	sol_domain.sol_int.rhs[:, direct_samp_idxs] = \
		flux[:, sol_domain.flux_rhs_idxs] - flux[:, sol_domain.flux_rhs_idxs + 1]
	sol_int.rhs[:, direct_samp_idxs] /= sol_domain.mesh.dx

	# compute source term
	if not solver.source_off:
		source, wf = sol_domain.reaction_model.calc_source(sol_domain.sol_int, solver.dt, direct_samp_idxs)
		sol_int.source[gas.mass_frac_slice[:, None], direct_samp_idxs[None, :]] = source
		sol_int.wf[:, direct_samp_idxs] = wf

		sol_int.rhs[3:, direct_samp_idxs] += sol_int.source[:, direct_samp_idxs]
