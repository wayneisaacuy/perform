import os

import numpy as np

from perform.constants import REAL_TYPE, R_UNIV
from perform.higherOrderFuncs import calc_cell_gradients


def calc_rhs(sol_domain, solver):
	"""
	Compute rhs function
	"""

	sol_int    = sol_domain.sol_int 
	sol_inlet  = sol_domain.sol_inlet 
	sol_outlet = sol_domain.sol_outlet

	# compute ghost cell state (if adjacent cell is sampled)
	# TODO: update this after higher-order contribution?
	# TODO: need to adapt sol_prim and sol_cons pass to calc_bound_state() depending on space scheme
	# TODO: will have to assign more than just one ghost cell for higher-order schemes
	if (sol_domain.direct_samp_idxs[0] == 0):
		sol_domain.sol_inlet.calc_bound_state(solver, sol_prim=sol_int.sol_prim[:,:2], sol_cons=sol_int.sol_cons[:,:2])
	if (sol_domain.direct_samp_idxs[-1] == (solver.mesh.num_cells-1)):
		sol_domain.sol_outlet.calc_bound_state(solver, sol_prim=sol_int.sol_prim[:,-2:], sol_cons=sol_int.sol_cons[:,-2:])

	sol_domain.fillSolFull() # fill sol_prim_full and sol_cons_full

	# first-order approx at faces
	sol_left  = sol_domain.sol_left
	sol_right = sol_domain.sol_right
	sol_left.sol_prim  = sol_domain.sol_prim_full[:, sol_domain.flux_samp_left_idxs]
	sol_left.sol_cons  = sol_domain.sol_cons_full[:, sol_domain.flux_samp_left_idxs]
	sol_right.sol_prim = sol_domain.sol_prim_full[:, sol_domain.flux_samp_right_idxs]
	sol_right.sol_cons = sol_domain.sol_cons_full[:, sol_domain.flux_samp_right_idxs]

	# add higher-order contribution
	if (solver.space_order > 1):
		sol_prim_grad = calc_cell_gradients(sol_domain, solver)
		sol_left.sol_prim[:,sol_domain.flux_left_extract]   += (solver.mesh.dx / 2.0) * sol_prim_grad[:, sol_domain.grad_left_extract]
		sol_right.sol_prim[:,sol_domain.flux_right_extract] -= (solver.mesh.dx / 2.0) * sol_prim_grad[:, sol_domain.grad_right_extract]
		sol_left.calc_state_from_prim(calc_r=True, calc_cp=True)
		sol_right.calc_state_from_prim(calc_r=True, calc_cp=True)

	# compute fluxes
	flux = calc_inv_flux(sol_domain, solver)
	if (solver.visc_scheme > 0):
		visc_flux = calc_visc_flux(sol_domain, solver)
		flux -= visc_flux	

	# compute rhs
	sol_domain.sol_int.rhs[:, sol_domain.direct_samp_idxs] = flux[:, sol_domain.flux_rhs_idxs] - flux[:, sol_domain.flux_rhs_idxs+1]
	sol_int.rhs[:, sol_domain.direct_samp_idxs] /= solver.mesh.dx

	# compute source term
	if solver.source_on:
		calc_source(sol_domain, solver)
		sol_int.rhs[3:, sol_domain.direct_samp_idxs] += sol_int.source[:, sol_domain.direct_samp_idxs]


def calc_inv_flux(sol_domain, solver):
	"""
	Compute inviscid fluxes
	"""

	# TODO: generalize to other flux schemes, expand beyond Roe flux
	# TODO: entropy fix

	sol_prim_left  = sol_domain.sol_left.sol_prim
	sol_cons_left  = sol_domain.sol_left.sol_cons
	sol_prim_right = sol_domain.sol_right.sol_prim
	sol_cons_right = sol_domain.sol_right.sol_cons

	# inviscid flux vector
	flux_left = np.zeros(sol_prim_left.shape, dtype=REAL_TYPE)
	flux_right = np.zeros(sol_prim_right.shape, dtype=REAL_TYPE)     

	# compute sqrhol, sqrhor, fac, and fac1
	sqrhol = np.sqrt(sol_cons_left[0,:])
	sqrhor = np.sqrt(sol_cons_right[0,:])
	fac = sqrhol / (sqrhol + sqrhor)
	fac1 = 1.0 - fac

	# Roe average stagnation enthalpy and density
	sol_domain.sol_left.h0 = sol_domain.gas_model.calc_stag_enth(sol_prim_left)
	sol_domain.sol_right.h0 = sol_domain.gas_model.calc_stag_enth(sol_prim_right)

	sol_ave = sol_domain.sol_ave
	sol_ave.h0 = fac * sol_domain.sol_left.h0 + fac1 * sol_domain.sol_right.h0 
	sol_ave.sol_cons[0,:] = sqrhol * sqrhor

	# compute Roe average primitive state, adjust iteratively to conform to Roe average density and enthalpy
	sol_ave.sol_prim = fac[None,:] * sol_prim_left + fac1[None,:] * sol_prim_right
	sol_ave.calc_state_from_rho_h0()

	# compute Roe average state at faces, associated fluid properties
	sol_ave.calc_state_from_prim(calc_r=True, calc_cp=True)
	sol_ave.gamma_mix = sol_domain.gas_model.calc_mix_gamma(sol_ave.r_mix, sol_ave.cp_mix)
	sol_ave.c = sol_domain.gas_model.calc_sound_speed(sol_ave.sol_prim[2,:], r_mix=sol_ave.r_mix, gamma_mix=sol_ave.gamma_mix,
											 mass_fracs=sol_ave.sol_prim[3:,:], cp_mix=sol_ave.cp_mix)

	# compute inviscid flux vectors of left and right state
	flux_left[0,:]   = sol_cons_left[1,:]
	flux_left[1,:]   = sol_cons_left[1,:] * sol_prim_left[1,:] + sol_prim_left[0,:]
	flux_left[2,:]   = sol_cons_left[0,:] * sol_domain.sol_left.h0 * sol_prim_left[1,:]
	flux_left[3:,:]  = sol_cons_left[3:,:] * sol_prim_left[[1],:]
	flux_right[0,:]  = sol_cons_right[1,:]
	flux_right[1,:]  = sol_cons_right[1,:] * sol_prim_right[1,:] + sol_prim_right[0,:]
	flux_right[2,:]  = sol_cons_right[0,:] * sol_domain.sol_right.h0 * sol_prim_right[1,:]
	flux_right[3:,:] = sol_cons_right[3:,:] * sol_prim_right[[1],:]

	# maximum wave speed for adapting dtau, if needed
	# TODO: need to adaptively size this for hyper-reduction
	if (sol_domain.time_integrator.adapt_dtau):
		srf = np.maximum(sol_ave.sol_prim[1,:] + sol_ave.c, sol_ave.sol_prim[1,:] - sol_ave.c)
		sol_domain.sol_int.srf = np.maximum(srf[:-1], srf[1:])

	# dissipation term
	d_sol_prim = sol_prim_left - sol_prim_right
	sol_domain.roe_diss = calc_roe_diss(sol_ave)
	diss_term = 0.5 * (sol_domain.roe_diss * np.expand_dims(d_sol_prim, 0)).sum(-2)

	# complete Roe flux
	flux = 0.5 * (flux_left + flux_right) + diss_term 

	return flux


def calc_roe_diss(sol_ave):
	"""
	Compute dissipation term of Roe flux
	"""

	diss_matrix = np.zeros((sol_ave.gas_model.num_eqs, sol_ave.gas_model.num_eqs, sol_ave.num_cells), dtype=REAL_TYPE)        
	
	# for clarity
	rho        = sol_ave.sol_cons[0,:]
	press      = sol_ave.sol_prim[0,:]
	vel        = sol_ave.sol_prim[1,:]
	temp       = sol_ave.sol_prim[2,:]
	mass_fracs = sol_ave.sol_prim[3:,:]

	# derivatives of density and enthalpy
	d_rho_d_press, d_rho_d_temp, d_rho_d_mass_frac = sol_ave.gas_model.calc_dens_derivs(sol_ave.sol_cons[0,:],
																						wrtPress=True, pressure=sol_ave.sol_prim[0,:],
																						wrtTemp=True, temperature=sol_ave.sol_prim[2,:],
																						wrtSpec=True, mass_fracs=sol_ave.sol_prim[3:,:])

	d_enth_d_press, d_enth_d_temp, d_enth_d_mass_frac = sol_ave.gas_model.calc_stag_enth_derivs(wrtPress=True,
																								wrtTemp=True, mass_fracs=sol_ave.sol_prim[3:,:],
																								wrtSpec=True, temperature=sol_ave.sol_prim[2,:])

	# save for Jacobian calculations
	sol_ave.d_rho_d_press = d_rho_d_press; sol_ave.d_rho_d_temp = d_rho_d_temp; sol_ave.d_rho_d_mass_frac = d_rho_d_mass_frac
	sol_ave.d_enth_d_press = d_enth_d_press; sol_ave.d_enth_d_temp = d_enth_d_temp; sol_ave.d_enth_d_mass_frac = d_enth_d_mass_frac
	
	# gamma terms for energy equation
	g_press = rho * d_enth_d_press + d_rho_d_press * sol_ave.h0 - 1.0
	g_temp = rho * d_enth_d_temp + d_rho_d_temp * sol_ave.h0
	g_mass_frac = rho[None,:] * d_enth_d_mass_frac +  sol_ave.h0[None,:] * d_rho_d_mass_frac

	# characteristic speeds
	lambda1 = vel + sol_ave.c
	lambda2 = vel - sol_ave.c
	lambda1_abs = np.absolute(lambda1)
	lambda2_abs = np.absolute(lambda2)

	r_roe = (lambda2_abs - lambda1_abs) / (lambda2 - lambda1)
	alpha = sol_ave.c * (lambda1_abs + lambda2_abs) / (lambda1 - lambda2)
	beta = np.power(sol_ave.c, 2.0) * (lambda1_abs - lambda2_abs) / (lambda1 - lambda2)
	phi = sol_ave.c * (lambda1_abs + lambda2_abs) / (lambda1 - lambda2)

	eta = (1.0 - rho * d_enth_d_press) / d_enth_d_temp
	psi = eta * d_rho_d_temp + rho * d_rho_d_press

	vel_abs = np.absolute(vel)

	beta_star = beta * psi
	beta_e = beta * (rho * g_press + g_temp * eta)
	phi_star = d_rho_d_press * phi + d_rho_d_temp * eta * (phi - vel_abs) / rho
	phi_e = g_press * phi + g_temp * eta * (phi - vel_abs) / rho
	m = rho * alpha
	e = rho * vel * alpha

	# continuity equation row
	diss_matrix[0,0,:] = phi_star
	diss_matrix[0,1,:] = beta_star
	diss_matrix[0,2,:] = vel_abs * d_rho_d_temp
	diss_matrix[0,3:,:] = vel_abs[None,:] * d_rho_d_mass_frac

	# momentum equation row
	diss_matrix[1,0,:] = vel * phi_star + r_roe
	diss_matrix[1,1,:] = vel * beta_star + m
	diss_matrix[1,2,:] = vel * vel_abs * d_rho_d_temp
	diss_matrix[1,3:,:] = (vel * vel_abs)[None,:] * d_rho_d_mass_frac

	# energy equation row
	diss_matrix[2,0,:] = phi_e + r_roe * vel
	diss_matrix[2,1,:] = beta_e + e
	diss_matrix[2,2,:] = g_temp * vel_abs
	diss_matrix[2,3:,:] = g_mass_frac * vel_abs[None,:]

	# species transport row
	diss_matrix[3:,0,:] = mass_fracs * phi_star[None,:]
	diss_matrix[3:,1,:] = mass_fracs * beta_star[None,:]
	diss_matrix[3:,2,:] = mass_fracs * (vel_abs * d_rho_d_temp)[None,:]
	# TODO: vectorize
	for mf_idx_out in range(3, sol_ave.gas_model.num_eqs):
		for mf_idx_in in range(3, sol_ave.gas_model.num_eqs):
			# TODO: might want to check this again against GEMS, something weird going on
			if (mf_idx_out == mf_idx_in):
				diss_matrix[mf_idx_out,mf_idx_in,:] = vel_abs * (rho + mass_fracs[mf_idx_out-3,:] * d_rho_d_mass_frac[mf_idx_in-3,:])
			else:
				diss_matrix[mf_idx_out,mf_idx_in,:] = vel_abs * mass_fracs[mf_idx_out-3,:] * d_rho_d_mass_frac[mf_idx_in-3,:]

	return diss_matrix

 
def calc_visc_flux(sol_domain, solver):
	"""
	Compute viscous fluxes
	"""

	gas    = sol_domain.gas_model 
	mesh   = solver.mesh
	sol_ave = sol_domain.sol_ave

	# compute 2nd-order state gradients at faces
	# TODO: generalize to higher orders of accuracy
	sol_prim_grad = np.zeros((gas.num_eqs+1, sol_domain.num_flux_faces), dtype=REAL_TYPE)
	sol_prim_grad[:-1,:] = (sol_domain.sol_prim_full[:,sol_domain.flux_samp_right_idxs] - sol_domain.sol_prim_full[:,sol_domain.flux_samp_left_idxs]) / mesh.dx

	# get gradient of last species for diffusion velocity term
	# TODO: maybe a sneakier way to do this?
	mass_fracs   = gas.calc_all_mass_fracs(sol_domain.sol_prim_full[3:,:], threshold=False)
	sol_prim_grad[-1,:] = (mass_fracs[-1,sol_domain.flux_samp_right_idxs] - mass_fracs[-1,sol_domain.flux_samp_left_idxs]) / mesh.dx

	# thermo and transport props
	mole_fracs     = gas.calcAllMoleFracs(sol_ave.sol_prim[3:,:])
	spec_dyn_visc  = gas.calcSpeciesDynamicVisc(sol_ave.sol_prim[2,:])
	therm_cond_mix = gas.calcMixThermalCond(spec_dyn_visc=spec_dyn_visc, mole_fracs=mole_fracs)
	dyn_visc_mix   = gas.calcMixDynamicVisc(spec_dyn_visc=spec_dyn_visc, mole_fracs=mole_fracs)
	mass_diff_mix  = gas.calcSpeciesMassDiffCoeff(sol_ave.sol_cons[0,:], spec_dyn_visc=spec_dyn_visc)
	hi             = gas.calcSpeciesEnthalpies(sol_ave.sol_prim[2,:])

	# copy for use later
	sol_ave.dyn_visc_mix   = dyn_visc_mix
	sol_ave.therm_cond_mix = therm_cond_mix
	sol_ave.mass_diff_mix  = mass_diff_mix
	sol_ave.hi             = hi

	tau = 4.0/3.0 * dyn_visc_mix * sol_prim_grad[1,:]							# stress "tensor"
	diff_vel = sol_ave.sol_cons[[0],:] * mass_diff_mix * sol_prim_grad[3:,:]	# diffusion velocity
	corr_vel = np.sum(diff_vel, axis=0) 										# correction velocity

	# viscous flux
	flux_visc = np.zeros((gas.num_eqs, sol_domain.num_flux_faces), dtype=REAL_TYPE)
	flux_visc[1,:]  += tau
	flux_visc[2,:]  += sol_ave.sol_prim[1,:] * tau + therm_cond_mix * sol_prim_grad[2,:] + np.sum(diff_vel * hi, axis = 0)
	flux_visc[3:,:] += diff_vel[gas.mass_frac_slice] - sol_ave.sol_prim[3:,:] * corr_vel[None,:]

	return flux_visc

 
def calc_source(sol_domain, solver):
	"""
	Compute chemical source term
	"""

	# TODO: expand to multiple global reactions
	# TODO: expand to general reaction w/ reverse direction
	# TODO: really need to check that this works for more than a two-species reaction

	gas = sol_domain.gas_model

	temp 	   = sol_domain.sol_int.sol_prim[2,sol_domain.direct_samp_idxs]
	mass_fracs = gas.calc_all_mass_fracs(sol_domain.sol_int.sol_prim[3:,sol_domain.direct_samp_idxs], threshold=False)
	rho 	   = sol_domain.sol_int.sol_cons[[0],sol_domain.direct_samp_idxs]

	# NOTE: act_energy here is -Ea/R
	# TODO: account for temperature exponential
	wf = gas.pre_exp_fact * np.exp(gas.act_energy / temp)
	
	rho_mass_frac = rho * mass_fracs

	spec_idxs = np.squeeze(np.argwhere(gas.nu_arr != 0.0))
	wf        = np.product( wf[None,:] * np.power((rho_mass_frac[spec_idxs,:] / gas.mol_weights[spec_idxs, None]), gas.nu_arr[spec_idxs, None]), axis=0)
	wf        = np.amin(np.minimum(wf[None,:], rho_mass_frac[spec_idxs,:] / solver.dt), axis=0)
	sol_domain.sol_int.wf = wf

	sol_domain.sol_int.source[gas.mass_frac_slice[:,None], sol_domain.direct_samp_idxs[None,:]] = -gas.mol_weight_nu[gas.mass_frac_slice, None] * wf[None, :]