import numpy as np 
from constants import realType, RUniv
import os
from solution import solutionPhys, boundaries
import stateFuncs
from boundaryFuncs import calcBoundaries
from higherOrderFuncs import calcCellGradients
from miscFuncs import writeToFile
import pdb	

# TODO: check for repeated calculations, just make a separate variable
# TODO: check references to muRef, might be some broadcast issues

# compute RHS function
def calcRHS(sol: solutionPhys, bounds: boundaries, solver):

	# compute ghost cell state or boundary fluxes
	# TODO: update this after higher-order contribution?
	calcBoundaries(sol, bounds, solver)

	# state at left and right of cell face
	solPrimL = np.concatenate((bounds.inlet.sol.solPrim, sol.solPrim), axis=0)
	solConsL = np.concatenate((bounds.inlet.sol.solCons, sol.solCons), axis=0)
	solPrimR = np.concatenate((sol.solPrim, bounds.outlet.sol.solPrim), axis=0)
	solConsR = np.concatenate((sol.solCons, bounds.outlet.sol.solCons), axis=0)       

	# add higher-order contribution
	if (solver.spaceOrder > 1):
		solPrimGrad = calcCellGradients(sol, bounds, solver)
		solPrimL[1:,:] 	+= (solver.mesh.dx / 2.0) * solPrimGrad 
		solPrimR[:-1,:] -= (solver.mesh.dx / 2.0) * solPrimGrad
		solConsL[1:,:], _, _ ,_ = stateFuncs.calcStateFromPrim(solPrimL[1:,:], solver.gasModel)
		solConsR[:-1,:], _, _ ,_ = stateFuncs.calcStateFromPrim(solPrimR[:-1,:], solver.gasModel)

	# compute fluxes
	flux, solPrimAve, solConsAve, CpAve = calcInvFlux(solPrimL, solConsL, solPrimR, solConsR, sol, solver)

	if (solver.viscScheme > 0):
		viscFlux = calcViscFlux(sol, solPrimAve, solConsAve, CpAve, bounds, solver)
		flux -= viscFlux

	# compute RHS
	sol.RHS = flux[:-1,:] - flux[1:,:]
	sol.RHS[:,:] /= solver.mesh.dx

	# compute source term
	if solver.sourceOn:
		calcSource(sol, solver)
		sol.RHS[:,3:] += sol.source 

# compute inviscid fluxes
# TODO: expand beyond Roe flux
# TODO: better naming conventions
# TODO: entropy fix
def calcInvFlux(solPrimL, solConsL, solPrimR, solConsR, sol: solutionPhys, solver):

	# inviscid flux vector
	EL = np.zeros(solPrimL.shape, dtype=realType)
	ER = np.zeros(solPrimR.shape, dtype=realType)     

	# compute sqrhol, sqrhor, fac, and fac1
	sqrhol = np.sqrt(solConsL[:, 0])
	sqrhor = np.sqrt(solConsR[:, 0])
	fac = sqrhol / (sqrhol + sqrhor)
	fac1 = 1.0 - fac

	# Roe average stagnation enthalpy and density
	h0L = stateFuncs.calcStagnationEnthalpy(solPrimL, solver.gasModel)
	h0R = stateFuncs.calcStagnationEnthalpy(solPrimR, solver.gasModel) 
	h0Ave = fac * h0L + fac1 * h0R 
	rhoAve = sqrhol * sqrhor

	# compute Roe average primitive state, adjust iteratively to conform to Roe average density and enthalpy
	solPrimAve = fac[:,None] * solPrimL + fac1[:,None] * solPrimR
	solPrimAve = stateFuncs.calcStateFromRhoH0(solPrimAve, rhoAve, h0Ave, solver.gasModel)

	# compute Roe average state at faces, associated fluid properties
	solConsAve, RAve, enthRefAve, CpAve = stateFuncs.calcStateFromPrim(solPrimAve, solver.gasModel)
	gammaAve = stateFuncs.calcGammaMixture(RAve, CpAve)
	cAve = np.sqrt(gammaAve * RAve * solPrimAve[:,2])

	# compute inviscid flux vectors of left and right state
	EL[:,0] = solConsL[:,1]
	EL[:,1] = solConsL[:,1] * solPrimL[:,1] + solPrimL[:,0]
	EL[:,2] = solConsL[:,0] * h0L * solPrimL[:,1]
	EL[:,3:] = solConsL[:,3:] * solPrimL[:,[1]]
	ER[:,0] = solConsR[:,1]
	ER[:,1] = solConsR[:,1] * solPrimR[:,1] + solPrimR[:,0]
	ER[:,2] = solConsR[:,0] * h0R * solPrimR[:,1]
	ER[:,3:] = solConsR[:,3:] * solPrimR[:,[1]]

	# maximum wave speed for adapting dtau, if needed
	if (solver.timeIntegrator.adaptDTau):
		srf = np.maximum(solPrimAve[:,1] + cAve, solPrimAve[:,1] - cAve)
		sol.srf = np.maximum(srf[:-1], srf[1:])

	# dissipation term
	dQp = solPrimL - solPrimR
	M_ROE = calcRoeDissipation(solPrimAve, solConsAve[:,0], h0Ave, cAve, CpAve, solver.gasModel)
	dissTerm = 0.5 * (M_ROE * np.expand_dims(dQp, -2)).sum(-1)

	# complete Roe flux
	flux = 0.5 * (EL + ER) + dissTerm 

	return flux, solPrimAve, solConsAve, CpAve

# compute dissipation term of Roe flux
# inputs are all from Roe average state
# TODO: a lot of these quantities need to be generalized for different gas models
def calcRoeDissipation(solPrim, rho, h0, c, Cp, gas):

	# allocate
	dissMat = np.zeros((solPrim.shape[0], gas.numEqs, gas.numEqs), dtype=realType)        
	
	# primitive variables for clarity
	press = solPrim[:,0]
	vel = solPrim[:,1]
	temp = solPrim[:,2]
	massFracs = solPrim[:,3:]

	rhoY = -np.square(rho) * (RUniv * temp / press * gas.mwInvDiffs)
	hY = gas.enthRefDiffs + (temp - gas.tempRef) * gas.CpDiffs

	rhop = rho / press 					# derivative of density with respect to pressure
	rhoT = -rho / temp 					# derivative of density with respect to temperature
	hT = Cp 							# derivative of enthalpy with respect to temperature
	hp = 0.0 							# derivative of enthalpy with respect to pressure

	# gamma terms for energy equation
	Gp = rho * hp + rhop * h0 - 1.0
	GT = rho *hT + rhoT * h0
	GY = rho * hY + rhoY * h0

	# characteristic speeds
	lambda1 = vel + c
	lambda2 = vel - c
	lam1 = np.absolute(lambda1)
	lam2 = np.absolute(lambda2)

	R_roe = (lam2 - lam1) / (lambda2 - lambda1)
	alpha = c * (lam1 + lam2) / (lambda1 - lambda2)
	beta = np.power(c, 2.0) * (lam1 - lam2) / (lambda1 - lambda2)
	phi = c * (lam1 + lam2) / (lambda1 - lambda2)

	eta = (1.0 - rho * hp) / hT
	psi = eta * rhoT + rho * rhop

	u_abs = np.absolute(vel)

	beta_star = beta * psi
	beta_e = beta * (rho * Gp + GT * eta)
	phi_star = rhop * phi + rhoT * eta * (phi - u_abs) / rho
	phi_e = Gp * phi + GT * eta * (phi - u_abs) / rho
	m = rho * alpha
	e = rho * vel * alpha

	dissMat[:,0,0] = phi_star
	dissMat[:,0,1] = beta_star
	dissMat[:,0,2] = u_abs * rhoT

	if (gas.numSpecies > 1):
		dissMat[:,0,3:] = u_abs * rhoY
	else:
		dissMat[:,0,3] = u_abs * rhoY
	dissMat[:,1,0] = vel * phi_star + R_roe
	dissMat[:,1,1] = vel * beta_star + m
	dissMat[:,1,2] = vel * u_abs * rhoT

	if (gas.numSpecies > 1):
		dissMat[:,1,3:] = vel * u_abs * rhoY
	else:
		dissMat[:,1,3] = vel * u_abs * rhoY

	dissMat[:,2,0] = phi_e + R_roe * vel
	dissMat[:,2,1] = beta_e + e
	dissMat[:,2,2] = GT * u_abs

	if (gas.numSpecies > 1):
		dissMat[:,2,3:] = GY * u_abs
	else:
		dissMat[:,2,3] = GY * u_abs

	for yIdx_out in range(3, gas.numEqs):
		dissMat[:, yIdx_out, 0] = massFracs[:, yIdx_out-3] * phi_star
		dissMat[:, yIdx_out, 1] = massFracs[:, yIdx_out-3] * beta_star
		dissMat[:, yIdx_out, 2] = massFracs[:, yIdx_out-3] * u_abs * rhoT

		for yIdx_in in range(3, gas.numEqs):
			# TODO: rhoY is currently calculated incorrectly for multiple species, only works for two species 
			# 		In a working model, rhoY should be rhoY[:, yIdxs_in - 3]
			if (yIdx_out == yIdx_in):
				dissMat[:, yIdx_out, yIdx_in] = u_abs * (rho + massFracs[:, yIdx_out-3] * rhoY)
			else:
				dissMat[:, yIdx_out, yIdx_in] = u_abs * massFracs[:, yIdx_out-3] * rhoY

	return dissMat

# compute viscous fluxes
def calcViscFlux(sol: solutionPhys, solPrimAve, solConsAve, CpAve, bounds: boundaries, solver):

	gas 	= solver.gasModel 
	mesh 	= solver.mesh

	# compute 2nd-order state gradients at face
	solPrimGrad = np.zeros((mesh.numCells+1, gas.numEqs), dtype=realType)
	solPrimGrad[1:-1,:] = (sol.solPrim[1:,:] - sol.solPrim[:-1,:]) / mesh.dCellCent[1:-1,:]
	solPrimGrad[0,:] 	= (sol.solPrim[0,:] - bounds.inlet.sol.solPrim) / mesh.dCellCent[0,:]
	solPrimGrad[-1,:] 	= (bounds.outlet.sol.solPrim - sol.solPrim[-1,:]) / mesh.dCellCent[-1,:]

	Ck = gas.muRef[:-1] * CpAve / gas.Pr[:-1] 									# thermal conductivity
	tau = 4.0/3.0 * gas.muRef[:-1] * solPrimGrad[:,1] 							# stress "tensor"

	Cd = gas.muRef[:-1] / gas.Sc[:-1] / solConsAve[:,0]							# mass diffusivity
	diff_rhoY = solConsAve[:,0] * Cd * np.squeeze(solPrimGrad[:,3:])  			# 
	hY = gas.enthRefDiffs + (solPrimAve[:,2] - gas.tempRef) * gas.CpDiffs 		# species enthalpies, TODO: replace with stateFuncs function

	Fv = np.zeros((mesh.numCells+1, gas.numEqs), dtype=realType)
	Fv[:,1] = Fv[:,1] + tau 
	Fv[:,2] = Fv[:,2] + solPrimAve[:,1] * tau + Ck * solPrimGrad[:,2]
	if (gas.numSpecies > 1):
		Fv[:,2] = Fv[:,2] + np.sum(diff_rhoY * hY, axis = 1)
		Fv[:,3:] = Fv[:,3:] + diff_rhoY 
	else:
		Fv[:,2] = Fv[:,2] + diff_rhoY * hY
		Fv[:,3] = Fv[:,3] + diff_rhoY

	return Fv

# compute source term
# TODO: bring in rho*Yi so it doesn't keep getting calculated
def calcSource(sol: solutionPhys, solver):

	gas = solver.gasModel

	temp = sol.solPrim[:,2]
	massFracs = sol.solPrim[:,3:]

	wf = gas.preExpFact * np.exp(gas.actEnergy / temp)
	
	rhoY = massFracs * sol.solCons[:,[0]]

	for specIdx in range(gas.numSpecies):
		if (gas.nuArr[specIdx] != 0.0):
			wf = wf * np.power((rhoY[:,specIdx] / gas.molWeights[specIdx]), gas.nuArr[specIdx])
			wf[massFracs[:, specIdx] < 0.0] = 0.0

	for specIdx in range(gas.numSpecies):
		if (gas.nuArr[specIdx] != 0.0):
			wf = np.minimum(wf, rhoY[specIdx] / solver.timeIntegrator.dt)

	# TODO: could vectorize this I think
	for specIdx in range(gas.numSpecies):
		sol.source[:,specIdx] = -gas.molWeightNu[specIdx] * wf
