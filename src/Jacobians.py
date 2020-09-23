import numpy as np
from solution import solutionPhys, boundaries
from classDefs import parameters, geometry, gasProps
from scipy.sparse import csc_matrix, bsr_matrix, block_diag
from stateFuncs import calcCpMixture, calcGasConstantMixture, calcStateFromPrim, calcGammaMixture
import constants
from spaceSchemes import calcInvFlux, calcViscFlux, calcSource, reconstruct_2nd, calcRoeDissipation, calcRHS
from matplotlib.pyplot import spy
import copy
import pdb

# repeated function for calculating the 
def calcRoeDissipation_alt(solPrim, rho, h0, c, R, Cp, gas: gasProps):
	
	dissMat = np.zeros((gas.numEqs, gas.numEqs, solPrim.shape[0]), dtype = constants.realType)
	if (solPrim.dtype == constants.complexType):
		dissMat = np.zeros((gas.numEqs, gas.numEqs, solPrim.shape), dtype = constants.complexType)        
	temp = solPrim[:,2]

	# TODO: use the relevant stateFunc
	massFracs = solPrim[:,3:]
	rhoY = -np.square(rho) * (constants.RUniv * temp / solPrim[:,0] * gas.mwDiffs)
	hY = gas.enthRefDiffs + (temp - gas.tempRef) * gas.CpDiffs

	rhop = 1.0 / (R * temp)
	rhoT = -rho / temp
	hT = Cp
	hp = 0.0

	Gp = rho * hp + rhop * h0 - 1.0
	GT = rho *hT + rhoT * h0

	GY = rho * hY + rhoY * h0

	u = solPrim[:,1]
	lambda1 = u + c
	lambda2 = u - c
	lam1 = np.absolute(lambda1)
	lam2 = np.absolute(lambda2)
	R_roe = (lam2 - lam1) / (lambda2 - lambda1)
	alpha = c * (lam1 + lam2) / (lambda1 - lambda2)
	beta = np.power(c, 2.0) * (lam1 - lam2) / (lambda1 - lambda2)
	phi = c * (lam1 + lam2) / (lambda1 - lambda2)

	eta = (1.0 - rho * hp) / hT
	psi = eta * rhoT + rho * rhop

	u_abs = np.absolute(solPrim[:,1])

	beta_star = beta * psi
	beta_e = beta * (rho * Gp + GT * eta)
	phi_star = rhop * phi + rhoT * eta * (phi - u_abs)
	phi_e = Gp * phi + GT * eta * (phi - u_abs)
	m = rho * alpha
	e = rho * u * alpha

	dissMat[0,0,:] = phi_star
	dissMat[0,1,:] = beta_star
	dissMat[0,2,:] = u_abs * rhoT

	if (gas.numSpecies > 1):
		dissMat[0,3:,:] = u_abs * rhoY
	else:
		dissMat[0,3,:] = u_abs * rhoY
	dissMat[1,0,:] = u *phi_star + R_roe
	dissMat[1,1,:] = u * beta_star + m
	dissMat[1,2,:] = u * u_abs * rhoT

	if (gas.numSpecies > 1):
		dissMat[1,3:,:] = u * u_abs * rhoY
	else:
		dissMat[1,3,:] = u * u_abs * rhoY

	dissMat[2,0,:] = phi_e + R_roe * u
	dissMat[2,1,:] = beta_e + e
	dissMat[2,2,:] = GT * u_abs

	if (gas.numSpecies > 1):
		dissMat[2,3:,:] = GY * u_abs
	else:
		dissMat[2,3,:] = GY * u_abs

	for yIdx_out in range(3, gas.numEqs):
		
		dissMat[yIdx_out, 0, :] = massFracs[:, yIdx_out-3] * phi_star
		dissMat[yIdx_out, 1, :] = massFracs[:, yIdx_out-3] * beta_star
		dissMat[yIdx_out, 2, :] = massFracs[:, yIdx_out-3] * u_abs * rhoT

		for yIdx_in in range(3, gas.numEqs):
			# TODO: rhoY is currently calculated incorrectly for multiple species, only works for two species 
			# 		In a working model, rhoY should be rhoY[:, yIdxs_in - 3]
			if (yIdx_out == yIdx_in):
				dissMat[yIdx_out, yIdx_in, :] = u_abs * (rho + massFracs[:, yIdx_out-3] * rhoY)
			else:
				dissMat[yIdx_out, yIdx_in, :] = u_abs * massFracs[:, yIdx_out-3] * rhoY

	return dissMat


### Gamma Inverse ###
def calc_dsolPrimdsolCons(solCons, solPrim, gas: gasProps):
	
	gamma_matrix_inv = np.zeros((gas.numEqs, gas.numEqs, solPrim.shape[0]))
	
	rho = solCons[:,0]
	p = solPrim[:,0]
	u = solPrim[:,1]
	T = solPrim[:,2]
	
	if (gas.numSpecies > 1):
		Y = solPrim[:,3:]
		massFracs = solPrim[:,3:]
	else:
		Y = solPrim[:,3]
		massFracs = solPrim[:,3]
		
	Ri = calcGasConstantMixture(massFracs, gas)
	Cpi = calcCpMixture(massFracs, gas)
	
	rhop = 1 / (Ri * T)
	rhoT = -rho / T
	hT = Cpi
	d = rho * rhop * hT + rhoT
	h0 = (solCons[:,2] + p) / rho
	
	if (gas.numSpecies == 0):
		gamma11 = (rho * hT + rhoT * (h0 - (u * u))) / d
		
	else:
		rhoY = -(rho * rho) * (constants.RUniv * T / p) * (1 /gas.molWeights[0] - 1 /gas.molWeights[gas.numSpecies_full-1])
		hY = gas.enthRefDiffs + (T - gas.tempRef) * (gas.Cp[0] - gas.Cp[gas.numSpecies_full-1])
		gamma11 = (rho * hT + rhoT * (h0 - (u * u)) + (Y * (rhoY * hT - rhoT * hY))) / d 
		
	gamma_matrix_inv[0,0,:] = gamma11
	gamma_matrix_inv[0,1,:] = u * rhoT / d
	gamma_matrix_inv[0,2,:] = -rhoT / d
	
	if (gas.numSpecies > 0):
		gamma_matrix_inv[0,3:,:] = (rhoT * hY - rhoY * hT) / d
		
	gamma_matrix_inv[1,0,:] = -u / rho
	gamma_matrix_inv[1,1,:] = 1 / rho
	
	if (gas.numSpecies == 0):
		gamma_matrix_inv[2,0,:] = (-rhop * (h0 - (u * u)) + 1.0) / d
		
	else:
		gamma_matrix_inv[2,0,:] = (-rhop * (h0 - (u * u)) + 1.0 + (Y * (rho * rhop * hY + rhoY)) / rho) / d
		gamma_matrix_inv[2,3:,:] = -(rho * rhop * hY + rhoY) / (rho * d)
		
	gamma_matrix_inv[2,1,:] = -u * rhop / d
	gamma_matrix_inv[2,2,:] = rhop / d
	
	if (gas.numSpecies > 0):
		gamma_matrix_inv[3:,0,:] = -Y / rho
		
		for i in range(3,gas.numEqs):
			gamma_matrix_inv[i,i,:] = 1 / rho
			
	return gamma_matrix_inv


# compute gradient of conservative variable solution w/r/t the primitive variable solution
def calc_dsolConsdsolPrim(solCons, solPrim, gas: gasProps):
	
	gamma_matrix = np.zeros((gas.numEqs, gas.numEqs, solPrim.shape[0]))
	rho = solCons[:,0]
	p = solPrim[:,0]
	u = solPrim[:,1]
	T = solPrim[:,2]

	if (gas.numSpecies > 1):
		Y = solPrim[:,3:]
		massFracs = solPrim[:,3:]
	else:
		Y = solPrim[:,3]
		massFracs = solPrim[:,3]
		
	Y = Y.reshape((Y.shape[0], gas.numSpecies))
	Ri = calcGasConstantMixture(massFracs, gas)
	Cpi = calcCpMixture(massFracs, gas)
	
	rhop = 1.0 / (Ri * T)
	rhoT = -rho / T
	hT = Cpi
	d = rho * rhop * hT + rhoT 	# Ashish, this is unused?
	hp = 0.0
	h0 = (solCons[:,2] + p) / rho
	
	if (gas.numSpecies > 0):
		#rhoY = -(rho**2)*(constants.RUniv * T/p)*(1/gas.molWeights[0] - 1/gas.molWeights[gas.numSpecies_full-1])
		rhoY = -np.square(rho) * (constants.RUniv * T / p * gas.mwDiffs)
		hY = gas.enthRefDiffs + (T - gas.tempRef) * (gas.Cp[0] - gas.Cp[gas.numSpecies_full-1])
		
	
	gamma_matrix[0,0,:] = rhop
	gamma_matrix[0,2,:] = rhoT
	
	if (gas.numSpecies > 0):
		gamma_matrix[0,3:,:] = rhoY
		
	gamma_matrix[1,0,:] = u * rhop
	gamma_matrix[1,1,:] = rho
	gamma_matrix[1,2,:] = u * rhoT
	
	if (gas.numSpecies > 0):
		gamma_matrix[1,3:,:] = u * rhoY
		
	gamma_matrix[2,0,:] = rhop * h0 + rho * hp - 1
	gamma_matrix[2,1,:] = rho * u
	gamma_matrix[2,2,:] = rhoT * h0 + rho * hT
	
	if (gas.numSpecies > 0):
		gamma_matrix[2,3:,:] = rhoY * h0 + rho * hY
		
		for i in range(3,gas.numEqs):
			gamma_matrix[i,0,:] = Y[:,i-3] * rhop
			gamma_matrix[i,2,:] = Y[:,i-3] * rhoT
			
			for j in range(3,gas.numEqs):
				rhoY = rhoY.reshape((rhoY.shape[0], gas.numSpecies))
				gamma_matrix[i,j,:] = (i==j) * rho + Y[:,i-3] * rhoY[:,j-3]
				
	return gamma_matrix

# compute Gamma Jacobian numerically, via complex step
def calc_dsolConsdsolPrim_imag(solCons, solPrim, gas: gasProps):
	
	gamma_matrix = np.zeros((gas.numEqs, gas.numEqs, solPrim.shape[0]))
	#gamma_matrix_an = calc_dsolConsdsolPrim(solCons, solPrim, gas)
	h = 1e-25
	
	for i in range(solPrim.shape[0]):
		for j in range(gas.numEqs):
			
			solPrim_curr = solPrim.copy()
			solPrim_curr = solPrim_curr.astype(dtype=constants.complexType)
			#adding complex perturbations
			solPrim_curr[i,j] = solPrim_curr[i,j] + complex(0, h)
			solCons_curr, RMix, enthRefMix, CpMix = calcStateFromPrim(solPrim_curr, gas)
			
			gamma_matrix[:,j,i] = solCons_curr[i,:].imag / h
			
			#Unperturbing
			solPrim_curr[i,j] = solPrim_curr[i,j] - complex(0, h)
	
	#diff = calcRAE(gamma_matrix.ravel(), gamma_matrix_an.ravel())
	
	return gamma_matrix
	

# compute Jacobian of source term
def calc_dSourcedsolPrim(sol, gas: gasProps, geom: geometry, dt):
	
	dSdQp = np.zeros((gas.numEqs, gas.numEqs, geom.numCells))

	rho = sol.solCons[:,0]
	p = sol.solPrim[:,0]
	T = sol.solPrim[:,2]

	
	if (gas.numSpecies > 1):
		Y = sol.solPrim[:,3:]
		massFracs = sol.solPrim[:,3:]
	else:
		Y = (sol.solPrim[:,3]).reshape((sol.solPrim[:,3].shape[0],1))
		massFracs = sol.solPrim[:,3]
		
	Ri = calcGasConstantMixture(massFracs, gas)
	
	rhop = 1/(Ri*T)
	rhoT = -rho/T
	
	#rhoY = -(rho*rho) * (constants.RUniv*T/p) * (1/gas.molWeights[0] - 1/gas.molWeights[gas.numSpecies_full-1])
	rhoY = -np.square(rho) * (constants.RUniv * T / p * gas.mwDiffs)
	
	wf_rho = 0
	
	A = gas.preExpFact
	wf = A * np.exp(gas.actEnergy / T)
	
	for i in range(gas.numSpecies):
		
		if (gas.nuArr[i] != 0):
			wf = wf * ((Y[:,i] * rho / gas.molWeights[i])**gas.nuArr[i])
			wf[Y[:,i] <= 0] = 0
			
	for i in range(gas.numSpecies):
		
		if (gas.nuArr[i] != 0):
			wf = np.minimum(wf, Y[:,i] / dt * rho)
	
	for i in range(gas.numSpecies):
		
		if (gas.nuArr[i] != 0):       
			wf_rho = wf_rho + wf * gas.nuArr[i] / rho
			
	wf_T = wf_rho * rhoT - wf * gas.actEnergy / T**2
	wf_p = wf_rho * rhop
	wf_Y = wf_rho * rhoY
	
	for i in range(gas.numSpecies):
		
		arr = (Y[:,i] > 0)
		s = wf_Y[arr].shape[0]
		wf_Y[arr] = wf_Y[arr] + wf[arr] * (gas.nuArr[i] / Y[arr,:]).reshape(s) 
		dSdQp[3+i,0,:] = -gas.molWeightNu[i] * wf_p
		dSdQp[3+i,2,:] = -gas.molWeightNu[i] * wf_T
		dSdQp[3+i,3+i,:] = -gas.molWeightNu[i] * wf_Y
		
	
	return dSdQp
	

# compute numerical Jacobian of the source term, via complex step
def calc_dSourcedsolPrim_imag(sol, gas: gasProps, geom: geometry, params: parameters, dt, h):  
	
	dSdQp = np.zeros((gas.numEqs, gas.numEqs, geom.numCells))
	dSdQp_an = calc_dSourcedsolPrim(sol, gas, geom, dt)
	
	for i in range(geom.numCells):
		for j in range(gas.numEqs):
			
			solprim_curr = sol.solPrim.copy()
			solprim_curr = solprim_curr.astype(dtype = constants.complexType)
			#adding complex perturbations
			solprim_curr[i,j] = solprim_curr[i,j] + complex(0, h)
			[solcons_curr, RMix, enthRefMix, CpMix] = calcStateFromPrim(solprim_curr, gas)
			S1 = calcSource(solprim_curr, solcons_curr[:,0], params, gas)
			
			#calculating jacobian
			Jac = S1.imag / h
			
			dSdQp[:,j,i] = Jac[i,:]
			
			#Unperturbing
			solprim_curr[i,j] = solprim_curr[i,j] - complex(0, h)
			
	diff = calcRAE(dSdQp_an.ravel(), dSdQp.ravel())
	
	return diff
		


# compute flux Jacobians   
def calc_Ap(solPrim, rho, cp, h0, gas, bounds):
	
	Ap = np.zeros((gas.numEqs, gas.numEqs, solPrim.shape[0]))
	
	p = solPrim[:,0]
	u = solPrim[:,1]
	T = solPrim[:,2]
	
	if (gas.numSpecies > 1):
		Y = solPrim[:,3:].reshape((solPrim.shape[0], gas.numSpecies))
		massFracs = solPrim[:,3:]
	else:
		Y = solPrim[:,3]#.reshape((solPrim.shape[0],gas.numSpecies))
		massFracs = solPrim[:,3]
		
	Ri = calcGasConstantMixture(massFracs, gas)
	Cpi = calcCpMixture(massFracs, gas)
	#rhoY = -(rho*rho)*(constants.RUniv*T/p)*(1/gas.molWeights[0] - 1/gas.molWeights[gas.numSpecies_full-1])
	rhoY = -np.square(rho) * (constants.RUniv * T / p * gas.mwDiffs)
	hY = gas.enthRefDiffs + (T-gas.tempRef)*(gas.CpDiffs)
	
	rhop = 1 / (Ri * T)
	rhoT = -rho / T
	hT = Cpi
	hp = 0
	
	Ap[0,0,:] = rhop * u
	Ap[0,1,:] = rho
	Ap[0,2,:] = rhoT * u
	
	if (gas.numSpecies > 0):
		Ap[0,3:,:] = u * rhoY
		
	Ap[1,0,:] = rhop * (u**2) + 1
	Ap[1,1,:] = 2.0 * rho * u
	Ap[1,2,:] = rhoT * (u**2)
	
	if (gas.numSpecies > 0):
		Ap[1,3:,:] = u**2 * rhoY
		
	h0 = np.squeeze(h0)
	Ap[2,0,:] = u * (rhop * h0 + rho * hp)
	Ap[2,1,:] = rho * (u**2 + h0)
	Ap[2,2,:] = u * (rhoT * h0 + rho * hT)
	
	if (gas.numSpecies > 0):
		Ap[2,3:,:] = u * (rhoY * h0 + rho * hY)
		
		for i in range(3,gas.numEqs):
			
			Ap[i,0,:] = Y * rhop * u
			Ap[i,1,:] = Y * rho
			Ap[i,2,:] = rhoT * u * Y
			
			for j in range(3,gas.numEqs):
				Ap[i,j,:] = u * ((i==j) * rho + Y * rhoY[:])
			
    #Adding the viscous flux jacobian terms
	Ap[1,1,:] = Ap[1,1,:] - (4.0 / 3.0) * gas.muRef[:-1]

	Ap[2,1,:] = Ap[2,1,:] - u * (4.0 / 3.0) * gas.muRef[:-1]

	Ck = gas.muRef[:-1] * cp / gas.Pr[:-1]
	Ap[2,2,:] = Ap[2,2,:] - Ck

	T = solPrim[:,2]
	if (gas.numSpecies > 0):
		Cd = gas.muRef[:-1] / gas.Sc[0] / rho      
		rhoCd = rho * Cd    
		hY = gas.enthRefDiffs + (T - gas.tempRef) * gas.CpDiffs
        
		for i in range(3,gas.numEqs):
			
			Ap[2,i,:] = Ap[2,i,:] - rhoCd * hY 
			Ap[i,i,:] = Ap[i,i,:] - rhoCd


	return Ap


# compute the gradient of the inviscid and viscous fluxes with respect to the PRIMITIVE variables
# TODO: get rid of the left and right dichotomy, just use slices of solCons and solPrim
# 	Redundant Ap calculations are EXPENSIVE
def calc_dFluxdsolPrim(solConsL, solPrimL, solConsR, solPrimR, 
						sol: solutionPhys, bounds: boundaries, geom: geometry, gas: gasProps):
		
	rHL = solConsL[:,[2]] + solPrimL[:,[0]]
	HL = rHL / solConsL[:,[0]]
	
	rHR = solConsR[:,[2]] + solPrimR[:,[0]]
	HR = rHR/solConsR[:,[0]]
	
	# Roe Average
	rhoi = np.sqrt(solConsR[:,0] * solConsL[:,0])
	di = np.sqrt(solConsR[:,[0]] / solConsL[:,[0]])
	dl = 1.0 / (1.0 + di)
	
	Qp_i = (solPrimR*di + solPrimL) * dl
	
	Hi = np.squeeze((di * HR + HL) * dl)
	
	if (gas.numSpecies > 1):
		massFracsRoe = Qp_i[:,3:]
	else:
		massFracsRoe = Qp_i[:,3]
		
	Ri = calcGasConstantMixture(massFracsRoe, gas)
	Cpi = calcCpMixture(massFracsRoe, gas)
	gammai = calcGammaMixture(Ri, Cpi)
	
	ci = np.sqrt(gammai * Ri * Qp_i[:,2])
	
	M_ROE = np.transpose(calcRoeDissipation(Qp_i, rhoi, Hi, ci, Ri, Cpi, gas), axes=(1,2,0))

	cp_l = np.concatenate((bounds.inlet.sol.CpMix, sol.CpMix), axis=0)
	cp_r = np.concatenate((sol.CpMix, bounds.outlet.sol.CpMix), axis=0)

	Ap_l = (calc_Ap(solPrimL, solConsL[:,0], cp_l, HL, gas, bounds))
	Ap_r = (calc_Ap(solPrimR, solConsR[:,0], cp_r, HL, gas, bounds))

    #Jacobian wrt current cell
	dFluxdQp = ((0.5*Ap_l[:,:,1:] + 0.5*M_ROE[:,:,1:]) + (-0.5*Ap_r[:,:,:-1] + 0.5*M_ROE[:,:,1:])) / geom.dx
    
    #Jacobian wrt left neighbour
	dFluxdQp_l = (-0.5*Ap_l[:,:,1:-1] - 0.5*M_ROE[:,:,:-2]) / geom.dx
    
    #Jacobian wrt right neighbour
	dFluxdQp_r = (0.5*Ap_r[:,:,1:-1] - 0.5*M_ROE[:,:,2:]) / geom.dx    
    
	
	return dFluxdQp, dFluxdQp_l, dFluxdQp_r


# compute Jacobian of the RHS function (i.e. fluxes, sources, body forces)  
def calcDResDSolPrim(sol: solutionPhys, gas: gasProps, geom: geometry, params: parameters, bounds: boundaries, 
						dt_inv, dtau_inv):
		
	dSdQp = calc_dSourcedsolPrim(sol, gas, geom, params.dt)
	
	gamma_matrix = calc_dsolConsdsolPrim(sol.solCons, sol.solPrim, gas)
		
	if (params.spaceOrder == 1):
		solPrimL = np.concatenate((bounds.inlet.sol.solPrim, sol.solPrim), axis=0)
		solConsL = np.concatenate((bounds.inlet.sol.solCons, sol.solCons), axis=0)
		solPrimR = np.concatenate((sol.solPrim, bounds.outlet.sol.solPrim), axis=0)
		solConsR = np.concatenate((sol.solCons, bounds.outlet.sol.solCons), axis=0)
	elif (params.spaceOrder == 2):
		[solPrimL, solConsL, solPrimR, solConsR, phi] = reconstruct_2nd(sol, bounds, geom, gas)
	else:
		raise ValueError("Higher-Order fluxes not implemented yet")
		
	dFdQp, dFdQp_l, dFdQp_r = calc_dFluxdsolPrim(solConsL, solPrimL, solConsR, solPrimR, sol, bounds, geom, gas)

	dRdQp = gamma_matrix * dtau_inv + gamma_matrix * dt_inv - dSdQp 
							
	dRdQp = resJacobAssemble(dRdQp, dFdQp, dFdQp_l, dFdQp_r)

	return dRdQp


# compute numerical RHS Jacobian, using complex step
def calcDResDSolPrimImag(sol: solutionPhys, gas: gasProps, geom: geometry, params: parameters, bounds: boundaries, dt_inv, dtau_inv):
	
	sol_curr = copy.deepcopy(sol)
	h = 1e-25
	
	nsamp = geom.numCells
	neq = gas.numEqs
	gamma_matrix = calc_dsolConsdsolPrim_imag(sol_curr.solCons, sol_curr.solPrim, gas)
	dRdQp_an = calc_dresdsolPrim(sol, gas, geom, params, bounds, dt_inv, dtau_inv)
	dRdQp = np.zeros((nsamp, neq, neq, nsamp))
	
	for i in range(nsamp):
		for j in range(neq):
			
			sol_curr = copy.deepcopy(sol)
			
			sol_curr.solPrim = sol_curr.solPrim.astype(dtype = constants.complexType)
			#adding complex perturbations
			sol_curr.solPrim[i,j] = sol_curr.solPrim[i,j] + complex(0, h)
			[sol_curr.solCons, RMix, enthRefMix, CpMix] = calcStateFromPrim(sol_curr.solPrim, gas)
			calcRHS(sol_curr, bounds, params, geom, gas)
			
			Jac = sol_curr.RHS
			Jac = Jac.imag / h
			
			dRdQp[:,:,j,i] = gamma_matrix[:,j,i] * dtau_inv + dt_inv * gamma_matrix[:,j,i] - Jac[i,:]
		
		
	diff = calcRAE(dRdQp_an.toarray().ravel(), dRdQp.ravel())
	
	return diff



### Miscellaneous ###   
# TODO: move these a different module

# compute relative absolute error (RAE)
# TODO: is this actually the relative absolute error?
def calcRAE(truth,pred):
	
	RAE = np.mean(np.abs(truth - pred)) / np.max(np.abs(truth))
	
	return RAE

# reassemble residual Jacobian into a 2D array for linear solve
def resJacobAssemble(mat1, mat2, mat3, mat4):
	
	'''
	Stacking block diagonal forms of mat1 and block tri-diagonal form of mat2
	mat1 : 3-D Form of Gamma*(1/dt) + Gamma*(1/dtau) - dS/dQp
	mat2 : 3-D Form of (dF/dQp)_i
	mat3 : 3-D Form of (dF/dQp)_(i-1) (Left Neighbour)
	mat4 : 3-D Form of (dF/dQp)_(i+1) (Right Neighbour)
	'''

	numEqs, _, numCells = mat1.shape
	 
	# put arrays in proper format for use with bsr_matrix
	# zeroPad is because I don't know how to indicate that a row should have no blocks added when using bsr_matrix
	zeroPad = np.zeros((1,numEqs,numEqs), dtype = constants.realType)
	center = np.transpose(mat1 + mat2, (2,0,1))
	lower = np.concatenate((zeroPad, np.transpose(mat3, (2,0,1))), axis=0) 
	upper = np.concatenate((np.transpose(mat4, (2,0,1)), zeroPad), axis=0)

	# BSR format indices and indices pointers
	indptr = np.arange(numCells+1)
	indices_center = np.arange(numCells)
	indices_lower = np.arange(numCells)
	indices_lower[1:] -= 1
	indices_upper = np.arange(1,numCells+1)
	indices_upper[-1] -= 1

	# format center, lower, and upper block diagonals
	jacDim = numEqs * numCells
	center_sparse = bsr_matrix((center, indices_center, indptr), shape=(jacDim, jacDim))
	lower_sparse  = bsr_matrix((lower, indices_lower, indptr), shape=(jacDim, jacDim))
	upper_sparse  = bsr_matrix((upper, indices_upper, indptr), shape=(jacDim, jacDim))

	# assemble full matrix
	resJacob  = center_sparse + lower_sparse + upper_sparse 

	return resJacob



