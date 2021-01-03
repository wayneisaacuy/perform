import constants
from solution import solutionPhys, boundaries
from stateFuncs import calcCpMixture, calcGasConstantMixture, calcStateFromPrim, calcGammaMixture
from spaceSchemes import calcRoeDissipation
from higherOrderFuncs import calcCellGradients
import numpy as np
from scipy.sparse import bsr_matrix
import pdb


### Gamma Inverse ###
def calcDSolPrimDSolCons(solCons, solPrim, gas):
	
	gammaMatrixInv = np.zeros((gas.numEqs, gas.numEqs, solPrim.shape[0]))
	
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
		rhoY = -(rho * rho) * (constants.RUniv * T / p) * (1 /gas.molWeights[0] - 1 /gas.molWeights[gas.numSpecies])
		hY = gas.enthRefDiffs + (T - gas.tempRef) * (gas.Cp[0] - gas.Cp[gas.numSpecies])
		gamma11 = (rho * hT + rhoT * (h0 - (u * u)) + (Y * (rhoY * hT - rhoT * hY))) / d 
		
	gammaMatrixInv[0,0,:] = gamma11
	gammaMatrixInv[0,1,:] = u * rhoT / d
	gammaMatrixInv[0,2,:] = -rhoT / d
	
	if (gas.numSpecies > 0):
		gammaMatrixInv[0,3:,:] = (rhoT * hY - rhoY * hT) / d
		
	gammaMatrixInv[1,0,:] = -u / rho
	gammaMatrixInv[1,1,:] = 1 / rho
	
	if (gas.numSpecies == 0):
		gammaMatrixInv[2,0,:] = (-rhop * (h0 - (u * u)) + 1.0) / d
		
	else:
		gammaMatrixInv[2,0,:] = (-rhop * (h0 - (u * u)) + 1.0 + (Y * (rho * rhop * hY + rhoY)) / rho) / d
		gammaMatrixInv[2,3:,:] = -(rho * rhop * hY + rhoY) / (rho * d)
		
	gammaMatrixInv[2,1,:] = -u * rhop / d
	gammaMatrixInv[2,2,:] = rhop / d
	
	if (gas.numSpecies > 0):
		gammaMatrixInv[3:,0,:] = -Y / rho
		
		for i in range(3,gas.numEqs):
			gammaMatrixInv[i,i,:] = 1 / rho
			
	return gammaMatrixInv


# compute gradient of conservative variable solution w/r/t the primitive variable solution
def calcDSolConsDSolPrim(solCons, solPrim, gas):
	
	gammaMatrix = np.zeros((gas.numEqs, gas.numEqs, solPrim.shape[0]))
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
		#rhoY = -(rho**2)*(constants.RUniv * T/p)*(1/gas.molWeights[0] - 1/gas.molWeights[gas.numSpecies])
		rhoY = -np.square(rho) * (constants.RUniv * T / p * gas.mwInvDiffs)
		hY = gas.enthRefDiffs + (T - gas.tempRef) * (gas.Cp[0] - gas.Cp[gas.numSpecies])
		
	
	gammaMatrix[0,0,:] = rhop
	gammaMatrix[0,2,:] = rhoT
	
	if (gas.numSpecies > 0):
		gammaMatrix[0,3:,:] = rhoY
		
	gammaMatrix[1,0,:] = u * rhop
	gammaMatrix[1,1,:] = rho
	gammaMatrix[1,2,:] = u * rhoT
	
	if (gas.numSpecies > 0):
		gammaMatrix[1,3:,:] = u * rhoY
		
	gammaMatrix[2,0,:] = rhop * h0 + rho * hp - 1
	gammaMatrix[2,1,:] = rho * u
	gammaMatrix[2,2,:] = rhoT * h0 + rho * hT
	
	if (gas.numSpecies > 0):
		gammaMatrix[2,3:,:] = rhoY * h0 + rho * hY
		
		for i in range(3,gas.numEqs):
			gammaMatrix[i,0,:] = Y[:,i-3] * rhop
			gammaMatrix[i,2,:] = Y[:,i-3] * rhoT
			
			for j in range(3,gas.numEqs):
				rhoY = rhoY.reshape((rhoY.shape[0], gas.numSpecies))
				gammaMatrix[i,j,:] = (i==j) * rho + Y[:,i-3] * rhoY[:,j-3]
				
	return gammaMatrix

# compute Jacobian of source term 
def calcDSourceDSolPrim(sol, gas, mesh, dt):
	
	dSdQp = np.zeros((gas.numEqs, gas.numEqs, mesh.numCells))

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
	
	#rhoY = -(rho*rho) * (constants.RUniv*T/p) * (1/gas.molWeights[0] - 1/gas.molWeights[gas.numSpecies])
	rhoY = -np.square(rho) * (constants.RUniv * T / p * gas.mwInvDiffs)
	
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
	

# compute flux Jacobians   
def calcAp(solPrim, rho, cp, h0, bounds: boundaries, solver):
	
	gas = solver.gasModel

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
	#rhoY = -(rho*rho)*(constants.RUniv*T/p)*(1/gas.molWeights[0] - 1/gas.molWeights[gas.numSpecies])
	rhoY = -np.square(rho) * (constants.RUniv * T / p * gas.mwInvDiffs)
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
	if (solver.viscScheme > 0):
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
def calcDFluxDSolPrim(solConsL, solPrimL, solConsR, solPrimR, 
						sol: solutionPhys, bounds: boundaries, solver):
		
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
	
	if (solver.gasModel.numSpecies > 1):
		massFracsRoe = Qp_i[:,3:]
	else:
		massFracsRoe = Qp_i[:,3]
		
	Ri = calcGasConstantMixture(massFracsRoe, solver.gasModel)
	Cpi = calcCpMixture(massFracsRoe, solver.gasModel)
	gammai = calcGammaMixture(Ri, Cpi)
	
	ci = np.sqrt(gammai * Ri * Qp_i[:,2])
	
	M_ROE = np.transpose(calcRoeDissipation(Qp_i, rhoi, Hi, ci, Cpi, solver.gasModel), axes=(1,2,0))

	# pdb.set_trace()

	cp_l = np.concatenate((bounds.inlet.sol.CpMix, sol.CpMix))
	cp_r = np.concatenate((sol.CpMix, bounds.outlet.sol.CpMix))

	Ap_l = calcAp(solPrimL, solConsL[:,0], cp_l, HL, bounds, solver)
	Ap_r = calcAp(solPrimR, solConsR[:,0], cp_r, HR, bounds, solver)

	Ap_l[:,:,:]  *= (0.5 / solver.mesh.dx[0,:])
	Ap_r[:,:,:]  *= (0.5 / solver.mesh.dx[0,:])
	M_ROE[:,:,:] *= (0.5 / solver.mesh.dx[0,:])

    #Jacobian wrt current cell
	dFluxdQp = (Ap_l[:,:,1:] + M_ROE[:,:,1:]) + (-Ap_r[:,:,:-1] + M_ROE[:,:,:-1])
    
    #Jacobian wrt left neighbour
	dFluxdQp_l = (-Ap_l[:,:,1:-1] - M_ROE[:,:,:-2])
    
    #Jacobian wrt right neighbour
	dFluxdQp_r = (Ap_r[:,:,1:-1] - M_ROE[:,:,2:]) 
    
	return dFluxdQp, dFluxdQp_l, dFluxdQp_r


# compute Jacobian of the RHS function (i.e. fluxes, sources, body forces)  
def calcDResDSolPrim(sol: solutionPhys, bounds: boundaries, solver):
		
	# contribution to main block diagonal from source term Jacobian
	dSdQp = np.zeros((solver.gasModel.numEqs, solver.gasModel.numEqs, solver.mesh.numCells), dtype=constants.realType)
	if solver.sourceOn:
		dSdQp = calcDSourceDSolPrim(sol, solver.gasModel, solver.mesh, solver.timeIntegrator.dt)
	
	# contribution to main block diagonal from physical/dual time solution Jacobian
	gammaMatrix = calcDSolConsDSolPrim(sol.solCons, sol.solPrim, solver.gasModel)
		
	# contribution from inviscid and viscous flux Jacobians
	# TODO: the face reconstruction should be held onto from the RHS calcs
	solPrimL = np.concatenate((bounds.inlet.sol.solPrim, sol.solPrim), axis=0)
	solConsL = np.concatenate((bounds.inlet.sol.solCons, sol.solCons), axis=0)
	solPrimR = np.concatenate((sol.solPrim, bounds.outlet.sol.solPrim), axis=0)
	solConsR = np.concatenate((sol.solCons, bounds.outlet.sol.solCons), axis=0)       

	# add higher-order contribution
	if (solver.spaceOrder > 1):
		solPrimGrad = calcCellGradients(sol, bounds, solver)
		solPrimL[1:,:] 	+= (solver.mesh.dx / 2.0) * solPrimGrad 
		solPrimR[:-1,:] -= (solver.mesh.dx / 2.0) * solPrimGrad
		solConsL[1:,:], _, _ ,_ = calcStateFromPrim(solPrimL[1:,:], solver.gasModel)
		solConsR[:-1,:], _, _ ,_ = calcStateFromPrim(solPrimR[:-1,:], solver.gasModel)
		
	# *_l is contribution to lower block diagonal, *_r is to upper block diagonal
	dFdQp, dFdQp_l, dFdQp_r = calcDFluxDSolPrim(solConsL, solPrimL, solConsR, solPrimR, sol, bounds, solver)

	# compute time step factors
	# TODO: make this specific for each implicitIntegrator
	dtCoeffIdx = min(solver.timeIntegrator.iter, solver.timeIntegrator.timeOrder) - 1
	dtInv = solver.timeIntegrator.coeffs[dtCoeffIdx][0] / solver.timeIntegrator.dt
	if (solver.timeIntegrator.adaptDTau):
		dtauInv = calcAdaptiveDTau(sol, gammaMatrix, solver)
	else:
		dtauInv = 1./solver.timeIntegrator.dtau

	# compute main block diagonal
	dRdQp = gammaMatrix * (dtauInv + dtInv) - dSdQp + dFdQp
							
	# assemble sparse Jacobian from main, upper, and lower block diagonals
	dRdQp = resJacobAssemble(dRdQp, dFdQp_l, dFdQp_r)

	return dRdQp

def calcAdaptiveDTau(sol: solutionPhys, gammaMatrix, solver):

	# compute initial dtau from input CFL and srf (max characteristic speed)
	# srf is computed in calcInvFlux
	dtaum = 1.0/sol.srf
	dtau = solver.timeIntegrator.CFL*dtaum

	# limit by von Neumann number
	# TODO: THIS NU IS NOT CORRECT FOR A GENERAL MIXTURE
	nu = solver.gasModel.muRef[0] / sol.solCons[:,0]
	dtau = np.minimum(dtau, solver.timeIntegrator.VNN / nu)
	dtaum = np.minimum(dtaum, 3.0 / nu)

	# limit dtau
	# TODO: implement entirety of solutionChangeLimitedTimeStep from gems_precon.f90
	# TODO: might be cheaper to calculate gammaMatrixInv directly, instead of inverting gammaMatrix?
	
	return  1.0 / dtau 

### Miscellaneous ###   
# TODO: move these a different module

# compute relative absolute error (RAE)
# TODO: is this actually the relative absolute error?
def calcRAE(truth,pred):
	
	RAE = np.mean(np.abs(truth - pred)) / np.max(np.abs(truth))
	
	return RAE

# reassemble residual Jacobian into a 2D array for linear solve
def resJacobAssemble(mat1, mat2, mat3):
	
	'''
	Stacking block diagonal forms of mat1 and block tri-diagonal form of mat2
	mat1 : 3-D Form of Gamma*(1/dt) + Gamma*(1/dtau) - dS/dQp + (dF/dQp)_i
	mat2 : 3-D Form of (dF/dQp)_(i-1) (Left Neighbour)
	mat3 : 3-D Form of (dF/dQp)_(i+1) (Right Neighbour)
	'''

	numEqs, _, numCells = mat1.shape
	 
	# put arrays in proper format for use with bsr_matrix
	# zeroPad is because I don't know how to indicate that a row should have no blocks added when using bsr_matrix
	zeroPad = np.zeros((1,numEqs,numEqs), dtype = constants.realType)
	center = np.transpose(mat1, (2,0,1))
	lower = np.concatenate((zeroPad, np.transpose(mat2, (2,0,1))), axis=0) 
	upper = np.concatenate((np.transpose(mat3, (2,0,1)), zeroPad), axis=0)

	# BSR format indices and indices pointers
	indptr = np.arange(numCells+1)
	indicesCenter = np.arange(numCells)
	indicesLower = np.arange(numCells)
	indicesLower[1:] -= 1
	indicesUpper = np.arange(1,numCells+1)
	indicesUpper[-1] -= 1

	# format center, lower, and upper block diagonals
	jacDim = numEqs * numCells
	centerSparse = bsr_matrix((center, indicesCenter, indptr), shape=(jacDim, jacDim))
	lowerSparse  = bsr_matrix((lower, indicesLower, indptr), shape=(jacDim, jacDim))
	upperSparse  = bsr_matrix((upper, indicesUpper, indptr), shape=(jacDim, jacDim))

	# assemble full matrix
	# convert to csr because spsolve requires this
	resJacob  = centerSparse + lowerSparse + upperSparse 
	resJacob = resJacob.tocsr(copy=False)

	return resJacob


