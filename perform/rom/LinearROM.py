import numpy as np
import matplotlib.pyplot as plt  
import copy
import constants
from math import sin, pi
#from classDefs import parameters, geometry, gasProps, catchInput
#from solution import solutionPhys, boundaries, boundary
#from stateFuncs import calcStateFromPrim
#from inputFuncs import readInputFile
#from boundaryFuncs import calcBoundaries
#from higherOrderFuncs import calcCellGradients
from perform.solution.solutionBoundary.solutionInlet import solutionInlet 
from perform.solution.solutionBoundary.solutionOutlet import solutionOutlet
from perform.Jacobians import calcDSourceDSolPrim, calcDSolPrimDSolCons, calcDRoeFluxDSolPrim
import outputFuncs
import time
import os
import sys
import pdb


class linSolROM:

	def __init__(self, romFile, solDomain, solver):

		romDict = readInputFile(romFile)

		gas = solDomain.solInt.gasModel
		geom = solver.mesh
		sol = solDomain.solInt

		self.modelDir 			= romDict["modelDir"]		
		self.ROMOutDir 			= romDict["ROMOutDir"]
		self.nMode		 		= romDict["nMode"]		# number of retained POD modes
		self.initTime			= romDict["initialTime"]
		self.modeName 			= romDict["modeName"] 
		self.normType 			= romDict["normType"]
		self.directProjection	= catchInput(romDict, "directProjection", False)

		self.romJacobian = np.zeros((gas.numEqs, self.nMode, gas.numEqs*geom.numCells), dtype=constants.realType)
		self.romJacobianL = np.zeros((gas.numEqs, self.nMode, gas.numEqs*geom.numCells), dtype=constants.realType)
		self.romJacobianR = np.zeros((gas.numEqs, self.nMode, gas.numEqs*geom.numCells), dtype=constants.realType)

		self.solPODOut  = np.zeros((self.nMode, gas.numEqs, params.numSteps), dtype = constants.realType)
		self.solROMOut  = np.zeros((self.nMode, gas.numEqs, params.numSteps), dtype = constants.realType)
		self.modalPower = np.zeros((self.nMode, gas.numEqs, params.numSteps), dtype = constants.realType)

		self.POD = np.load(os.path.join(self.modelDir, self.modeName+".npy"))  # Nx4xM 
		self.POD = np.transpose(self.POD,(1, 0, 2))

		self.fomSol = np.load(os.path.join(params.workdir, "LinearROMInOut/solPrim_FOM.npy")) 

		try: 
			normSubIn = romDict["normSubIn"]
			if (type(normSubIn) == list):
				assert(len(normSubIn) == sol.solPrim.shape[-1])
				normSubVals = np.arrray(normSubIn, dtype=realType)		
				self.normSubProf = onesProf * normSubVals
			elif (type(normSubIn) == str):
				self.normSubProf = np.load(os.path.join(self.modelDir, normSubIn))				
				assert(self.normSubProf.shape == self.fomSol[:,:,0].shape)
		except:
			print("WARNING: normSubIn load failed or not specified, defaulting to zeros...")
			#self.normSubProf = np.zeros(sol.solPrim.shape, dtype=realType)

		try: 
			normFacIn = romDict["normFacIn"]
			if (type(normFacIn) == list):
				assert(len(normFacIn) == sol.solPrim.shape[-1])
				normFacVals = np.array(normFacIn, dtype=realType)		
				self.normFacProf = onesProf * normFacVals
			elif (type(normFacIn) == str):
				self.normFacProf = np.load(os.path.join(self.modelDir, normFacIn))				
				assert(self.normFacProf.shape == self.fomSol[:,:,0].shape)
		except:
			print("WARNING: normFacIn load failed or not specified, defaulting to ones...")
			#self.normSubProf = np.ones(sol.solPrim.shape, dtype=realType)
		

		try: 
			centIn = romDict["centIn"]
			if (type(centIn) == list):
				assert(len(centIn) == sol.solPrim.shape[-1])
				centVals = np.array(centIn, dtype=realType)		
				self.centProf = onesProf * centVals
			elif (type(centIn) == str):
				self.centProf = np.load(os.path.join(self.modelDir, centIn))					
				assert(self.centProf.shape == self.fomSol[:,:,0].shape)
		except:
			print("WARNING: centIn load failed or not specified, defaulting to zeros...")
			#self.normSubProf = np.zeros(sol.solPrim.shape, dtype=realType)

		fomSol0 = self.fomSol[:, :, 1]
		fomSol0 = self.normalization(fomSol0, center = True)
		fomSol0 = np.transpose(fomSol0, (1, 0))
		self.solROM = np.transpose(self.POD,(0, 2, 1)) @ np.expand_dims(fomSol0, axis=2)
		self.solROM = np.squeeze(self.solROM, axis=2)
		self.solROM = np.transpose(self.solROM, (1, 0)) 

	def initROM(self, solDomain, solver):

		
		gas = solDomain.solInt.gasModel
		geom = solver.mesh
		sol = solDomain.solInt
		calcBoundaries(sol, bounds, params, gas)

		gamma_matrix_inv = calcDSolPrimDSolCons(sol.solCons, sol.solPrim, gas)
		dSdQp = np.zeros((gas.numEqs, gas.numEqs, geom.numCells), dtype=constants.realType)
		if params.sourceOn:
			dSdQp = calcDSourceDSolPrim(sol, gas, geom, params.dt)

		# state at left and right of cell face
		solPrimL = np.concatenate((bounds.inlet.sol.solPrim, sol.solPrim), axis=0)
		solConsL = np.concatenate((bounds.inlet.sol.solCons, sol.solCons), axis=0)
		solPrimR = np.concatenate((sol.solPrim, bounds.outlet.sol.solPrim), axis=0)
		solConsR = np.concatenate((sol.solCons, bounds.outlet.sol.solCons), axis=0)       

		# add higher-order contribution (not functional right now)
		if (params.spaceOrder > 1):
			solPrimGrad = calcCellGradients(sol, params, bounds, geom, gas)
			solPrimL[1:,:] 	+= (geom.dx / 2.0) * solPrimGrad 
			solPrimR[:-1,:] -= (geom.dx / 2.0) * solPrimGrad
			solConsL[1:,:], _, _ ,_ = calcStateFromPrim(solPrimL[1:,:], gas)
			solConsR[:-1,:], _, _ ,_ = calcStateFromPrim(solPrimR[:-1,:], gas)

		dFdQp, dFdQp_l, dFdQp_r = calcDFluxDSolPrim(solConsL, solPrimL, solConsR, solPrimR, sol, params, bounds, geom, gas, linRHS = True)

		if (params.timeType == "explicit"): 
			dFdQp = np.transpose(gamma_matrix_inv,(2, 0, 1)) @ np.transpose(dFdQp, (2, 0, 1))
			self.dFlxdQpR = np.transpose(gamma_matrix_inv,(2, 0, 1)) @ np.transpose(dFdQp_r, (2, 0, 1))
			self.dFlxdQpL = np.transpose(gamma_matrix_inv,(2, 0, 1)) @ np.transpose(dFdQp_l, (2, 0, 1))
			if params.sourceOn:
				dSdQp = np.transpose(gamma_matrix_inv,(2, 0, 1)) @ np.transpose(dSdQp, (2, 0, 1))
				self.dFPlusdS = dFdQp + dSdQp
			else:
				self.dFPlusdS = dFdQp

		for i in range(gas.numEqs):
			for j in range(gas.numEqs):
				self.dFlxdQpL[:,i,j] /= self.normFacProf[:, i]
				self.dFPlusdS[:,i,j] /= self.normFacProf[:, i]
				self.dFlxdQpR[:,i,j] /= self.normFacProf[:, i]

		for i in range(gas.numEqs):
			for j in range(geom.numCells):
				self.romJacobian[i, :, j*gas.numEqs:(j+1)*gas.numEqs] = np.kron(np.expand_dims(self.POD[i, j, :], axis=1) , np.expand_dims(self.dFPlusdS[j, i, :], axis=0))
				self.romJacobianL[i, :, j*gas.numEqs:(j+1)*gas.numEqs] = np.kron(np.expand_dims(self.POD[i, j, :], axis=1) , np.expand_dims(self.dFlxdQpL[j, i, :], axis=0))
				self.romJacobianR[i, :, j*gas.numEqs:(j+1)*gas.numEqs] = np.kron(np.expand_dims(self.POD[i, j, :], axis=1) , np.expand_dims(self.dFlxdQpR[j, i, :], axis=0))

	def linROMRHS(self, sol: solutionPhys, bounds: boundaries, params: parameters, gas: gasProps):

		InBoundState = self.calcLinearInlet(sol, bounds.inlet, params, gas)
		OutBoundState = self.calcLinearOutlet(sol, bounds.outlet, params, gas)

		solPrimL = np.concatenate((InBoundState, sol.solPrim), axis=0)  
		solPrimR = np.concatenate((sol.solPrim, OutBoundState), axis=0) 

		self.romRHS = self.romJacobian @ sol.solPrim.flatten()
		self.romRHS += self.romJacobianL @ solPrimL[:-1,:].flatten()
		self.romRHS += self.romJacobianR @ solPrimR[1:,:].flatten()
		self.romRHS = np.transpose(self.romRHS, (1, 0))

	def reconstruction(self, sol: solutionPhys):
		sol.solPrim = self.POD @ np.expand_dims(np.transpose(self.solROM, (1, 0)), axis=2)  
		sol.solPrim = np.transpose(np.squeeze(sol.solPrim, axis=2), (1, 0)) 
		sol.solPrim = self.denormalization(sol.solPrim)

	def calcLinearInlet(self, sol: solutionPhys, inlet: boundary, params: parameters, gas: gasProps):

		InBoundary = np.zeros((1, gas.numEqs))	
		massFracUp	= inlet.massFrac[:-1]
		rhoCMean 	= inlet.vel 
		rhoCpMean 	= inlet.rho

		if (inlet.pertType == "pressure"):
			pressUp *= inlet.calcPert(params.solTime)

		# interior quantities
		pressIn 	= sol.solPrim[:2,0]
		velIn 		= sol.solPrim[:2,1]

		# characteristic variables
		w3In 	= velIn - pressIn / rhoCMean  

			# extrapolate to exterior
		if (params.spaceOrder == 1):
			w3Bound = w3In[0]
		elif (params.spaceOrder == 2):
			w3Bound = 2.0*w3In[0] - w3In[1]
		else:
			raise ValueError("Higher order extrapolation implementation required for spatial order "+str(params.spaceOrder))

		# compute exterior state
		pressBound 	= -0.5 * w3Bound * rhoCMean
		velBound 	= -pressBound / rhoCMean 
		tempBound 	= pressBound / rhoCpMean
		massFracBound = 0.

		InBoundary[0,0] = pressBound
		InBoundary[0,1] = velBound
		InBoundary[0,2] = tempBound
		InBoundary[0,3:] = massFracBound
		
		return InBoundary

	def calcLinearOutlet(self, sol: solutionPhys, outlet: boundary, params: parameters, gas: gasProps):

		# specify rho*C and rho*Cp from mean solution, back pressure is static pressure at infinity
		OutBoundary = np.zeros((1, gas.numEqs))
		rhoCMean 	= outlet.vel 
		rhoCpMean 	= outlet.rho
		pressBack 	= outlet.press 

		if (outlet.pertType == "pressure"):
			pressBack *= outlet.calcPert(params.solTime)  

		# interior quantities
		pressOut 	= sol.solPrim[-2:,0]
		velOut 		= sol.solPrim[-2:,1]
		tempOut 	= sol.solPrim[-2:,2]
		massFracOut = sol.solPrim[-2:,3:]

		# characteristic variables
		w1Out 	= tempOut - pressOut / rhoCpMean
		w2Out 	= velOut + pressOut / rhoCMean
		w4Out 	= massFracOut 

		# extrapolate to exterior
		if (params.spaceOrder == 1):
			w1Bound = w1Out[0]
			w2Bound = w2Out[0]
			w4Bound = w4Out[0,:]
		elif (params.spaceOrder == 2):
			w1Bound = 2.0*w1Out[0] - w1Out[1]
			w2Bound = 2.0*w2Out[0] - w2Out[1]
			w4Bound = 2.0*w4Out[0,:] - w4Out[1,:]
		else:
			raise ValueError("Higher order extrapolation implementation required for spatial order "+str(params.spaceOrder))

		# compute exterior state
		pressBound 	= (w2Bound * rhoCMean + pressBack) / 2.0
		velBound 	= (pressBound - pressBack) / rhoCMean 
		tempBound 	= w1Bound + pressBound / rhoCpMean 
		massFracBound = w4Bound 

		OutBoundary[0, 0] = pressBound
		OutBoundary[0, 1] = velBound
		OutBoundary[0, 2] = tempBound
		OutBoundary[0, 3:] = massFracBound
  
		return OutBoundary

	def normalization(self, Array, center = False):

		if center:
			Array -= self.centProf
			Array /= self.normFacProf
		else:	
			Array /= self.normFacProf

		return Array

	def denormalization(self, Array, decenter = False):

		if decenter:
			Array += self.centProf
			Array *= self.normFacProf
		else:	
			Array *= self.normFacProf

		return Array


	