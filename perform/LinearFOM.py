import numpy as np
import matplotlib.pyplot as plt
import copy
import constants
from math import sin, pi
from numpy.linalg import svd
from perform.solution.solutionBoundary.solutionInlet import solutionInlet 
from perform.solution.solutionBoundary.solutionOutlet import solutionOutlet
from Jacobians import calcDSourceDSolPrim, calcDSolPrimDSolCons, calcDFluxDSolPrim

import time
import os
import sys
import pdb


class linearization:

	#def __init__(self, params: parameters):

		#self.NonlinearSol=np.load(os.path.join(params.unsOutDir, "solPrim_LinFOM.npy"), allow_pickle=True) #(256, 4, 1001)

	def initLinear(self, solDomain, solver):

		calcBoundaries(sol, bounds, params, gas)

		gas = solDomain.solInt.gasModel

		gamma_matrix_inv = calcDSolPrimDSolCons(solInt)
		dSdQp = np.zeros((solDomain.gasModel.numEqs, solDomain.gasModel.numEqs, solver.mesh.numCells), dtype=constants.realType)
		if params.sourceOn:
			dSdQp = calcDSourceDSolPrim(solDomain.timeIntegrator.dt)

		dFdQp, dFdQp_l, dFdQp_r = calcDRoeFluxDSolPrim(solDomain, solver, linRHS = True)

		if (solDomain.timeIntegrator.timeType == "explicit"): 
			dFdQp = np.transpose(gamma_matrix_inv,(2, 0, 1)) @ np.transpose(dFdQp, (2, 0, 1))
			self.dFlxdQpR = np.transpose(gamma_matrix_inv,(2, 0, 1)) @ np.transpose(dFdQp_r, (2, 0, 1))
			self.dFlxdQpL = np.transpose(gamma_matrix_inv,(2, 0, 1)) @ np.transpose(dFdQp_l, (2, 0, 1))
			if params.sourceOn:
				dSdQp = np.transpose(gamma_matrix_inv,(2, 0, 1)) @ np.transpose(dSdQp, (2, 0, 1))
				self.dFPlusdS = dFdQp + dSdQp
			else:
				self.dFPlusdS = dFdQp

		if params.initFromRestart:
			solver.solTime, solDomain.solInt.solPrim = readRestartFile(params.restOutDir)
		else:
			solDomain.solInt.solPrim = np.zeros((solver.mesh.numCells, solDomain.gasModel.numEqs), dtype = constants.realType)


	def linearRHS(self, solDomain, solver):

		gas = solDomain.solInt.gasModel
		solInt = solDomain.solInt.solPrim

		InBoundState = self.calcLinearInlet(solver, solInt, gas)
		OutBoundState = self.calcLinearOutlet(solver, solInt, gas)

		solPrimL = np.concatenate((InBoundState, solInt), axis=0)  
		solPrimR = np.concatenate((solInt, OutBoundState), axis=0)

		linRHS = self.dFPlusdS @ np.expand_dims(solInt, axis=2) 
		linRHS = linRHS + self.dFlxdQpR @ np.expand_dims(solPrimR[1:,:], axis=2) 
		linRHS = linRHS + self.dFlxdQpL @ np.expand_dims(solPrimL[:-1,:], axis=2)
		solDomain.solInt.RHS = np.squeeze(linRHS, axis=2)
		#import sys
		#sys.exit()

	def calcLinearInlet(self, solver, solInt, gas):

		inlet = solutionInlet(gas, solver)
		

		InBoundary = np.zeros((1, solDomain.gasModel.numEqs))	
	#	pressUp 	= inlet.press 
	#	tempUp 		= inlet.temp
		massFracUp	= inlet.massFrac[:-1]
		rhoCMean 	= inlet.vel 
		rhoCpMean 	= inlet.rho

		if (inlet.pertType == "pressure"):
			pressUp *= inlet.calcPert(solver.solTime)

		# interior quantities
		pressIn 	= solInt[:2,0]
		velIn 		= solInt[:2,1]

		# characteristic variables
		w3In 	= velIn - pressIn / rhoCMean  

			# extrapolate to exterior
		if (params.spaceOrder == 1):
			w3Bound = w3In[0]
		elif (params.spaceOrder == 2):
			w3Bound = 2.0*w3In[0] - w3In[1]
		else:
			raise ValueError("Higher order extrapolation implementation required for spatial order "+str(solver.spaceOrder))

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

	def calcLinearOutlet(self, solver, solInt, gas):

		# specify rho*C and rho*Cp from mean solution, back pressure is static pressure at infinity
		outlet = solutionOutlet(gas, solver)

		OutBoundary = np.zeros((1, solDomain.gasModel.numEqs))
		rhoCMean 	= outlet.vel 
		rhoCpMean 	= outlet.rho
		pressBack 	= outlet.press 

		if (outlet.pertType == "pressure"):
			pressBack *= outlet.calcPert(solver.solTime)  

		# interior quantities
		pressOut 	= solInt[-2:,0]
		velOut 		= solInt[-2:,1]
		tempOut 	= solInt[-2:,2]
		massFracOut = solInt[-2:,3:]

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
			raise ValueError("Higher order extrapolation implementation required for spatial order "+str(solver.spaceOrder))

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













