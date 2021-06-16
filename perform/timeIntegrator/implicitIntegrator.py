import perform.constants as const
from perform.inputFuncs import catchInput
from perform.timeIntegrator.timeIntegrator import timeIntegrator

import numpy as np
import pdb

class implicitIntegrator(timeIntegrator):
	"""
	Base class for implicit time integrators
	Solves implicit system via Newton's method
	"""

	def __init__(self, paramDict):
		super().__init__(paramDict)
		self.timeType 		= "implicit"
		self.subiterMax		= catchInput(paramDict, "subiterMax", const.subiterMaxImpDefault)
		self.resTol 		= catchInput(paramDict, "resTol", const.l2ResTolDefault)

		# dual time-stepping, robustness controls
		self.dualTime 		= catchInput(paramDict, "dualTime", True)
		self.dtau 			= catchInput(paramDict, "dtau", const.dtauDefault)
		if (self.dualTime):
			self.adaptDTau 	= catchInput(paramDict, "adaptDTau", False)
		else:
			self.adaptDTau 	= False
		self.CFL 			= catchInput(paramDict, "CFL", const.CFLDefault) 	# reference CFL for advective control of dtau
		self.VNN 			= catchInput(paramDict, "VNN", const.VNNDefault) 	# von Neumann number for diffusion control of dtau
		self.refConst 		= catchInput(paramDict, "refConst", [None])  			# constants for limiting dtau	
		self.relaxConst 	= catchInput(paramDict, "relaxConst", [None]) 			#


class bdf(implicitIntegrator):
	"""
	Backwards difference formula (up to fourth-order)
	"""

	def __init__(self, paramDict):
		super().__init__(paramDict)

		self.coeffs = [None]*4
		self.coeffs[0] = np.array([1.0, -1.0], dtype=const.realType)
		self.coeffs[1] = np.array([1.5, -2.0, 0.5], dtype=const.realType)
		self.coeffs[2] = np.array([11./16., -3.0, 1.5, -1./3.], dtype=const.realType)
		self.coeffs[3] = np.array([25./12., -4.0, 3.0, -4./3., 0.25], dtype=const.realType)
		assert (self.timeOrder <= 4), str(self.timeOrder)+"th-order accurate scheme not implemented for "+self.timeScheme+" scheme"


	def calcResidual(self, solHist, rhs, solver):
		
		timeOrder = min(solver.iter, self.timeOrder) 	# cold start
		timeOrder = max(self.staleStatetimeOrder, timeOrder) # updating time order if stale states are available

		coeffs = self.coeffs[timeOrder-1]

		residual = coeffs[0] * solHist[0]
		for iterIdx in range(1, timeOrder+1):
			residual += coeffs[iterIdx] * solHist[iterIdx]
		
		residual = -(residual / self.dt) + rhs

		return residual

	def calcReducedResidualVec(self, romDomain, solDomain, solver):

		timeOrder = min(solver.iter, self.timeOrder) 	# cold start
		timeOrder = max(self.staleStatetimeOrder, timeOrder) # updating time order if stale states are available
		resVec = np.zeros(np.sum(romDomain.latentDims))

		coeffs = self.coeffs[timeOrder - 1]

		idx = np.insert(np.cumsum(romDomain.latentDims), 0, 0)

		for modelIdx, model in enumerate(romDomain.modelList):
			#TODO: Idx for any combination of the models
			residual = coeffs[0] * model.codeHist[0]
			model.calcRHSLowDim(romDomain, solDomain)

			for iterIdx in range(1, timeOrder + 1):
				residual += coeffs[iterIdx] * model.codeHist[iterIdx]
			residual = -(residual / self.dt) + model.rhsLowDim
			resVec[idx[modelIdx]:idx[modelIdx+1]] = residual
		return resVec








