import os
import numpy as np
import scipy.linalg as LA
#from scipy.linalg import orth


class AdaptROM():
    def __init__(self, model, solver, rom_domain, sol_domain):

        # attributes needed:
        # window of high-dim RHS
        
        # methods needed: update_residualSampling_window
        # adeim
        # initialize_window
        
        # this assumes vector construction of ROM
        self.window = np.zeros((4*sol_domain.mesh.num_cells, rom_domain.adaptiveROMWindowSize - 1))
        self.residual_samplepts = np.zeros(rom_domain.adaptiveROMnumResSample)
        self.residual_samplepts_comp = np.zeros(4*sol_domain.mesh.num_cells - rom_domain.adaptiveROMnumResSample) # this is the complement
        
        # self.adaptiveROMMethod = romDomain.adaptiveROMMethod
        # self.adaptsubIteration   = False

        # if self.adaptiveROMMethod == "OSAB":
        #     """
        #     Method developed by Prof. Karthik Duraisamy (University of Michigan)
        #     """
        #     #TODO: Implement residual sampling step
        #     self.adaptsubIteration              =   True
        #     self.trueStandardizedState          =   np.zeros((model.numVars, solver.mesh.numCells))
        #     self.adaptionResidual               =   np.zeros((model.numVars*solver.mesh.numCells))
        #     self.basisUpdate                    =   np.zeros(model.trialBasis.shape)

        #     self.adaptROMResidualSampStep       =   romDomain.adaptROMResidualSampStep
        #     self.adaptROMnumResSamp             =   romDomain.adaptROMnumResSamp

        # elif self.adaptiveROMMethod == "AADEIM":
        #     """
        #     Method developed by Prof. Benjamin Peherstorfer (NYU)
        #     """

        #     self.adaptROMResidualSampStep   =   romDomain.adaptROMResidualSampStep  #specifies the number of step after which full residual is computed
        #     self.adaptROMnumResSamp         =   romDomain.adaptROMnumResSamp        #specifies the number of samples of the residual
        #     self.adaptROMWindowSize         =   romDomain.adaptROMWindowSize        #look-back widow size
        #     self.adaptROMUpdateRank         =   romDomain.adaptROMUpdateRank        #basis upadate rank
        #     self.adaptROMInitialSnap        =   romDomain.adaptROMInitialSnap       #specifies the "number" of samples available (same as the number of samples used for basis computation)

        #     self.FWindow                    =   np.zeros((model.numVars, solver.mesh.numCells, self.adaptROMWindowSize)) #look-back window
        #     self.residualSampleIdx          =   [] #residual sample indexes (indexes fron all the states will be pooled together)

        #     self.interPolAdaptionWindow     =   []
        #     self.interPolBasis              =   []
        #     self.scaledAdaptionWindow       =   np.zeros((model.numVars*solver.mesh.numCells, self.adaptROMWindowSize))


        #     assert(self.adaptROMWindowSize-1<=self.adaptROMInitialSnap), 'Look back window size minus 1 should be less than equal to the available stale states'

        # else:
        #     raise ValueError("Invalid selection for adaptive ROM type")
        
    def init_window(self, rom_domain):
        # this has to be done for every model in model list
        # initializes the window
        model_dir = rom_domain.model_dir
        
        try:
            temp_window = np.load(os.path.join(model_dir, rom_domain.adaptiveROMFOMfile))
            temp_window = temp_window[:,:,:rom_domain.adaptiveROMWindowSize-1]
            self.window = np.reshape(temp_window, (-1, temp_window.shape[-1]), order="C")

        except:
            raise Exception("File for snapshots not found")
    
    def cycle_window(self, NewState):
        """ Cycles the window and add new state
        """
        
        # nCols = self.window.shape[1]
        tempWindow = self.window.copy()
        tempWindow = tempWindow[:,:-1]
        
        self.window = np.concatenate((NewState,tempWindow), axis=1)
        
    
    def update_residualSampling_window(self, solver, rom_domain, sol_domain):
        # this updates the window and finds the sampling points (and its complement) for the residual
        
        # compute Q[:,k]
        Q_k_temp = np.zeros((sol_domain.gas_model.num_eqs, sol_domain.mesh.num_cells))
        
        for model_idx, model in enumerate(rom_domain.model_list):
            Q_k_temp[model.var_idxs, :] = model.decode_sol(model.code)
            # extract flattened indices. only works for vector ROM!
            deim_idx_flat = model.direct_samp_idxs_flat
            trial_basis = model.trial_basis
        
        Q_k = Q_k_temp.reshape((-1,1))
        
        if solver.time_iter == 1 or solver.time_iter % rom_domain.adaptiveROMUpdateFreq  == 0:
            
            # compute F[:,k]
            F_k = sol_domain.time_integrator.calc_fullydiscrhs(sol_domain, Q_k, solver)
            # F_k = F_k.reshape((-1,1))
            
            # update window
            if self.window.shape[1] == rom_domain.adaptiveROMWindowSize:
                self.cycle_window(F_k)
            else:
                self.window = np.concatenate((F_k,self.window), axis=1)
            
            # compute R_k
            R_k = self.window - (trial_basis @ np.linalg.pinv(trial_basis[deim_idx_flat, :]) @ self.window[deim_idx_flat , :])    
            
            # find s_k and \breve{s}_k
            sorted_idxs = np.argsort(-np.sum(R_k**2,axis=1))
            
            self.residual_samplepts = sorted_idxs[:rom_domain.adaptiveROMnumResSample]
            self.residual_samplepts_comp = sorted_idxs[rom_domain.adaptiveROMnumResSample:]
            
        else:
            
            F_k = np.zeros((sol_domain.gas_model.num_eqs * sol_domain.mesh.num_cells, 1))
            
            # extract components from Q_k
            sampled_StateArg = np.zeros((sol_domain.gas_model.num_eqs * sol_domain.mesh.num_cells, 1))
            sampled_StateArg[self.residual_samplepts,:] = Q_k[self.residual_samplepts,:]
            
            # now call the rhs function
            F_k[self.residual_samplepts, :] = sol_domain.time_integrator.calc_fullydiscrhs(sol_domain, sampled_StateArg, solver, self.residual_samplepts)
            F_k[self.residual_samplepts_comp, :] = trial_basis[self.residual_samplepts_comp, :] @ np.linalg.pinv(trial_basis[deim_idx_flat, :]) @ F_k[deim_idx_flat, :]
            
            # update window

            self.cycle_window(F_k)
            
    def adeim(self, trial_basis, deim_idx_flat, deim_dim, nMesh, rom_domain):
        
        r = rom_domain.adaptiveROMUpdateRank
        Fp = self.window[deim_idx_flat, :]
        FS = self.window[self.residual_samplepts, :]
            
        C = np.linalg.lstsq(trial_basis[deim_idx_flat, :], Fp) # not sure if it should be solve or lstsq
        R = trial_basis[self.residual_samplepts, :] @ C - FS
        
        _, Sv, Srh = np.linalg.svd(R)
        Sr = Srh.T
        
        CT_pinv = np.linalg.pinv(C.T)
        
        r = min(r, len(Sv))
        
        for i in range(r):
            alfa = -R @ Sr[:, i:i+1]
            beta = CT_pinv @ Sr[:, i:i+1]
            trial_basis[self.residual_samplepts, :] = trial_basis[self.residual_samplepts, :] + alfa @ beta.T
            
        # orthogonalize basis
            
        trial_basis, _ = np.linalg.qr(trial_basis)    
        
        # apply qdeim
            
        _, _, sampling = LA.qr(trial_basis.T, pivoting=True)
        sampling_trunc = sampling[:deim_dim]
        
        # take modulo of deim sampling points 
        sampling_id = np.remainder(sampling_trunc, nMesh)
        sampling_id = np.unique(sampling_id)
        
        ctr = 0
        while sampling_id.shape[0] < deim_dim:
            # get the next sampling index
            sampling_id = np.append(sampling_id, sampling[deim_dim + ctr])
        
            # ensure all entries are unique
            sampling_id = np.unique(sampling_id)
            ctr = ctr + 1
        
        return trial_basis, sampling_id

    def initializeLookBackWindow(self, romDomain, model):

        self.FWindow[:, :, :-1] = romDomain.staleConsSnapshots[model.varIdxs, :, -(self.adaptROMWindowSize-1):]

    def initializeHistory(self, romDomain, solDomain, solver, model):
        '''Computes the coded state and the reconstructed state for initializing the sub-iteration'''

        timeOrder = min(solver.iter, romDomain.timeIntegrator.timeOrder)  # cold start
        timeOrder = max(romDomain.timeIntegrator.staleStatetimeOrder, timeOrder)  # updating time order if stale states are available

        for timeIdx in range(timeOrder + 1):
            #update the coded history
            solCons = solDomain.solInt.solHistCons[timeIdx].copy()
            solCons = model.standardizeData(solCons[model.varIdxs, :], normalize=True,
                                            normFacProf=model.normFacProfCons, normSubProf=model.normSubProfCons,
            								   center=True, centProf=model.centProfCons, inverse=False)
            model.codeHist[timeIdx] = model.projectToLowDim(model.trialBasis, solCons, transpose=True)

        model.code = model.codeHist[0].copy()
        model.updateSol(solDomain)


    def gatherSamplePoints(self, romDomain, solDomain, solver, model):

        if not romDomain.timeIntegrator.timeType == "implicit": raise ValueError('AADEIM not implemented for explicit framework')

        if (solver.timeIter == 1 or solver.timeIter % self.adaptROMResidualSampStep == 0):

            self.FWindow[:, :, -1] = self.previousStateEstimate(romDomain, solDomain, solver, model)

            # adaption window
            adaptionWindow = self.FWindow.reshape(-1, self.adaptROMWindowSize, order='C')
            self.scaledAdaptionWindow = (adaptionWindow - model.centProfCons.ravel(order='C')[:,None] - model.normSubProfCons.ravel(order='C')[:,None]) / model.normFacProfCons.ravel(order='C')[:, None]

            self.interPolAdaptionWindow = (self.scaledAdaptionWindow.reshape(model.numVars, solver.mesh.numCells, -1, order = "C")[:,solDomain.directSampIdxs,:]).reshape(-1, self.adaptROMWindowSize, order = 'C')
            self.interPolBasis          = (model.trialBasis.reshape(model.numVars, -1, model.latentDim)[:,solDomain.directSampIdxs,:]).reshape(-1, model.latentDim, order = "C")

            # residual
            reconstructedWindow = model.trialBasis @ np.linalg.pinv(self.interPolBasis) @ self.interPolAdaptionWindow
            residual = (self.scaledAdaptionWindow - reconstructedWindow).reshape((model.numVars, solver.mesh.numCells, -1), order='C')
            SampleIdx = np.unique((np.argsort(np.sum(residual ** 2, axis=2), axis=1)[:, -self.adaptROMnumResSamp:]).ravel())

            romDomain.residualCombinedSampleIdx = np.unique(np.append(romDomain.residualCombinedSampleIdx, SampleIdx).astype(int))

        else:
            raise ValueError('Reduced frequency residual update not implemented (modified - AADEIM)')

    def adaptModel(self, romDomain, solDomain, solver, model):

        if romDomain.adaptiveROMMethod == "OSAB":
            if romDomain.timeIntegrator.timeType == "implicit" : raise ValueError('One step adaptive basis not implemented for implicit framework')

            self.adaptionResidual = (self.trueStandardizedState - model.applyTrialBasis(model.code)).flatten(order = "C").reshape(-1, 1)
            self.basisUpdate = np.dot(self.adaptionResidual, model.code.reshape(1, -1)) / np.linalg.norm(model.code)**2

            model.trialBasis = model.trialBasis + self.basisUpdate

            model.updateSol(solDomain)

            solDomain.solInt.updateState(fromCons=True)

        elif romDomain.adaptiveROMMethod == "AADEIM":
            self.residualSampleIdx = romDomain.residualCombinedSampleIdx
            reshapedWindow = (self.scaledAdaptionWindow).reshape(model.numVars, -1,  self.adaptROMWindowSize, order = "C")
            sampledAdaptionWindow         = (reshapedWindow[:,self.residualSampleIdx,:]).reshape(-1, self.adaptROMWindowSize, order = "C")
            sampledBasis                  = (model.trialBasis.reshape(model.numVars, -1, model.latentDim)[:,self.residualSampleIdx,:]).reshape(-1, model.latentDim, order = "C")

            #Computing coefficient matrix
            CMat = np.linalg.pinv(self.interPolBasis) @ self.interPolAdaptionWindow
            pinvCtranspose = np.linalg.pinv(CMat.T)

            #Computing residual
            R = sampledBasis @ CMat - sampledAdaptionWindow

            # Computing SVD
            _, singValues, rightBasis_h = np.linalg.svd(R)
            rightBasis = rightBasis_h.T

            rank = min(self.adaptROMUpdateRank, len(singValues))

            # Basis Update
            idx = (np.arange(model.numVars*solver.mesh.numCells).reshape(model.numVars, -1, order = "C")[:, self.residualSampleIdx]).ravel(order = "C")
            for irank in range(rank):
                alpha = -R @ rightBasis[:, irank]
                beta = pinvCtranspose @ rightBasis[:, irank]
                update = alpha[:, None] @ beta[None, :]
                # model.trialBasis[idx, :] +=  update

            # model.trialBasis = orth( model.trialBasis)

            self.FWindow[:, :, :-1] = self.FWindow[:, :, 1:]

    def previousStateEstimate(self, romDomain, solDomain, solver, model):

        solInt = solDomain.solInt

        timeOrder = min(solver.iter, romDomain.timeIntegrator.timeOrder)  # cold start
        timeOrder = max(romDomain.timeIntegrator.staleStatetimeOrder, timeOrder)  # updating time order if stale states are available

        coeffs = romDomain.timeIntegrator.coeffs[timeOrder - 1]

        state = coeffs[0] * solInt.solHistCons[0][model.varIdxs, :].copy()

        for iterIdx in range(2, timeOrder + 1):
            state += coeffs[iterIdx] * solInt.solHistCons[iterIdx][model.varIdxs, :].copy()

        PreviousState = (- state + (romDomain.timeIntegrator.dt*solInt.RHS[model.varIdxs, :])) / coeffs[1]

        return PreviousState