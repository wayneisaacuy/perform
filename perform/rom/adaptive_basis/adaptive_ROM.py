import os
import numpy as np
import scipy.linalg as LA

class AdaptROM():
    def __init__(self, solver, rom_domain, sol_domain):

        # attributes needed:
        # window of high-dim RHS
        
        # methods needed: update_residualSampling_window
        # adeim
        # initialize_window
        
        # this assumes vector construction of ROM
        # these initializations need to be changed for the scalar ROM case
        
        self.window = np.zeros((sol_domain.gas_model.num_eqs * sol_domain.mesh.num_cells, rom_domain.adaptiveROMWindowSize - 1))
        self.residual_samplepts = np.zeros(rom_domain.adaptiveROMnumResSample)
        self.residual_samplepts_comp = np.zeros(sol_domain.gas_model.num_eqs * sol_domain.mesh.num_cells - rom_domain.adaptiveROMnumResSample) # this is the complement

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
        tempWindow = tempWindow[:,1:]
        
        self.window = np.concatenate((tempWindow, NewState), axis=1)
        
    
    def update_residualSampling_window(self, rom_domain, solver, sol_domain, trial_basis, deim_idx_flat, decoded_ROM):
        # this updates the window and finds the sampling points (and its complement) for the residual
        
        # compute Q[:,k]
        #Q_k_temp = np.zeros((sol_domain.gas_model.num_eqs, sol_domain.mesh.num_cells))
        
        #for model_idx, model in enumerate(rom_domain.model_list):
            #Q_k_temp[model.var_idxs, :] = model.decode_sol(model.code)
            # extract flattened indices. only works for vector ROM!
            #deim_idx_flat = model.direct_samp_idxs_flat
            #trial_basis = model.trial_basis

        Q_k_temp = decoded_ROM
        Q_k = Q_k_temp.reshape((-1,1))
        
        if solver.time_iter == 1 or solver.time_iter % rom_domain.adaptiveROMUpdateFreq  == 0:
            
            # compute F[:,k]
            F_k = sol_domain.time_integrator.calc_fullydiscrhs(sol_domain, Q_k, solver)
            # F_k = F_k.reshape((-1,1))
            
            # update window
            if self.window.shape[1] == rom_domain.adaptiveROMWindowSize:
                self.cycle_window(F_k)
            else:
                self.window = np.concatenate((self.window, F_k), axis=1)
            
            # compute R_k
            R_k = self.window - (trial_basis @ np.linalg.pinv(trial_basis[deim_idx_flat, :]) @ self.window[deim_idx_flat , :])    
            
            # find s_k and \breve{s}_k
            sorted_idxs = np.argsort(-np.sum(R_k**2,axis=1))
            
            self.residual_samplepts = sorted_idxs[:rom_domain.adaptiveROMnumResSample]
            self.residual_samplepts_comp = sorted_idxs[rom_domain.adaptiveROMnumResSample:]

        else:
            
            F_k = np.zeros((sol_domain.gas_model.num_eqs * sol_domain.mesh.num_cells, 1))
            
            # take the union of s_k and p_k
            idx_union = np.concatenate((self.residual_samplepts, deim_idx_flat))
            idx_union = np.unique(idx_union)
            idx_union = np.sort(idx_union)
            
            # first evaluate the fully discrete RHS
            # note that the computationally efficient approach would be to only evaluate
            
            F_k[idx_union, :] = sol_domain.time_integrator.calc_fullydiscrhs(sol_domain, Q_k, solver, idx_union)
            
            # # extract components from Q_k
            # sampled_StateArg = np.zeros((sol_domain.gas_model.num_eqs * sol_domain.mesh.num_cells, 1))
            # sampled_StateArg[self.residual_samplepts,:] = Q_k[self.residual_samplepts,:]
            
            # now call the rhs function
            # F_k[self.residual_samplepts, :] = sol_domain.time_integrator.calc_fullydiscrhs(sol_domain, sampled_StateArg, solver, self.residual_samplepts)
            # F_k[self.residual_samplepts_comp, :] = trial_basis[self.residual_samplepts_comp, :] @ np.linalg.pinv(trial_basis[deim_idx_flat, :]) @ F_k[deim_idx_flat, :]
            
            F_k[self.residual_samplepts_comp, :] = trial_basis[self.residual_samplepts_comp, :] @ np.linalg.pinv(trial_basis[idx_union, :]) @ F_k[idx_union, :]

            # update window

            self.cycle_window(F_k)
            
    def adeim(self, rom_domain, trial_basis, deim_idx_flat, deim_dim, nMesh):
        
        r = rom_domain.adaptiveROMUpdateRank
        Fp = self.window[deim_idx_flat, :]
        FS = self.window[self.residual_samplepts, :]

        C, _, _, _ = np.linalg.lstsq(trial_basis[deim_idx_flat, :], Fp, rcond=None) # not sure if it should be solve or lstsq
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
        
        sampling_id = np.sort(sampling_id)
        
        return trial_basis, sampling_id
