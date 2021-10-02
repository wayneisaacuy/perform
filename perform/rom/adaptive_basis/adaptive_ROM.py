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
        
        #self.window = np.zeros((sol_domain.gas_model.num_eqs * sol_domain.mesh.num_cells, rom_domain.adaptiveROMWindowSize - 1))
        self.window = np.zeros((sol_domain.gas_model.num_eqs * sol_domain.mesh.num_cells, 0))
        self.residual_samplepts = np.zeros(rom_domain.adaptiveROMnumResSample)
        self.residual_samplepts_comp = np.zeros(sol_domain.gas_model.num_eqs * sol_domain.mesh.num_cells - rom_domain.adaptiveROMnumResSample) # this is the complement
        self.FOM_snapshots = np.array([])
        self.FOM_snapshots_prim = np.array([])
        self.FOM_snapshots_scaled = np.array([])
        self.rel_proj_err = np.array([])
        self.rel_proj_err_origspace = np.array([])
        self.rel_proj_err_states = np.zeros((sol_domain.gas_model.num_eqs, 0))
        self.rel_proj_err_origspace_states = np.zeros((sol_domain.gas_model.num_eqs, 0))
        #self.rhs_FOM_diff = np.zeros((sol_domain.gas_model.num_eqs * sol_domain.mesh.num_cells, 0))
        
        self.rhs_FOM_diff = []
        
        self.rel_proj_err_origspace_prim = np.array([])
        self.rel_proj_err_states_prim = np.zeros((sol_domain.gas_model.num_eqs, 0))
        
        self.basis_inc = np.array([])
        
        self.sing_val = []
        self.sing_val_states = []
        
        #self.denom_norm = np.array([])
        
    def save_debugstats(self, rom_domain, dt):
        
        # convert singular values list to array
        self.sing_val = np.asarray(self.sing_val)
        self.sing_val_states = np.asarray(self.sing_val_states)
        
        model_dir = rom_domain.model_dir
        
        fname_param_relprojerr = "unsteady_field_results/relprojerr" + rom_domain.param_string + "_dt_" + str(dt) + ".npz"
        
        # fname_relprojerr = os.path.join(model_dir, "unsteady_field_results/relprojerr.npz")
        fname_relprojerr = os.path.join(model_dir, fname_param_relprojerr)
        np.savez(fname_relprojerr, relprojerr_scaled = self.rel_proj_err, relprojerr_origspace = self.rel_proj_err_origspace, 
                 relprojerr_scaled_states = self.rel_proj_err_states, relprojerr_origspace_states = self.rel_proj_err_origspace_states,
                 rel_proj_err_origspace_prim = self.rel_proj_err_origspace_prim, rel_proj_err_states_prim = self.rel_proj_err_states_prim,
                 basis_inc = self.basis_inc)
        
        fname_param_singval = "unsteady_field_results/singval" + rom_domain.param_string + "_dt_" + str(dt) + ".npz"
        fname_singval = os.path.join(model_dir, fname_param_singval)
        np.savez(fname_singval, sing_val = self.sing_val, sing_val_states = self.sing_val_states, init_singval = rom_domain.init_singval, init_singval_states = rom_domain.init_singval_states)
        
        fname_rhsFOMdiff = os.path.join(model_dir, "unsteady_field_results/rhsFOMdiff" +  rom_domain.param_string + "_dt_" + str(dt))
        np.save(fname_rhsFOMdiff, np.asarray(self.rhs_FOM_diff))
        
        # fname_FOMsolnorm = os.path.join(model_dir, "unsteady_field_results/FOMsolNorm")
        # np.save(fname_FOMsolnorm, self.denom_norm)
        
    def compute_relprojerr(self, decoded_ROM, solver, sol_domain, model, prim_sol):
        # compute relative projection error
        
        if solver.time_iter % solver.out_interval == 0:
            decoded_ROM_origspace = decoded_ROM.copy()
            decoded_ROM_origspace = decoded_ROM_origspace.reshape((-1,))
            
            curr_prim_sol = prim_sol.copy()
            curr_prim_sol = curr_prim_sol.reshape((-1,))
            
            # need to scale decoded ROM
            # decoded_ROM = model.scale_profile(decoded_ROM,
            #                     normalize=True,
            #                     norm_fac_prof=model.norm_fac_prof_cons,
            #                     norm_sub_prof=model.norm_sub_prof_cons,
            #                     center=True,
            #                     cent_prof=model.cent_prof_cons,
            #                     inverse=False,
            #                     )
            
            # decoded_ROM = decoded_ROM.reshape((-1,1))
            # decoded_ROM = decoded_ROM[:,0]
            
            FOM_sol = self.FOM_snapshots_scaled[:,solver.time_iter]
            FOM_sol_origspace = self.FOM_snapshots[:,solver.time_iter]
            FOM_sol_prim = self.FOM_snapshots_prim[:,solver.time_iter]
            
            # FOM_sol = FOM_sol.reshape((sol_domain.gas_model.num_eqs, sol_domain.mesh.num_cells), order="C")
            
            # # need to scale FOM
            # FOM_sol = model.scale_profile(FOM_sol, normalize=True,
            #                               norm_fac_prof=model.norm_fac_prof_cons,
            #                               norm_sub_prof=model.norm_sub_prof_cons,
            #                               center=True,
            #                               cent_prof=model.cent_prof_cons,
            #                               inverse=False,
            #                               )
            # FOM_sol = FOM_sol.reshape((-1,1))
            # FOM_sol = FOM_sol[:,0]
            
            # try something else
    
            reprojROM =  model.trial_basis @ model.code
            proj_err = LA.norm(FOM_sol - reprojROM)/LA.norm(FOM_sol)
            proj_err_origspace = LA.norm(FOM_sol_origspace - decoded_ROM_origspace)/LA.norm(FOM_sol_origspace)
            proj_err_prim = LA.norm(FOM_sol_prim - curr_prim_sol)/LA.norm(FOM_sol_prim)
    
            #proj_err = LA.norm(FOM_sol - decoded_ROM)/LA.norm(FOM_sol)
            self.rel_proj_err = np.concatenate((self.rel_proj_err, np.array([proj_err])))
            self.rel_proj_err_origspace = np.concatenate((self.rel_proj_err_origspace, np.array([proj_err_origspace])))
            self.rel_proj_err_origspace_prim = np.concatenate((self.rel_proj_err_origspace_prim, np.array([proj_err_prim])))
            #self.denom_norm = np.concatenate((self.denom_norm, np.array([LA.norm(FOM_sol)])))
            
            proj_err_states = np.zeros((sol_domain.gas_model.num_eqs, 1))
            proj_err_origspace_states = np.zeros((sol_domain.gas_model.num_eqs, 1))
            proj_err_states_prim = np.zeros((sol_domain.gas_model.num_eqs, 1))
            
            nMesh = sol_domain.mesh.num_cells
            for idx in range(sol_domain.gas_model.num_eqs):
                proj_err_states[idx,:] = LA.norm(FOM_sol[idx*nMesh:(idx+1)*nMesh] - reprojROM[idx*nMesh:(idx+1)*nMesh])/LA.norm(FOM_sol[idx*nMesh:(idx+1)*nMesh])
                proj_err_origspace_states[idx,:] = LA.norm(FOM_sol_origspace[idx*nMesh:(idx+1)*nMesh] - decoded_ROM_origspace[idx*nMesh:(idx+1)*nMesh])/LA.norm(FOM_sol_origspace[idx*nMesh:(idx+1)*nMesh])
                proj_err_states_prim[idx,:] = LA.norm(FOM_sol_prim[idx*nMesh:(idx+1)*nMesh] - curr_prim_sol[idx*nMesh:(idx+1)*nMesh])/LA.norm(FOM_sol_prim[idx*nMesh:(idx+1)*nMesh])
            
            self.rel_proj_err_states = np.concatenate((self.rel_proj_err_states, proj_err_states), axis = 1)
            self.rel_proj_err_origspace_states = np.concatenate((self.rel_proj_err_origspace_states, proj_err_origspace_states), axis = 1)
            self.rel_proj_err_states_prim = np.concatenate((self.rel_proj_err_states_prim, proj_err_states_prim), axis = 1)
        
        else:
            pass
                
    def load_FOM(self, rom_domain, model):
        # this has to be done for every model in model list
        # initializes the window
        model_dir = rom_domain.model_dir
        
        # conservative variables
        
        try:
            FOM_snap = np.load(os.path.join(model_dir, rom_domain.adaptiveROMFOMfile))
            self.FOM_snapshots = np.reshape(FOM_snap, (-1, FOM_snap.shape[-1]), order="C")
            FOM_snap_scaled = np.zeros_like(FOM_snap)
            nSnaps = FOM_snap_scaled.shape[-1]

            # scale snapshot
            
            for i in range(nSnaps):
                FOM_snap_scaled[:, :, i] = model.scale_profile(
                                                    FOM_snap[:, :, i],
                                                    normalize=True,
                                                    norm_fac_prof=model.norm_fac_prof_cons,
                                                    norm_sub_prof=model.norm_sub_prof_cons,
                                                    center=True,
                                                    cent_prof=model.cent_prof_cons,
                                                    inverse=False,
                                                    )

            self.FOM_snapshots_scaled = np.reshape(FOM_snap_scaled, (-1, FOM_snap_scaled.shape[-1]), order="C")

        except:
            raise Exception("File for snapshots in conservative variables not found")
        
        # primitive variables
        
        try:
            FOM_snap_prim = np.load(os.path.join(model_dir, rom_domain.adaptiveROMFOMprimfile))
            self.FOM_snapshots_prim = np.reshape(FOM_snap_prim, (-1, FOM_snap_prim.shape[-1]), order="C")
            
        except:
            raise Exception("File for snapshots in primitive variables not found")

            
    # def init_window(self, rom_domain, model):
    #     # this has to be done for every model in model list
    #     # initializes the window
    #     model_dir = rom_domain.model_dir
        
    #     try:
    #         temp_window = np.load(os.path.join(model_dir, rom_domain.adaptiveROMFOMfile))
    #         temp_window = temp_window[:,:,:rom_domain.adaptiveROMWindowSize-1]
            
    #         temp_window_scaled = np.zeros_like(temp_window)
    #         nSnaps = temp_window_scaled.shape[-1]
            
    #         # scale snapshot
            
    #         for i in range(nSnaps):
    #             temp_window_scaled[:, :, i] = model.scale_profile(
    #                                                 temp_window[:, :, i],
    #                                                 normalize=True,
    #                                                 norm_fac_prof=model.norm_fac_prof_cons,
    #                                                 norm_sub_prof=model.norm_sub_prof_cons,
    #                                                 center=True,
    #                                                 cent_prof=model.cent_prof_cons,
    #                                                 inverse=False,
    #                                                 )
            
    #         self.window = np.reshape(temp_window_scaled, (-1, temp_window_scaled.shape[-1]), order="C")

    #     except:
    #         raise Exception("File for snapshots not found")
    
    def cycle_window(self, NewState):
        """ Cycles the window and add new state
        """
        
        # nCols = self.window.shape[1]
        tempWindow = self.window.copy()
        tempWindow = tempWindow[:,1:]
        
        self.window = np.concatenate((tempWindow, NewState), axis=1)
        
    
    def update_residualSampling_window(self, rom_domain, solver, sol_domain, trial_basis, deim_idx_flat, decoded_ROM, model, use_FOM, debugROM):
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
        #breakpoint()
        if solver.time_iter == 1 or solver.time_iter % rom_domain.adaptiveROMUpdateFreq  == 0:
            
            # compute F[:,k]
            if use_FOM == 1:
                F_k = self.FOM_snapshots[:,solver.time_iter-1:solver.time_iter].copy()
            elif use_FOM == 0:
                F_k = sol_domain.time_integrator.calc_fullydiscrhs(sol_domain, Q_k, solver, rom_domain)
            elif use_FOM == 2:
                FOM_qk = self.FOM_snapshots[:,solver.time_iter:solver.time_iter+1].copy()
                F_k = sol_domain.time_integrator.calc_fullydiscrhs(sol_domain, FOM_qk, solver, rom_domain)
            
            F_k_copy = F_k.copy()
            
            # scale snapshot
            F_k = F_k.reshape((sol_domain.gas_model.num_eqs, sol_domain.mesh.num_cells), order="C")
            F_k = model.scale_profile(F_k, normalize=True,
                                      norm_fac_prof=model.norm_fac_prof_cons,
                                      norm_sub_prof=model.norm_sub_prof_cons,
                                      center=True,
                                      cent_prof=model.cent_prof_cons,
                                      inverse=False,
                                      )
            F_k = F_k.reshape((-1,1))

            if debugROM and solver.time_iter % solver.out_interval == 0:
                # rhs_FOM_diff = self.FOM_snapshots_scaled[:,solver.time_iter-1:solver.time_iter] - F_k
                # self.rhs_FOM_diff = np.concatenate((self.rhs_FOM_diff,rhs_FOM_diff), axis = 1)
                rhs_FOM_diff = np.linalg.norm(self.FOM_snapshots[:,solver.time_iter-1:solver.time_iter] - F_k_copy,'fro')/np.linalg.norm(self.FOM_snapshots[:,solver.time_iter-1:solver.time_iter],'fro')
                self.rhs_FOM_diff.append(rhs_FOM_diff)
                
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
            # note that the computationally efficient approach would be to only evaluate the right hand side at select components
            
            # F_k[idx_union, :] = sol_domain.time_integrator.calc_fullydiscrhs(sol_domain, Q_k, solver, idx_union)
            
            # inefficient. first evaluate right hand side at all components and only select those components needed
            if use_FOM == 1:
                temp_F_k = self.FOM_snapshots[:,solver.time_iter-1:solver.time_iter] 
            elif use_FOM == 0:
                temp_F_k = sol_domain.time_integrator.calc_fullydiscrhs(sol_domain, Q_k, solver, rom_domain)
            elif use_FOM == 2:
                FOM_qk = self.FOM_snapshots[:,solver.time_iter:solver.time_iter+1].copy()
                temp_F_k = sol_domain.time_integrator.calc_fullydiscrhs(sol_domain, FOM_qk, solver, rom_domain)
            
            temp_F_k_copy = temp_F_k.copy()
            
            # scale snapshot
            temp_F_k = temp_F_k.reshape((sol_domain.gas_model.num_eqs, sol_domain.mesh.num_cells), order="C")
            temp_F_k = model.scale_profile(temp_F_k, normalize=True,
                                      norm_fac_prof=model.norm_fac_prof_cons,
                                      norm_sub_prof=model.norm_sub_prof_cons,
                                      center=True,
                                      cent_prof=model.cent_prof_cons,
                                      inverse=False,
                                      )
            temp_F_k = temp_F_k.reshape((-1,1))
            F_k[idx_union, :] = temp_F_k[idx_union, :]
            
            # # extract components from Q_k
            # sampled_StateArg = np.zeros((sol_domain.gas_model.num_eqs * sol_domain.mesh.num_cells, 1))
            # sampled_StateArg[self.residual_samplepts,:] = Q_k[self.residual_samplepts,:]
            
            # now call the rhs function
            # F_k[self.residual_samplepts, :] = sol_domain.time_integrator.calc_fullydiscrhs(sol_domain, sampled_StateArg, solver, self.residual_samplepts)
            # F_k[self.residual_samplepts_comp, :] = trial_basis[self.residual_samplepts_comp, :] @ np.linalg.pinv(trial_basis[deim_idx_flat, :]) @ F_k[deim_idx_flat, :]
            
            F_k[self.residual_samplepts_comp, :] = trial_basis[self.residual_samplepts_comp, :] @ np.linalg.pinv(trial_basis[idx_union, :]) @ F_k[idx_union, :]

            if debugROM and solver.time_iter % solver.out_interval == 0:
                # rhs_FOM_diff = self.FOM_snapshots_scaled[:,solver.time_iter-1:solver.time_iter] - F_k
                # self.rhs_FOM_diff = np.concatenate((self.rhs_FOM_diff,rhs_FOM_diff), axis = 1)
                rhs_FOM_diff = np.linalg.norm(self.FOM_snapshots[:,solver.time_iter-1:solver.time_iter] - temp_F_k_copy,'fro')/np.linalg.norm(self.FOM_snapshots[:,solver.time_iter-1:solver.time_iter],'fro')
                self.rhs_FOM_diff.append(rhs_FOM_diff)

            # update window
            if self.window.shape[1] == rom_domain.adaptiveROMWindowSize:
                self.cycle_window(F_k)
            else:
                self.window = np.concatenate((self.window, F_k), axis=1)

    def adeim(self, rom_domain, trial_basis, deim_idx_flat, deim_dim, nMesh, solver):
        
        old_basis = trial_basis.copy()
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
            sampling_id = np.append(sampling_id, np.remainder(sampling[deim_dim + ctr], nMesh))

            # ensure all entries are unique
            sampling_id = np.unique(sampling_id)
            ctr = ctr + 1
        
        sampling_id = np.sort(sampling_id)
        
        if solver.time_iter % solver.out_interval == 0:
            basis_change = np.linalg.norm(old_basis - trial_basis @ trial_basis.T @ old_basis, 'fro')/np.linalg.norm(old_basis, 'fro')
            self.basis_inc = np.concatenate((self.basis_inc, np.array([basis_change])))

        return trial_basis, sampling_id
    
    def PODbasis(self, deim_dim, nMesh, old_basis, solver):
        
        # compute basis using POD from snapshots
        U, Sv, _ = np.linalg.svd(self.window)
        trial_basis = U[:, :deim_dim]
        
        # compute singular values and store
        self.sing_val.append(Sv)
        n_dof = self.window.shape[0]//nMesh
        sing_val_states = []
        for idx in range(n_dof):
            window_states = self.window[idx*nMesh:(idx + 1)*nMesh,:]
            _, Sv_states, _ = np.linalg.svd(window_states)
            sing_val_states.append(Sv_states)
        
        sing_val_states = np.asarray(sing_val_states)
        self.sing_val_states.append(sing_val_states)
        
        # orthogonalize basis, redundant

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
            sampling_id = np.append(sampling_id, np.remainder(sampling[deim_dim + ctr], nMesh))
        
            # ensure all entries are unique
            sampling_id = np.unique(sampling_id)
            ctr = ctr + 1
        
        sampling_id = np.sort(sampling_id)
        
        if solver.time_iter % solver.out_interval == 0:
            basis_change = np.linalg.norm(old_basis - trial_basis @ trial_basis.T @ old_basis, 'fro')/np.linalg.norm(old_basis, 'fro')
            self.basis_inc = np.concatenate((self.basis_inc, np.array([basis_change])))

        return trial_basis, sampling_id
