import os
from time import sleep

import numpy as np

from perform.constants import REAL_TYPE
from perform.input_funcs import read_input_file, catch_list, catch_input
from perform.solution.solution_phys import SolutionPhys
from perform.time_integrator import get_time_integrator
from perform.rom import get_rom_model, gen_ROMbasis, gen_DEIMsampling
import copy

class RomDomain:
    """Container class for all ROM models to be applied within a given SolutionDomain.

    The concept of a ROM solution being composed of multiple ROM models derives from the concept of
    "vector" vs. "scalar" ROMs initially referenced by Yuxiang Zhou in their 2010 Master's thesis.
    The "vector" concept follows the most common scenario in which a ROM provides a single mapping from a
    low-dimensional state to the complete physical state vector. The "scalar" concept is less common, whereby
    several ROM models map to separate subsets of the physical state variables (e.g. one model maps to density
    and energy, while another maps to momentum and density-weighted species mass fraction). Thus, some container
    for separate models is necessary.

    When running a ROM simulation, perform.driver.main() will generate a RomDomain for each SolutionDomain for which
    a ROM simulation is requested. The RomDomain will handle reading in the input parameters from the ROM parameter
    input file, checking the validity of these parameters, and initializing all requested RomModel's.

    During simulation runtime, RomDomain is responsible for executing most of each RomModel's higher-level functions
    and executing accessory functions, e.g. filtering. Beyond this, member functions of RomDomain generally handle
    operations that apply to the entire ROM solution, e.g. time integration, calculating residual norms, etc.

    Args:
        sol_domain: SolutionDomain with which this RomDomain is associated.
        solver: SystemSolver containing global simulation parameters.

    Attributes:
        rom_dict: Dictionary of parameters read from ROM parameter input file.
        rom_method: String of ROM method to be applied (e.g. "LinearGalerkinProj").
        num_models: Number of separate models encapsulated by RomDomain.
        latent_dims: List of latent variable dimensions for each RomModel.
        model_var_idxs: List of list of zero-indexed indices indicating which state variables each RomModel maps to.
        model_dir: String path to directory containing all files required to execute each RomModel.
        model_files:
            list of strings of file names associated with each RomModel's primary data structure
            (e.g. a linear model's trail basis), relative to model_dir.
        cent_ic: Boolean flag of whether the initial condition file should be used to center the solution profile.
        norm_sub_cons_in:
            list of strings of file names associated with each RomModel's conservative variable subtractive
            normalization profile, if needed, relative to model_dir.
        norm_fac_cons_in:
            list of strings of file names associated with each RomModel's conservative variable divisive
            normalization profile, if needed, relative to model_dir.
        cent_cons_in:
            list of strings of file names associated with each RomModel's conservative variable centering profile,
            if needed, relative to model_dir.
        norm_sub_prim_in:
            list of strings of file names associated with each RomModel's primitive variable subtractive
            normalization profile, if needed, relative to model_dir.
        norm_fac_prim_in:
            list of strings of file names associated with each RomModel's primitive variable divisive
            normalization profile, if needed, relative to model_dir.
        cent_prim_in:
            list of strings of file names associated with each RomModel's primitive variable centering profile
            if needed, relative to model_dir.
        has_time_integrator: Boolean flag indicating whether a given rom_method requires numerical time integration.
        is_intrusive:
            Boolean flag indicating whether a given rom_method is intrusive,
            i.e. requires computation of the governing equations RHS and its Jacobian.
        target_cons: Boolean flag indicating whether a given rom_method maps to the conservative variables.
        target_prim: Boolean flag indicating whether a given rom_method maps to the primitive variables.
        has_cons_norm:
            Boolean flag indicating whether a given rom_method requires conservative variable normalization profiles.
        has_cons_cent:
            Boolean flag indicating whether a given rom_method requires conservative variable centering profiles.
        has_prim_norm:
            Boolean flag indicating whether a given rom_method requires primitive variable normalization profiles.
        has_prim_cent:
            Boolean flag indicating whether a given rom_method requires primitive variable centering profiles.
        hyper_reduc: Boolean flag indicating whether hyper-reduction is to be used for an intrusive rom_method.
        model_list: list containing num_models RomModel objects associated with this RomDomain.
        low_dim_init_files:
            list of strings of file names associated with low-dimensional state initialization profiles
            for each RomModel.
        
        Added:
        adaptiveROM: Boolean flag indicating whether adaptive ROM is to be used for an intrusive rom_method.
    """

    def __init__(self, sol_domain, solver, args):
        
        #latent_dims = None, adapt_basis = None, init_window_size = None, adapt_window_size = None, adapt_update_freq = None, ADEIM_update = None, initbasis_snap_skip = None, use_FOM = None, adapt_every = None, update_rank = None, learning_rate = None):
        
        # unpack arguments
        latent_dims = args.latent_dims
        adapt_basis = args.adaptive
        init_window_size = args.init_window_size
        adapt_window_size = args.adapt_window_size
        #adapt_update_freq = args.adapt_update_freq
        ADEIM_update = args.ADEIM_update
        initbasis_snap_skip = args.initbasis_snap_skip
        use_FOM = args.use_FOM
        adapt_every = args.adapt_every
        update_rank = args.update_rank
        learning_rate = args.learn_rate
        sampling_update_freq = args.sampling_update_freq
        num_residual_comp = args.num_residual_comp
        use_line_search = args.use_line_search
        
        if use_line_search == None:
            self.use_line_search = 0
        else:
            self.use_line_search = 1
        
        self.param_string = "" # string containing parameters # AADEIM, init window size, window size, update rank, update freq, POD, useFOM, how many residual components
          
        rom_dict = read_input_file(solver.rom_inputs)
        self.rom_dict = rom_dict

        # Load model parameters
        self.rom_method = str(rom_dict["rom_method"])
        self.num_models = int(rom_dict["num_models"])
        
        model_var_idxs = catch_list(rom_dict, "model_var_idxs", [[-1]], len_highest=self.num_models)
        
        if latent_dims == None:
            self.latent_dims = catch_list(rom_dict, "latent_dims", [0], len_highest=self.num_models)
        else:
            self.latent_dims = len(model_var_idxs) * [ latent_dims ]

        # add rom dimension to parameter string
        self.param_string = self.param_string + "_dim_"
        for i in range(len(self.latent_dims)):
            self.param_string = self.param_string + str(self.latent_dims[i]) + "_"
        self.param_string = self.param_string[:-1]
        
        # load learning rate
        if learning_rate == None:
            self.learning_rate = 1
        else:
            self.learning_rate = learning_rate

        # Load initial rom basis parameters
        
        if init_window_size == None:
            self.initbasis_snapIterEnd = catch_input(rom_dict, "initbasis_snapIterEnd", solver.num_steps )
        else:
            self.initbasis_snapIterEnd = init_window_size
            
        self.initbasis_snapIterStart = catch_input(rom_dict, "initbasis_snapIterStart", 0 )
        
        if initbasis_snap_skip == None:
            self.initbasis_snapIterSkip = catch_input(rom_dict, "initbasis_snapIterSkip", 1 )
        else:
            self.initbasis_snapIterSkip = initbasis_snap_skip

        self.initbasis_centType = catch_input(rom_dict, "initbasis_centType", "mean" )
        self.initbasis_normType = catch_input(rom_dict, "initbasis_normType", "minmax" )

        # Check model parameters
        for i in self.latent_dims:
            assert i > 0, "latent_dims must contain positive integers"

        if self.num_models == 1:
            assert len(self.latent_dims) == 1, "Must provide only one value of latent_dims when num_models = 1"
            assert self.latent_dims[0] > 0, "latent_dims must contain positive integers"
        else:
            if len(self.latent_dims) == self.num_models:
                pass
            elif len(self.latent_dims) == 1:
                print("Only one value provided in latent_dims," + " applying to all models")
                sleep(1.0)
                self.latent_dims = [self.latent_dims[0]] * self.num_models
            else:
                raise ValueError("Must provide either num_models" + "or 1 entry in latent_dims")

        # Load and check model_var_idxs
        for model_idx in range(self.num_models):
            assert model_var_idxs[model_idx][0] != -1, "model_var_idxs input incorrectly, probably too few lists"
        assert len(model_var_idxs) == self.num_models, "Must specify model_var_idxs for every model"
        model_var_sum = 0
        for model_idx in range(self.num_models):
            model_var_sum += len(model_var_idxs[model_idx])
            for model_var_idx in model_var_idxs[model_idx]:
                assert model_var_idx >= 0, "model_var_idxs must be non-negative integers"
                assert (
                    model_var_idx < sol_domain.gas_model.num_eqs
                ), "model_var_idxs must less than the number of governing equations"
        assert model_var_sum == sol_domain.gas_model.num_eqs, (
            "Must specify as many model_var_idxs entries as governing equations ("
            + str(model_var_sum)
            + " != "
            + str(sol_domain.gas_model.num_eqs)
            + ")"
        )
        model_var_idxs_one_list = sum(model_var_idxs, [])
        assert len(model_var_idxs_one_list) == len(
            set(model_var_idxs_one_list)
        ), "All entries in model_var_idxs must be unique"
        self.model_var_idxs = model_var_idxs

        self.set_model_flags()
        
        # Get time integrator, if necessary
        # TODO: time_scheme should be specific to RomDomain, not the solver
        if self.has_time_integrator:
            self.time_integrator = get_time_integrator(solver.time_scheme, solver.param_dict, solver)
        else:
            self.time_integrator = None  # TODO: this might be pointless
            
        # check init files
        self.low_dim_init_files = catch_list(rom_dict, "low_dim_init_files", [""])
        if (len(self.low_dim_init_files) != 1) or (self.low_dim_init_files[0] != ""):
            assert len(self.low_dim_init_files) == self.num_models, (
                "If initializing any ROM model from a file, must provide list entries for every model. "
                + "If you don't wish to initialize from file for a model, input an empty string "
                " in the list entry."
            )
        else:
            self.low_dim_init_files = [""] * self.num_models
        
        # Load standardization profiles, if they are required
        self.cent_ic = catch_input(rom_dict, "cent_ic", False)
        self.norm_sub_prim_in = catch_list(rom_dict, "norm_sub_prim", [""])
        self.norm_fac_prim_in = catch_list(rom_dict, "norm_fac_prim", [""])
        self.cent_prim_in = catch_list(rom_dict, "cent_prim", [""])
        
        if "model_dir" in rom_dict:
            self.model_dir = str(rom_dict["model_dir"])
        else:
            self.model_dir = solver.working_dir

        if self.is_intrusive:
            self.hyper_reduc = catch_input(rom_dict, "hyper_reduc", False)
        
        self.model_files = [None] * self.num_models
        
        self.init_singval = np.array([])
        self.init_singval_states = np.array([])
        
        self.adaptive_init_window = None
        
        # check if basis and deim files are provided 
        if "model_files" in rom_dict:
            # Load and check model input locations    
            model_files = rom_dict["model_files"]
            assert len(model_files) == self.num_models, "Must provide model_files for each model"
            for model_idx in range(self.num_models):
                in_file = os.path.join(self.model_dir, model_files[model_idx])
                assert os.path.isfile(in_file), "Could not find model file at " + in_file
                self.model_files[model_idx] = in_file
            
            # Load standardization profiles, if they are required
            self.norm_sub_cons_in = catch_list(rom_dict, "norm_sub_cons", [""])
            self.norm_fac_cons_in = catch_list(rom_dict, "norm_fac_cons", [""])
            self.cent_cons_in = catch_list(rom_dict, "cent_cons", [""])
            
            if self.hyper_reduc:
                self.load_hyper_reduc(sol_domain)
                
        else:
            
            # not yet implemented for ROMs which are not linear
            if self.rom_method[-7:] == "tfkeras":
                raise Exception('Automated computation of basis and DEIM sampling points not yet supported for nonlinear ROMs.')
            
            # compute basis and scaling profiles
            spatial_modes, cent_file, norm_sub_file, norm_fac_file, \
                self.init_singval, self.init_singval_states, self.adaptive_init_window \
                = gen_ROMbasis(self.model_dir, solver.dt, self.initbasis_snapIterStart, \
                self.initbasis_snapIterEnd, self.initbasis_snapIterSkip, self.initbasis_centType, \
                self.initbasis_normType, self.model_var_idxs, self.latent_dims)
            
            self.cent_cons_in = cent_file
            self.norm_sub_cons_in = norm_sub_file
            self.norm_fac_cons_in = norm_fac_file
            
            for model_idx in range(self.num_models):
                 self.model_files[model_idx] = spatial_modes[model_idx]
                
            # compute hyperreduction sampling points
            if self.hyper_reduc:
                sampling_id = gen_DEIMsampling(self.model_var_idxs, spatial_modes[0], self.latent_dims[0])
                self.load_hyper_reduc(sol_domain, samp_idx = sampling_id, hyperred_basis = spatial_modes, hyperred_dims = self.latent_dims)
        
        # Set up hyper-reduction, if necessary
        if self.is_intrusive:
                
            # First check if using adaptive basis
            # Adaptive basis
            if adapt_basis == None:
                self.adaptiveROM = catch_input(rom_dict, "adaptiveROM", False)
            else:
                self.adaptiveROM = bool(adapt_basis)
                
            # Set up adaptive basis, if necessary
            
            if self.adaptiveROM:
                
                self.param_string = self.param_string + "_AADEIM_"
                
                # check that hyper reduction is true
                assert self.hyper_reduc, "Hyper reduction is needed for adaptive basis"
                
                # check that the time scheme is bdf
                assert solver.time_scheme == "bdf", "Adaptive basis requires implicit time-stepping"
                
                # check that the time order of bdf scheme is 1
                assert solver.param_dict['time_order'] == 1, "Adaptive basis rhs evaluation needs backward Euler discretization"
                
                # check that ROM and hyperreduction dimensions are the same
                assert np.abs(np.asarray(self.hyper_reduc_dims) - np.asarray(self.latent_dims)).max() == 0, "ROM and hyperreduction basis dimensions must be the same"
                
                # check that the ROM and hyperreduction bases are the same
                
                ROMDEIM_basis_same = 1
                for idx in range(self.num_models):
                    if isinstance(self.model_files[idx], np.ndarray):
                        rom_basis = self.model_files[idx]
                    else:
                        rom_basis = np.load(self.model_files[idx])
                    rom_basis = rom_basis[:,:,:self.latent_dims[idx]]
                    
                    if isinstance(self.hyper_reduc_files[idx], np.ndarray):
                        deim_basis = self.hyper_reduc_files[idx]
                    else:
                        deim_basis = np.load(self.hyper_reduc_files[idx])
                    deim_basis = deim_basis[:,:,:self.hyper_reduc_dims[idx]]
                    
                    if not np.allclose(rom_basis, deim_basis):
                        ROMDEIM_basis_same = 0
                        break
                
                assert ROMDEIM_basis_same == 1, "ROM and DEIM basis have to be the same"                    
                
                if update_rank == None:
                    self.adaptiveROMUpdateRank = catch_input(rom_dict, "adaptiveROMUpdateRank", 1)
                else:
                    self.adaptiveROMUpdateRank = update_rank
                    
                if sampling_update_freq == None:
                    self.adaptiveROMUpdateFreq = catch_input(rom_dict, "adaptiveROMUpdateFreq", 1)
                else:
                    self.adaptiveROMUpdateFreq = sampling_update_freq
                #self.adaptiveROMWindowSize = catch_input(rom_dict, "adaptiveROMWindowSize", [tempWindowSize + 1 for tempWindowSize in self.hyper_reduc_dims])
                
                if adapt_window_size == None:
                    self.adaptiveROMWindowSize = catch_input(rom_dict, "adaptiveROMWindowSize", max(self.hyper_reduc_dims)+1)
                else:
                    self.adaptiveROMWindowSize = adapt_window_size
                    
                # adjust window size if necessary
                if self.adaptiveROMWindowSize < max(self.hyper_reduc_dims)+1:
                    self.adaptiveROMWindowSize = max(self.hyper_reduc_dims)+1
                    
                #self.adaptiveROMInitTime = catch_input(rom_dict, "adaptiveROMInitTime", [tempInitTime + 1 for tempInitTime in self.adaptiveROMWindowSize])
                self.adaptiveROMInitTime = catch_input(rom_dict, "adaptiveROMInitTime", self.initbasis_snapIterEnd) #self.adaptiveROMWindowSize + 1)
                
                # if self.adaptiveROMInitTime < self.adaptiveROMWindowSize:
                #     self.adaptiveROMInitTime = copy.copy(self.adaptiveROMWindowSize)
                
                assert self.adaptiveROMInitTime >= self.adaptiveROMWindowSize, "initial window size has to be at least adaptive window size."
                
                if num_residual_comp == None:
                    self.adaptiveROMnumResSample = catch_input(rom_dict, "adaptiveROMnumResSample", sol_domain.gas_model.num_eqs * sol_domain.mesh.num_cells)
                else:
                    self.adaptiveROMnumResSample = num_residual_comp
                
                self.adaptiveROMFOMfile = catch_input(rom_dict, "adaptiveROMFOMfile", "unsteady_field_results/sol_cons_FOM_dt_" + str(solver.dt) + ".npy")
                self.adaptiveROMDebug = catch_input(rom_dict, "adaptiveROMDebug", 0)
                
                if use_FOM == None:
                    self.adaptiveROMuseFOM = catch_input(rom_dict, "adaptiveROMuseFOM", 0)
                else:
                    self.adaptiveROMuseFOM = use_FOM
                    
                if ADEIM_update == None:
                    self.adaptiveROMADEIMadapt = catch_input(rom_dict, "adaptiveROMADEIMadapt", "ADEIM")
                else:
                    self.adaptiveROMADEIMadapt = ADEIM_update
                    
                self.adaptiveROMFOMprimfile = catch_input(rom_dict, "adaptiveROMFOMprimfile", "unsteady_field_results/sol_prim_FOM_dt_" + str(solver.dt) + ".npy")
                
                if adapt_every == None:
                    self.adaptiveROMadaptevery = catch_input(rom_dict, "adaptiveROMadaptevery", 1)
                else:
                    self.adaptiveROMadaptevery = adapt_every
                        
                self.basis_adapted = 0
                
                assert self.adaptiveROMInitTime < solver.num_steps, "Initial time for adaptive ROM has to be less than the maximum number of time steps!"
                
                self.param_string = self.param_string + "iw_" + str(self.adaptiveROMInitTime)
                self.param_string = self.param_string + "_ws_" + str(self.adaptiveROMWindowSize)
                self.param_string = self.param_string + "_uf_" + str(self.adaptiveROMUpdateFreq)
                self.param_string = self.param_string + "_res_" + str(self.adaptiveROMnumResSample)
                self.param_string = self.param_string + "_useFOM_" + str(self.adaptiveROMuseFOM)
                
                self.param_string = self.param_string + "_" + self.adaptiveROMADEIMadapt
                
                # if self.adaptiveROMADEIMadapt:
                #     self.param_string = self.param_string + "_ADEIM"
                # else:
                #     self.param_string = self.param_string + "_POD"
                
                if solver.out_interval > 1:
                    self.param_string = self.param_string + "_skip_" + str(solver.out_interval)
                
                if self.adaptiveROMadaptevery > 1:
                    self.param_string = self.param_string + "_ae_" + str(self.adaptiveROMadaptevery)
                
                if self.adaptiveROMUpdateRank > 1:
                    self.param_string = self.param_string + "_ur_" + str(self.adaptiveROMUpdateRank)
                
                self.param_string = self.param_string + "_lr_" + str(self.learning_rate)
                    
     
        # Initialize
        self.model_list = [None] * self.num_models
        for model_idx in range(self.num_models):
            # Initialize model
            self.model_list[model_idx] = get_rom_model(model_idx, self, sol_domain, solver)
            model = self.model_list[model_idx]

            # Initialize state
            init_file = self.low_dim_init_files[model_idx]
            if init_file != "":
                assert os.path.isfile(init_file), "Could not find ROM initialization file at " + init_file
                model.code = np.load(init_file)
                model.update_sol(sol_domain)
            else:
                model.init_from_sol(sol_domain)

            # Initialize code history
            model.code_hist = [model.code.copy()] * (self.time_integrator.time_order + 1)
        
        if self.adaptiveROM:
                
                # check that we are constructing a vector ROM
                assert len(self.model_list) == 1, "AADEIM only works for vector ROM for now."

        sol_domain.sol_int.update_state(from_cons=self.target_cons)

        # Overwrite history with initialized solution
        sol_domain.sol_int.sol_hist_cons = [sol_domain.sol_int.sol_cons.copy()] * (self.time_integrator.time_order + 1)
        sol_domain.sol_int.sol_hist_prim = [sol_domain.sol_int.sol_prim.copy()] * (self.time_integrator.time_order + 1)
    
    def save_debug_quantities(self, solver):
        
        if self.adaptiveROM:
            for model_idx, model in enumerate(self.model_list):
                model.adapt.save_debugstats(self, solver.dt)
        else:
            pass

    def advance_iter(self, sol_domain, solver):
        """Advance low-dimensional state and full solution forward one physical time iteration.

        For non-intrusive ROMs without a time integrator, simply advances the solution one step.

        For intrusive and non-intrusive ROMs with a time integrator, begins numerical time integration
        and steps through sub-iterations.

        Args:
            sol_domain: SolutionDomain with which this RomDomain is associated.
            solver: SystemSolver containing global simulation parameters.
        """
        
        # where you solve the ROM for a current time step

        print("Iteration " + str(solver.iter))

        # check if basis was adapted
        if self.has_time_integrator and self.adaptiveROM:
            if self.basis_adapted == 1:

                # update code and FOM approx with respect to new basis    
                for model in self.model_list:
                    
                    model.code = np.dot(model.trial_basis.T, np.dot(model.prev_basis, model.code))
                    model.code_hist[0] = model.code.copy()
                    model.code_hist[1] = model.code.copy()
                    model.update_sol(sol_domain)
                
                sol_domain.sol_int.update_state(from_cons=(not sol_domain.time_integrator.dual_time))
                sol_domain.sol_int.sol_hist_cons[0] = sol_domain.sol_int.sol_cons.copy()
                sol_domain.sol_int.sol_hist_prim[0] = sol_domain.sol_int.sol_prim.copy()
            
                sol_domain.sol_int.sol_hist_cons[1] = sol_domain.sol_int.sol_cons.copy()
                sol_domain.sol_int.sol_hist_prim[1] = sol_domain.sol_int.sol_prim.copy()

                self.basis_adapted = 0                         

        # Update model which does NOT require numerical time integration
        if not self.has_time_integrator:
            raise ValueError("Iteration advance for models without numerical time integration not yet implemented")

        # If method requires numerical time integration
        else:
            
            # if self.basis_adapted == 1:
            #     # change ROM to the new basis
            #     # update model.code
            #     # add warning about explicit integrator
            #     pass
 
            for self.time_integrator.subiter in range(self.time_integrator.subiter_max):

                self.advance_subiter(sol_domain, solver)

                if self.time_integrator.time_type == "implicit":
                    self.calc_code_res_norms(sol_domain, solver, self.time_integrator.subiter)

                    if sol_domain.sol_int.res_norm_l2 < self.time_integrator.res_tol:
                        break

        # if adaptive, make adjustments to the stored ROM solution if time iteration is at most initial window size
        if self.adaptiveROM and solver.time_iter <= self.adaptiveROMInitTime :
            self.correct_code_adaptive_initwindow(solver, sol_domain)
        
        sol_domain.sol_int.update_sol_hist()
        self.update_code_hist()

        # update basis here
        if self.has_time_integrator:
            if self.adaptiveROM:

                # initialize window here
                if solver.time_iter == 1:
                    
                    for model_idx, model in enumerate(self.model_list):
                        #model.adapt.init_window(self, model)
                        
                        if self.adaptiveROMDebug == 1 or self.adaptiveROMuseFOM == 1:
                            model.adapt.load_FOM(self, model)
                            
                #self.basis_adapted = 0
                    
                for model_idx, model in enumerate(self.model_list):
                    deim_idx_flat = model.direct_samp_idxs_flat
                    trial_basis = model.trial_basis
                    decoded_ROM = model.decode_sol(model.code)
                    deim_dim = model.hyper_reduc_dim

                    # compute projection error
                    # if self.adaptiveROMDebug == 1:
                    model.adapt.compute_relprojerr(decoded_ROM, solver, sol_domain, model, sol_domain.sol_int.sol_prim)
                    
                    # update residual sampling points
                    
                    model.adapt.update_residualSampling_window(self, solver, sol_domain, trial_basis, deim_idx_flat, decoded_ROM, model, self.adaptiveROMuseFOM, self.adaptiveROMDebug)
                    
                    # call adeim
                    if model.adapt.window.shape[1] >= self.adaptiveROMWindowSize and solver.time_iter > self.adaptiveROMInitTime and solver.time_iter % self.adaptiveROMadaptevery == 0:
                        
                        if self.adaptiveROMADEIMadapt != "POD": #self.adaptiveROMADEIMadapt == "ADEIM" or self.adaptiveROMADEIMadapt == "AODEIM":
                            updated_basis, updated_interp_pts = model.adapt.adeim(self, trial_basis, deim_idx_flat, deim_dim, sol_domain.mesh.num_cells, solver, model.code)
                        else:    
                            updated_basis, updated_interp_pts = model.adapt.PODbasis(deim_dim, sol_domain.mesh.num_cells, trial_basis, solver)

                        # update deim interpolation points
                        # update rom_domain and sol_domain attributes. call method below to update rest
                        self.direct_samp_idxs = updated_interp_pts
                        sol_domain.direct_samp_idxs = updated_interp_pts
                        model.flatten_deim_idxs(self, sol_domain)

                        # update basis. make sure to update the deim basis too
                    
                        model.update_basis(updated_basis, self)
                        
                    # save debug quantities to file
                    if solver.time_iter == solver.num_steps:
                        model.adapt.save_debugstats(self, solver.dt)
                
                if model.adapt.window.shape[1] >= self.adaptiveROMWindowSize and solver.time_iter > self.adaptiveROMInitTime and solver.time_iter % self.adaptiveROMadaptevery == 0:
                    self.compute_cellidx_hyper_reduc(sol_domain)
                    self.basis_adapted = 1

                # update quantities that depend on the basis and the interpolation points. also adapt trial basis and hyperreduction basis

    def advance_subiter(self, sol_domain, solver):
        """Advance low-dimensional state and full solution forward one subiteration of time integrator.

        For intrusive ROMs, computes RHS and RHS Jacobian (if necessary).

        Args:
            sol_domain: SolutionDomain with which this RomDomain is associated.
            solver: SystemSolver containing global simulation parameters.
        """

        sol_int = sol_domain.sol_int
        res, res_jacob = None, None

        if self.is_intrusive:
            sol_domain.calc_rhs(solver)

        if self.time_integrator.time_type == "implicit":

            # Compute residual and residual Jacobian
            if self.is_intrusive:
                res = self.time_integrator.calc_residual(
                    sol_int.sol_hist_cons, sol_int.rhs, solver, samp_idxs=sol_domain.direct_samp_idxs
                )
                res_jacob = sol_domain.calc_res_jacob(solver)

            # Compute change in low-dimensional state
            for model_idx, model in enumerate(self.model_list):
                d_code, code_lhs, code_rhs = model.calc_d_code(res_jacob, res, sol_domain)
                
                if self.use_line_search:
                    self.learning_rate = 10
                    self.learning_rate = self.do_line_search(self.learning_rate, sol_domain, res_jacob, res, model, d_code, solver)
                
                model.code += self.learning_rate * d_code
                model.code_hist[0] = model.code.copy()
                model.update_sol(sol_domain)

                # Compute ROM residual for convergence measurement
                model.res = code_lhs @ d_code - code_rhs

            sol_int.update_state(from_cons=(not sol_domain.time_integrator.dual_time))
            sol_int.sol_hist_cons[0] = sol_int.sol_cons.copy()
            sol_int.sol_hist_prim[0] = sol_int.sol_prim.copy()

        else:

            for model_idx, model in enumerate(self.model_list):

                model.calc_rhs_low_dim(self, sol_domain)
                d_code = self.time_integrator.solve_sol_change(model.rhs_low_dim)
                model.code = model.code_hist[0] + d_code
                model.update_sol(sol_domain)

            sol_int.update_state(from_cons=True)
    
    def do_line_search(self, learn_rate, sol_domain, res_jacob, res, model, d_code, solver):
        
        sigma = 1e-4

        n_iter = 23
        
        sol_int = sol_domain.sol_int
        
        for iter_id in range(n_iter):
            
            # evaluate rhs of Armijo rule
            rhs = model.compute_linesearch_rhs_norm(res, sigma, learn_rate, res_jacob, d_code, sol_domain)
            
            new_state = model.decode_sol(model.code + learn_rate * d_code)
            new_state = new_state.reshape((-1,1))
            new_res = self.time_integrator.calc_fullydisc_residual(sol_int.sol_hist_cons, sol_domain,\
                            new_state, solver, self, samp_idxs=sol_domain.direct_samp_idxs)
            lhs = model.compute_linesearch_lhs_norm(new_res, sol_domain)
            
            if lhs <= rhs:
                break
            
            learn_rate = 0.5*learn_rate
        
        return learn_rate
        
        # new_state = model.decode_sol_oldbasis(model.code)
            
    def correct_code_adaptive_initwindow(self, solver, sol_domain):
        
        # extract centered and normalized FOM
        FOM_snap = self.adaptive_init_window[:, solver.time_iter]
        
        for model in self.model_list:
            # project to ROM space
            ROM_soln = np.dot(model.trial_basis.T, FOM_snap)
            
            # update model attributes
            model.code = ROM_soln
            model.code_hist[0] = model.code.copy()
            model.update_sol(sol_domain)
            
        sol_int = sol_domain.sol_int
            
        # update sol_int attributes
        
        sol_int.update_state(from_cons=(not sol_domain.time_integrator.dual_time))
        sol_int.sol_hist_cons[0] = sol_int.sol_cons.copy()
        sol_int.sol_hist_prim[0] = sol_int.sol_prim.copy()

    def update_code_hist(self):
        """Update low-dimensional state history after physical time step."""

        for model in self.model_list:

            model.code_hist[1:] = model.code_hist[:-1]
            model.code_hist[0] = model.code.copy()

    def calc_code_res_norms(self, sol_domain, solver, subiter):
        """Calculate and print low-dimensional linear solve residual norms.

        Computes L2 and L1 norms of low-dimensional linear solve residuals for each RomModel,
        as computed in advance_subiter(). These are averaged across all RomModels and printed to the terminal,
        and are used in advance_iter() to determine whether the Newton's method iterative solve has
        converged sufficiently. If the norm is below numerical precision, it defaults to 1e-16.

        Note that terminal output is ORDER OF MAGNITUDE (i.e. 1e-X, where X is the order of magnitude).

        Args:
            sol_domain: SolutionDomain with which this RomDomain is associated.
            solver: SystemSolver containing global simulation parameters.
            subiter: Current subiteration number within current time step's Newton's method iterative solve.
        """

        # Compute residual norm for each model
        norm_l2_sum = 0.0
        norm_l1_sum = 0.0
        for model in self.model_list:
            norm_l2, norm_l1 = model.calc_code_norms()
            norm_l2_sum += norm_l2
            norm_l1_sum += norm_l1

        # Average over all models
        norm_l2 = norm_l2_sum / self.num_models
        norm_l1 = norm_l1_sum / self.num_models

        # Norm is sometimes zero, just default to -16 I guess
        if norm_l2 == 0.0:
            norm_out_l2 = -16.0
        else:
            norm_out_l2 = np.log10(norm_l2)

        if norm_l1 == 0.0:
            norm_out_l1 = -16.0
        else:
            norm_out_l1 = np.log10(norm_l1)

        # Print to terminal
        out_string = (str(subiter + 1) + ":\tL2: %18.14f, \tL1: %18.14f") % (norm_out_l2, norm_out_l1,)
        print(out_string)

        sol_domain.sol_int.res_norm_l2 = norm_l2
        sol_domain.sol_int.resNormL1 = norm_l1
        sol_domain.sol_int.res_norm_hist[solver.iter - 1, :] = [norm_l2, norm_l1]

    def set_model_flags(self):
        """Set universal ROM method flags that dictate various execution behaviors.

        If a new RomModel is created, its flags should be set here.
        """

        self.has_time_integrator = False
        self.is_intrusive = False
        self.target_cons = False
        self.target_prim = False

        self.has_cons_norm = False
        self.has_cons_cent = False
        self.has_prim_norm = False
        self.has_prim_cent = False

        if self.rom_method == "linear_galerkin_proj":
            self.has_time_integrator = True
            self.is_intrusive = True
            self.target_cons = True
            self.has_cons_norm = True
            self.has_cons_cent = True
        elif self.rom_method == "linear_lspg_proj":
            self.has_time_integrator = True
            self.is_intrusive = True
            self.target_cons = True
            self.has_cons_norm = True
            self.has_cons_cent = True
        elif self.rom_method == "linear_splsvt_proj":
            self.has_time_integrator = True
            self.is_intrusive = True
            self.target_prim = True
            self.has_cons_norm = True
            self.has_prim_norm = True
            self.has_prim_cent = True
        elif self.rom_method == "autoencoder_galerkin_proj_tfkeras":
            self.has_time_integrator = True
            self.is_intrusive = True
            self.target_cons = True
            self.has_cons_norm = True
            self.has_cons_cent = True
        elif self.rom_method == "autoencoder_lspg_proj_tfkeras":
            self.has_time_integrator = True
            self.is_intrusive = True
            self.target_cons = True
            self.has_cons_norm = True
            self.has_cons_cent = True
        elif self.rom_method == "autoencoder_splsvt_proj_tfkeras":
            self.has_time_integrator = True
            self.is_intrusive = True
            self.target_prim = True
            self.has_cons_norm = True
            self.has_prim_norm = True
            self.has_prim_cent = True
        else:
            raise ValueError("Invalid ROM method name: " + self.rom_method)

        # TODO: not strictly true for the non-intrusive models
        assert self.target_cons != self.target_prim, "Model must target either the primitive or conservative variables"

    def load_hyper_reduc(self, sol_domain, samp_idx = [], hyperred_basis = [], hyperred_dims = []):
        """Loads direct sampling indices and determines cell indices for hyper-reduction array slicing.

        Numerous array slicing indices are required for various operations in efficiently computing
        the non-linear RHS term, such as calculating fluxes, gradients, source terms, etc. as well as for computing
        the RHS Jacobian if required. These slicing arrays are first generated here based on the initial sampling
        indices, but may later be updated during sampling adaptation.

        Todos:
            Many of these operations should be moved to their own separate functions when
            recomputing sampling for adaptive sampling.

        Args:
            sol_domain: SolutionDomain with which this RomDomain is associated.
        """

        # TODO: add some explanations for what each index array accomplishes

        if not isinstance(samp_idx, np.ndarray):
            # load and check sample points
            samp_file = catch_input(self.rom_dict, "samp_file", "")
            assert samp_file != "", "Must supply samp_file if performing hyper-reduction"
            samp_file = os.path.join(self.model_dir, samp_file)
            assert os.path.isfile(samp_file), "Could not find samp_file at " + samp_file
    
            # Indices of directly sampled cells, within sol_prim/cons
            # NOTE: assumed that sample indices are zero-indexed
            sol_domain.direct_samp_idxs = np.load(samp_file).flatten()
        else:
            sol_domain.direct_samp_idxs = samp_idx.flatten()
            
        sol_domain.direct_samp_idxs = (np.sort(sol_domain.direct_samp_idxs)).astype(np.int32)
        sol_domain.num_samp_cells = len(sol_domain.direct_samp_idxs)
        assert (
            sol_domain.num_samp_cells <= sol_domain.mesh.num_cells
        ), "Cannot supply more sampling points than cells in domain."
        assert np.amin(sol_domain.direct_samp_idxs) >= 0, "Sampling indices must be non-negative integers"
        assert (
            np.amax(sol_domain.direct_samp_idxs) < sol_domain.mesh.num_cells
        ), "Sampling indices must be less than the number of cells in the domain"
        assert (
            len(np.unique(sol_domain.direct_samp_idxs)) == sol_domain.num_samp_cells
        ), "Sampling indices must be unique"
        
        # Paths to hyper-reduction files (unpacked later)
        self.hyper_reduc_files = [None] * self.num_models
        if hyperred_basis == []:
            hyper_reduc_files = self.rom_dict["hyper_reduc_files"]
            assert len(hyper_reduc_files) == self.num_models, "Must provide hyper_reduc_files for each model"
            for model_idx in range(self.num_models):
                in_file = os.path.join(self.model_dir, hyper_reduc_files[model_idx])
                assert os.path.isfile(in_file), "Could not find hyper-reduction file at " + in_file
                self.hyper_reduc_files[model_idx] = in_file
        else:
            for model_idx in range(self.num_models):
                self.hyper_reduc_files[model_idx] = hyperred_basis[model_idx]

        # Load hyper reduction dimensions and check validity
        if hyperred_dims != []:
            self.hyper_reduc_dims = hyperred_dims
        else:
            self.hyper_reduc_dims = catch_list(self.rom_dict, "hyper_reduc_dims", [0], len_highest=self.num_models)

        for i in self.hyper_reduc_dims:
            assert i > 0, "hyper_reduc_dims must contain positive integers"
        if self.num_models == 1:
            assert (
                len(self.hyper_reduc_dims) == 1
            ), "Must provide only one value of hyper_reduc_dims when num_models = 1"
            assert self.hyper_reduc_dims[0] > 0, "hyper_reduc_dims must contain positive integers"
        else:
            if len(self.hyper_reduc_dims) == self.num_models:
                pass
            elif len(self.hyper_reduc_dims) == 1:
                print("Only one value provided in hyper_reduc_dims, applying to all models")
                sleep(1.0)
                self.hyper_reduc_dims = [self.hyper_reduc_dims[0]] * self.num_models
            else:
                raise ValueError("Must provide either num_models or 1 entry in hyper_reduc_dims")

        # Copy indices for ease of use
        self.num_samp_cells = sol_domain.num_samp_cells
        self.direct_samp_idxs = sol_domain.direct_samp_idxs

        self.compute_cellidx_hyper_reduc(sol_domain)

    def compute_cellidx_hyper_reduc(self, sol_domain):
        
        # moved part of load_hyper_reduc here so that this function can be called if DEIM interpolation points are adapted
        
        # Compute indices for inviscid flux calculations
        # NOTE: have to account for fact that boundary cells are prepended/appended
        # Indices of "left" cells for flux calcs, within sol_prim/cons_full
        sol_domain.flux_samp_left_idxs = np.zeros(2 * sol_domain.num_samp_cells, dtype=np.int32)
        sol_domain.flux_samp_left_idxs[0::2] = sol_domain.direct_samp_idxs
        sol_domain.flux_samp_left_idxs[1::2] = sol_domain.direct_samp_idxs + 1

        # Indices of "right" cells for flux calcs, within sol_prim/cons_full
        sol_domain.flux_samp_right_idxs = np.zeros(2 * sol_domain.num_samp_cells, dtype=np.int32)
        sol_domain.flux_samp_right_idxs[0::2] = sol_domain.direct_samp_idxs + 1
        sol_domain.flux_samp_right_idxs[1::2] = sol_domain.direct_samp_idxs + 2

        # Eliminate repeated indices
        sol_domain.flux_samp_left_idxs = np.unique(sol_domain.flux_samp_left_idxs)
        sol_domain.flux_samp_right_idxs = np.unique(sol_domain.flux_samp_right_idxs)
        sol_domain.num_flux_faces = len(sol_domain.flux_samp_left_idxs)

        # Indices of flux array which correspond to left face of cell and map to direct_samp_idxs
        sol_domain.flux_rhs_idxs = np.zeros(sol_domain.num_samp_cells, np.int32)
        for i in range(1, sol_domain.num_samp_cells):
            # if this cell is adjacent to previous sampled cell
            if sol_domain.direct_samp_idxs[i] == (sol_domain.direct_samp_idxs[i - 1] + 1):
                sol_domain.flux_rhs_idxs[i] = sol_domain.flux_rhs_idxs[i - 1] + 1
            # otherwise
            else:
                sol_domain.flux_rhs_idxs[i] = sol_domain.flux_rhs_idxs[i - 1] + 2

        # Compute indices for gradient calculations
        # NOTE: also need to account for prepended/appended boundary cells
        # TODO: generalize for higher-order schemes
        if sol_domain.space_order > 1:
            if sol_domain.space_order == 2:

                # Indices of cells for which gradients need to be calculated, within sol_prim/cons_full
                sol_domain.grad_idxs = np.concatenate(
                    (sol_domain.direct_samp_idxs + 1, sol_domain.direct_samp_idxs, sol_domain.direct_samp_idxs + 2,)
                )
                sol_domain.grad_idxs = np.unique(sol_domain.grad_idxs)

                # Exclude left neighbor of inlet, right neighbor of outlet
                if sol_domain.grad_idxs[0] == 0:
                    sol_domain.grad_idxs = sol_domain.grad_idxs[1:]

                if sol_domain.grad_idxs[-1] == (sol_domain.mesh.num_cells + 1):
                    sol_domain.grad_idxs = sol_domain.grad_idxs[:-1]

                sol_domain.num_grad_cells = len(sol_domain.grad_idxs)

                # Indices of gradient cells and their immediate neighbors, within sol_prim/cons_full
                sol_domain.grad_neigh_idxs = np.concatenate((sol_domain.grad_idxs - 1, sol_domain.grad_idxs + 1))
                sol_domain.grad_neigh_idxs = np.unique(sol_domain.grad_neigh_idxs)

                # Exclude left neighbor of inlet, right neighbor of outlet
                if sol_domain.grad_neigh_idxs[0] == -1:
                    sol_domain.grad_neigh_idxs = sol_domain.grad_neigh_idxs[1:]

                if sol_domain.grad_neigh_idxs[-1] == (sol_domain.mesh.num_cells + 2):
                    sol_domain.grad_neigh_idxs = sol_domain.grad_neigh_idxs[:-1]

                # Indices within gradient neighbor indices to extract gradient cells, excluding boundaries
                _, _, sol_domain.grad_neigh_extract = np.intersect1d(
                    sol_domain.grad_idxs, sol_domain.grad_neigh_idxs, return_indices=True,
                )

                # Indices of grad_idxs in flux_samp_left_idxs and flux_samp_right_idxs and vice versa
                _, sol_domain.grad_left_extract, sol_domain.flux_left_extract = np.intersect1d(
                    sol_domain.grad_idxs, sol_domain.flux_samp_left_idxs, return_indices=True,
                )

                # Indices of grad_idxs in flux_samp_right_idxs and flux_samp_right_idxs and vice versa
                _, sol_domain.grad_right_extract, sol_domain.flux_right_extract = np.intersect1d(
                    sol_domain.grad_idxs, sol_domain.flux_samp_right_idxs, return_indices=True,
                )

            else:
                raise ValueError("Sampling for higher-order schemes" + " not implemented yet")

        # for Jacobian calculations
        if sol_domain.direct_samp_idxs[0] == 0:
            sol_domain.jacob_left_samp = sol_domain.flux_rhs_idxs[1:].copy()
        else:
            sol_domain.jacob_left_samp = sol_domain.flux_rhs_idxs.copy()

        if sol_domain.direct_samp_idxs[-1] == (sol_domain.sol_int.num_cells - 1):
            sol_domain.jacob_right_samp = sol_domain.flux_rhs_idxs[:-1].copy() + 1
        else:
            sol_domain.jacob_right_samp = sol_domain.flux_rhs_idxs.copy() + 1

        # re-initialize solution objects to proper size
        gas = sol_domain.gas_model
        ones_prof = np.ones((gas.num_eqs, sol_domain.num_flux_faces), dtype=REAL_TYPE)
        sol_domain.sol_left = SolutionPhys(gas, sol_domain.num_flux_faces, sol_prim_in=ones_prof)
        sol_domain.sol_right = SolutionPhys(gas, sol_domain.num_flux_faces, sol_prim_in=ones_prof)

        if sol_domain.invisc_flux_name == "roe":
            ones_prof = np.ones((gas.num_eqs, sol_domain.num_flux_faces), dtype=REAL_TYPE)
            sol_domain.sol_ave = SolutionPhys(gas, sol_domain.num_flux_faces, sol_prim_in=ones_prof)

        # Redo CSR matrix indices for sparse Jacobian
        num_cells = sol_domain.mesh.num_cells
        num_samp_cells = sol_domain.num_samp_cells
        num_elements_center = gas.num_eqs ** 2 * num_samp_cells
        if sol_domain.direct_samp_idxs[0] == 0:
            num_elements_lower = gas.num_eqs ** 2 * (num_samp_cells - 1)
        else:
            num_elements_lower = num_elements_center
        if sol_domain.direct_samp_idxs[-1] == (num_cells - 1):
            num_elements_upper = gas.num_eqs ** 2 * (num_samp_cells - 1)
        else:
            num_elements_upper = num_elements_center
        sol_domain.sol_int.jacob_dim_first = gas.num_eqs * num_samp_cells
        sol_domain.sol_int.jacob_dim_second = gas.num_eqs * num_cells

        row_idxs_center = np.zeros(num_elements_center, dtype=np.int32)
        col_idxs_center = np.zeros(num_elements_center, dtype=np.int32)
        row_idxs_upper = np.zeros(num_elements_upper, dtype=np.int32)
        col_idxs_upper = np.zeros(num_elements_upper, dtype=np.int32)
        row_idxs_lower = np.zeros(num_elements_lower, dtype=np.int32)
        col_idxs_lower = np.zeros(num_elements_lower, dtype=np.int32)

        lin_idx_A = 0
        lin_idx_B = 0
        lin_idx_C = 0
        for i in range(gas.num_eqs):
            for j in range(gas.num_eqs):
                for k in range(num_samp_cells):

                    row_idxs_center[lin_idx_A] = i * num_samp_cells + k
                    col_idxs_center[lin_idx_A] = j * num_cells + sol_domain.direct_samp_idxs[k]
                    lin_idx_A += 1

                    if sol_domain.direct_samp_idxs[k] < (num_cells - 1):
                        row_idxs_upper[lin_idx_B] = i * num_samp_cells + k
                        col_idxs_upper[lin_idx_B] = j * num_cells + sol_domain.direct_samp_idxs[k] + 1
                        lin_idx_B += 1

                    if sol_domain.direct_samp_idxs[k] > 0:
                        row_idxs_lower[lin_idx_C] = i * num_samp_cells + k
                        col_idxs_lower[lin_idx_C] = j * num_cells + sol_domain.direct_samp_idxs[k] - 1
                        lin_idx_C += 1

        sol_domain.sol_int.jacob_row_idxs = np.concatenate((row_idxs_center, row_idxs_lower, row_idxs_upper))
        sol_domain.sol_int.jacob_col_idxs = np.concatenate((col_idxs_center, col_idxs_lower, col_idxs_upper))

        # Gamma inverse indices
        # TODO: once the conservative Jacobians get implemented, this is unnecessary, remove and clean
        if sol_domain.time_integrator.dual_time:
            sol_domain.gamma_idxs = sol_domain.direct_samp_idxs
        else:
            sol_domain.gamma_idxs = np.concatenate(
                (sol_domain.direct_samp_idxs, sol_domain.direct_samp_idxs + 1, sol_domain.direct_samp_idxs - 1)
            )
            sol_domain.gamma_idxs = np.unique(sol_domain.gamma_idxs)
            if sol_domain.gamma_idxs[0] == -1:
                sol_domain.gamma_idxs = sol_domain.gamma_idxs[1:]
            if sol_domain.gamma_idxs[-1] == sol_domain.mesh.num_cells:
                sol_domain.gamma_idxs = sol_domain.gamma_idxs[:-1]

        _, sol_domain.gamma_idxs_center, _ = np.intersect1d(
            sol_domain.gamma_idxs, sol_domain.direct_samp_idxs, return_indices=True,
        )

        _, sol_domain.gamma_idxs_left, _ = np.intersect1d(
            sol_domain.gamma_idxs, sol_domain.direct_samp_idxs - 1, return_indices=True,
        )

        _, sol_domain.gamma_idxs_right, _ = np.intersect1d(
            sol_domain.gamma_idxs, sol_domain.direct_samp_idxs + 1, return_indices=True,
        )