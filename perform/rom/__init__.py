# linear models
from perform.rom.projection_rom.linear_proj_rom.linear_galerkin_proj import LinearGalerkinProj
from perform.rom.projection_rom.linear_proj_rom.linear_lspg_proj import LinearLSPGProj
from perform.rom.projection_rom.linear_proj_rom.linear_splsvt_proj import LinearSPLSVTProj

import os

import numpy as np
from numpy.linalg import svd
import scipy.linalg as LA

# TensorFlow-Keras autoencoder models
TFKERAS_IMPORT_SUCCESS = True
try:
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # don't print all the TensorFlow warnings
    from perform.rom.projection_rom.autoencoder_proj_rom.autoencoder_tfkeras.autoencoder_galerkin_proj_tfkeras import (
        AutoencoderGalerkinProjTFKeras,
    )
    from perform.rom.projection_rom.autoencoder_proj_rom.autoencoder_tfkeras.autoencoder_lspg_proj_tfkeras import (
        AutoencoderLSPGProjTFKeras,
    )
    from perform.rom.projection_rom.autoencoder_proj_rom.autoencoder_tfkeras.autoencoder_splsvt_proj_tfkeras import (
        AutoencoderSPLSVTProjTFKeras,
    )

except ImportError:
    TFKERAS_IMPORT_SUCCESS = False


def get_rom_model(model_idx, rom_domain, sol_domain, solver):
    """Helper function to retrieve ROM models"""

    # linear subspace methods
    if rom_domain.rom_method == "linear_galerkin_proj":
        model = LinearGalerkinProj(model_idx, rom_domain, sol_domain, solver)

    elif rom_domain.rom_method == "linear_lspg_proj":
        model = LinearLSPGProj(model_idx, rom_domain, sol_domain)

    elif rom_domain.rom_method == "linear_splsvt_proj":
        model = LinearSPLSVTProj(model_idx, rom_domain, sol_domain)

    # TensorFlow-Keras autoencoder models
    elif rom_domain.rom_method[-7:] == "tfkeras":
        if TFKERAS_IMPORT_SUCCESS:

            if rom_domain.rom_method == "autoencoder_galerkin_proj_tfkeras":
                model = AutoencoderGalerkinProjTFKeras(model_idx, rom_domain, sol_domain)

            elif rom_domain.rom_method == "autoencoder_lspg_proj_tfkeras":
                model = AutoencoderLSPGProjTFKeras(model_idx, rom_domain, sol_domain)

            elif rom_domain.rom_method == "autoencoder_splsvt_proj_tfkeras":
                model = AutoencoderSPLSVTProjTFKeras(model_idx, rom_domain, sol_domain)

            else:
                raise ValueError("Invalid TF-Keras ROM method name: " + rom_domain.rom_method)

        else:
            raise ValueError(
                "TF-Keras models failed to import," + " please check that TensorFlow >= 2.0" + " is installed"
            )

    # TODO: PyTorch autoencoder models

    else:
        raise ValueError("Invalid ROM method name: " + rom_domain.rom_method)

    return model

def gen_ROMbasis(data_dir, dt, iter_start, iter_end, iter_skip, cent_type, norm_type, var_idxs, max_modes):

    # construct data file
    data_file = "sol_cons_FOM_dt_" + str(dt) + ".npy"
    
    # load data, subsample
    in_file = os.path.join(data_dir, data_file)
    snap_arr = np.load(in_file)

    snap_arr = snap_arr[:, :, iter_start : iter_end + 1 : iter_skip]
    _, num_cells, num_snaps = snap_arr.shape

    spatial_modes = []
    cent_file = []
    norm_sub_file = []
    norm_fac_file = []
    
    # loop through groups
    for group_idx, var_idx_list in enumerate(var_idxs):

        print("ROM basis processing variable group " + str(group_idx + 1))

        # break data array into different variable groups
        group_arr = snap_arr[var_idx_list, :, :]
        num_vars = group_arr.shape[0]

        # center and normalize data
        group_arr, cent_prof = center_data(group_arr, cent_type)
        group_arr, norm_sub_prof, norm_fac_prof = normalize_data(group_arr, norm_type)

        min_dim = min(num_cells * num_vars, num_snaps)
        modes_out = min(min_dim, max_modes[group_idx])

        # compute SVD
        group_arr = np.reshape(group_arr, (-1, group_arr.shape[-1]), order="C")
        U, s, VT = svd(group_arr)
        U = np.reshape(U, (num_vars, num_cells, U.shape[-1]), order="C")

        # truncate modes
        basis = U[:, :, :modes_out]

        spatial_modes.append(basis)
        cent_file.append(cent_prof)
        norm_sub_file.append(norm_sub_prof)
        norm_fac_file.append(norm_fac_prof)
        
    print("POD basis generated!")
    
    return spatial_modes, cent_file, norm_sub_file, norm_fac_file

def gen_DEIMsampling(var_idxs, basis, deim_dim):
    
    assert len(var_idxs) == 1, "Non-vector rom not implemented yet for DEIM"

    # find number of nodes
    nNodes = basis.shape[1]

    # reshape the basis matrix
    group_arr = basis[var_idxs[0], :, :]
    group_arr = np.reshape(group_arr, (-1, group_arr.shape[-1]), order="C")
    
    # perform qr with pivoting
    _, _, sampling = LA.qr(group_arr.T, pivoting=True)
    sampling_trunc = sampling[:deim_dim]

    # apply modulo
    sampling_id = np.remainder(sampling_trunc, nNodes)
    sampling_id = np.unique(sampling_id)
    
    ctr = 0
    while sampling_id.shape[0] < deim_dim:
        # get the next sampling index
        sampling_id = np.append(sampling_id, np.remainder(sampling[deim_dim + ctr], nNodes))
        
        # ensure all entries are unique
        sampling_id = np.unique(sampling_id)
        ctr = ctr + 1
    
    # sort indices
    sampling_id = np.sort(sampling_id)  

    print("DEIM sampling points generated!")
    return sampling_id

# center training data
def center_data(data_arr, cent_type):

    # center around the initial condition
    if cent_type == "init_cond":
        cent_prof = data_arr[:, :, [0]]

    # center around the sample mean
    elif cent_type == "mean":
        cent_prof = np.mean(data_arr, axis=2, keepdims=True)

    else:
        raise ValueError("Invalid cent_type input: " + str(cent_type))

    data_arr -= cent_prof

    return data_arr, np.squeeze(cent_prof, axis=-1)


# normalize training data
def normalize_data(data_arr, norm_type):

    ones_prof = np.ones((data_arr.shape[0], data_arr.shape[1], 1), dtype=np.float64)
    zero_prof = np.zeros((data_arr.shape[0], data_arr.shape[1], 1), dtype=np.float64)

    # normalize by  (X - min(X)) / (max(X) - min(X))
    if norm_type == "minmax":
        min_vals = np.amin(data_arr, axis=(1, 2), keepdims=True)
        max_vals = np.amax(data_arr, axis=(1, 2), keepdims=True)
        norm_sub_prof = min_vals * ones_prof
        norm_fac_prof = (max_vals - min_vals) * ones_prof

    # normalize by L2 norm sqaured of each variable
    elif norm_type == "l2":
        data_arr_sq = np.square(data_arr)
        norm_fac_prof = np.sum(np.sum(data_arr_sq, axis=1, keepdims=True), axis=2, keepdims=True)
        norm_fac_prof /= data_arr.shape[1] * data_arr.shape[2]
        norm_fac_prof = norm_fac_prof * ones_prof
        norm_sub_prof = zero_prof

    else:
        raise ValueError("Invalid norm_type input: " + str(norm_type))

    data_arr = (data_arr - norm_sub_prof) / norm_fac_prof

    return data_arr, np.squeeze(norm_sub_prof, axis=-1), np.squeeze(norm_fac_prof, axis=-1)