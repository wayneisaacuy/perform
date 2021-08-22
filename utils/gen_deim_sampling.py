import os

import numpy as np
from numpy.linalg import svd
import scipy.linalg as LA

# Perform QDEIM

# ----- BEGIN USER INPUT -----

#data_dir = "~/path/to/data/dir"
#data_dir = "/Users/wayneisaacuy/Desktop/NYU/2021/adeimdom/perform/examples/contact_surface/hyperred_input"
#data_dir = "/Users/wayneisaacuy/Desktop/NYU/2021/adeimdom/perform/examples/standing_flame/hyperred_input"
data_dir = "/Users/wayneisaacuy/Desktop/NYU/2021/adeimdom/perform/examples/standing_flame/rom_input"
#data_file = "hyperred_modes_0_1_2_3.npy"
data_file = "spatial_modes_cons_0_1_2_3.npy"

# zero-indexed list of lists for group variables
#var_idxs = [[0], [1], [2], [3]]
var_idxs = [[0, 1, 2, 3]]

max_modes = 2

#out_dir = "/Users/wayneisaacuy/Desktop/NYU/2021/adeimdom/perform/examples/contact_surface/hyperred_input"
#out_dir = "/Users/wayneisaacuy/Desktop/NYU/2021/adeimdom/perform/examples/standing_flame/hyperred_input"
out_dir = "/Users/wayneisaacuy/Desktop/NYU/2021/adeimdom/perform/examples/standing_flame/rom_input"

# ----- END USER INPUT -----

out_dir = os.path.expanduser(out_dir)
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)


def main():
    
    assert len(var_idxs) == 1, "Non-vector rom not implemented yet"

    # load data, subsample
    in_file = os.path.join(data_dir, data_file)
    snap_arr = np.load(in_file)
    # _, num_cells, num_snaps = snap_arr.shape

    # find number of nodes
    nNodes = snap_arr.shape[1]

    # reshape the basis matrix
    group_arr = snap_arr[var_idxs[0], :, :]
    group_arr = np.reshape(group_arr, (-1, group_arr.shape[-1]), order="C")
    group_arr = group_arr[:,:max_modes]
    
    # perform qr with pivoting
    _, _, sampling = LA.qr(group_arr.T, pivoting=True)
    sampling_trunc = sampling[:max_modes]

    # apply modulo
    sampling_id = np.remainder(sampling_trunc, nNodes)
    sampling_id = np.unique(sampling_id)
    
    ctr = 0
    while sampling_id.shape[0] < max_modes:
        # get the next sampling index
        sampling_id = np.append(sampling_id, sampling[max_modes + ctr])
        
        # ensure all entries are unique
        sampling_id = np.unique(sampling_id)
        ctr = ctr + 1
    
    # # loop through groups
    # for group_idx, var_idx_list in enumerate(var_idxs):

    #     print("Processing variable group " + str(group_idx + 1))

    #     # break data array into different variable groups
    #     group_arr = snap_arr[var_idx_list, :, :]
    #     num_vars = group_arr.shape[0]

    #     # center and normalize data
    #     group_arr, cent_prof = center_data(group_arr)
    #     group_arr, norm_sub_prof, norm_fac_prof = normalize_data(group_arr)

    #     min_dim = min(num_cells * num_vars, num_snaps)
    #     modes_out = min(min_dim, max_modes)

    #     # compute SVD
    #     group_arr = np.reshape(group_arr, (-1, group_arr.shape[-1]), order="C")
    #     U, s, VT = svd(group_arr)
    #     U = np.reshape(U, (num_vars, num_cells, U.shape[-1]), order="C")

    #     # truncate modes
    #     basis = U[:, :, :modes_out]

    
    # suffix for output files
    suffix = ""
    suffix += str(max_modes)
    suffix += ".npy"

    # save data to disk
        
    spatial_mode_file = os.path.join(out_dir, "sampling_matrix_")
    np.save(spatial_mode_file + suffix, sampling_id)    

    print("DEIM sampling points generated!")


if __name__ == "__main__":
    main()
