# useful constants used throughout the code
import numpy as np

# relevant directory absolute paths, set at runtime (not really constants)
working_dir         = None 	
unsteady_output_dir = None 
probe_output_dir    = None 
image_output_dir    = None 
restart_output_dir  = None

# Precision of real and complex numbers
REAL_TYPE 	 = np.float64
COMPLEX_TYPE = np.complex128

R_UNIV 		= 8314.4621 # universal gas constant, J/(K*kmol)
SUTH_TEMP 	= 110.4 	# Sutherland temperature

TINY_NUM 	= 1.0e-25 	# very small number
HUGE_NUM 	= 1.0e25	# very large number

# time integrator defaults
SUBITER_MAX_IMP_DEFAULT = 50
L2_RES_TOL_DEFAULT      = 1.0e-12
L2_STEADY_TOL_DEFAULT   = 1.0e-12
RES_NORM_PRIM_DEFAULT   = [1.0e5, 10.0, 300.0, 1.0]
DTAU_DEFAULT            = 1.0e-5
CFL_DEFAULT             = 1.0
VNN_DEFAULT             = 20.0

FD_STEP_DEFAULT = 1.0e-6

# visualization constants
FIG_WIDTH_DEFAULT  = 12
FIG_HEIGHT_DEFAULT = 6

# output directory names
UNSTEADY_OUTPUT_DIR_NAME = "UnsteadyFieldResults"
PROBE_OUTPUT_DIR_NAME    = "ProbeResults"
IMAGE_OUTPUT_DIR_NAME    = "ImageResults"
RESTART_OUTPUT_DIR_NAME  = "RestartFiles"

# input files
PARAM_INPUTS = "solverParams.inp"
ROM_INPUTS   = "romParams.inp"