import numpy as np

from perform.constants import REAL_TYPE
from perform.limiter.limiter import Limiter


class VenkatLimiter(Limiter):
    """Barth-Jespersen limiter.

    This implements the limiter of Venkatakrishnan (1993) in one dimension.
    The limiting function is differentiable, but limits in uniform regions.
    """

    def __init__(self):

        super().__init__()

    def calc_limiter(self, sol_domain, grad):
        """Compute multiplicative limiter.

        Args:
            sol_domain: SolutionDomain with which this Limiter is associated.
            grad: NumPy array of the un-limited gradient directly computed from finite difference stencil.

        Returns:
            NumPy array of the multiplicative gradient limiter profile.
        """

        sol_prim = sol_domain.sol_prim_full[:, sol_domain.grad_idxs]

        # get min/max of cell and neighbors
        sol_prim_min, sol_prim_max = self.calc_neighbor_minmax(sol_domain.sol_prim_full[:, sol_domain.grad_neigh_idxs])

        # extract gradient cells
        sol_prim_min = sol_prim_min[:, sol_domain.grad_neigh_extract]
        sol_prim_max = sol_prim_max[:, sol_domain.grad_neigh_extract]

        # unconstrained reconstruction at neighboring cell centers
        d_sol_prim = grad * sol_domain.mesh.dx
        sol_prim_left = sol_prim - d_sol_prim
        sol_prim_right = sol_prim + d_sol_prim

        # limiter defaults to 1
        phi_left = np.ones(sol_prim.shape, dtype=REAL_TYPE)
        phi_right = np.ones(sol_prim.shape, dtype=REAL_TYPE)

        # find idxs where difference is either positive or negative
        cond1_left = (sol_prim_left - sol_prim) > 0
        cond1_right = (sol_prim_right - sol_prim) > 0
        cond2_left = (sol_prim_left - sol_prim) < 0
        cond2_right = (sol_prim_right - sol_prim) < 0

        # apply smooth Venkatakrishnan function
        phi_left[cond1_left] = self.venkat_function(
            sol_prim_max[cond1_left], sol_prim[cond1_left], sol_prim_left[cond1_left]
        )

        phi_right[cond1_right] = self.venkat_function(
            sol_prim_max[cond1_right], sol_prim[cond1_right], sol_prim_right[cond1_right]
        )

        phi_left[cond2_left] = self.venkat_function(
            sol_prim_min[cond2_left], sol_prim[cond2_left], sol_prim_left[cond2_left]
        )

        phi_right[cond2_right] = self.venkat_function(
            sol_prim_min[cond2_right], sol_prim[cond2_right], sol_prim_right[cond2_right]
        )

        # take minimum limiter from left and right
        phi = np.minimum(phi_left, phi_right)

        return phi

    def venkat_function(self, maxmin_vals, cell_vals, face_vals):
        """Venkatakrishnan limiting function.

        Args:
            maxmin_vals:
                NumPy array of neighbor maximum/minimum values, depending on portion of limiter being computed.
            cell_vals: NumPy array of cell-centered solution profile.
            face_vals: NumPy array of left or right face reconstruction solution profile.

        Returns:
            NumPy array of Venkatakrishnan multiplicative gradient limiter values.
        """
        frac = (maxmin_vals - cell_vals) / (face_vals - cell_vals)
        frac_sq = np.square(frac)
        venk_vals = (frac_sq + 2.0 * frac) / (frac_sq + frac + 2.0)
        return venk_vals
