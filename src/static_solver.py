"""
Reference:
    Trapper, P.A. (2019). "Numerical technique for static equilibrium
    analysis of a marine pipeline during laying". Applied Ocean Research,
    88, 48–62.

Computes the static equilibrium catenary configuration including:
  - Node angles theta
  - Node positions (x, y)
  - Pretension T0 at each node

The result is consumed by DynamicSolver to linearize the FEM about the static equilibrium state.
"""

import numpy as np
from scipy.optimize import fsolve

class StaticSolver:
    """
    Trapper (2019) FDM static pipeline catenary solver.

    Parameters
    ----------
    OD       : float   outer diameter [m]
    WT       : float   wall thickness [m]
    E        : float   Young's modulus [Pa]
    rho_s    : float   effective pipe density [kg/m³]
    phi0     : float   departure angle at LOP from horizontal [rad]
    P0       : float   horizontal lay tension [N]
    WD       : float   water depth [m]
    ks       : float   seabed stiffness [N/m²]
    delta_s  : float   element arc length [m]
    """

    def __init__(self, OD, WT, E, rho_s, phi0, P0, WD, ks, delta_s):

        self.OD = OD
        self.WT = WT
        self.E = E
        self.rho_s = rho_s
        self.rho_w = 1025.0
        self.g = 9.81
        self.phi0 = phi0
        self.P0 = P0
        self.WD = WD
        self.ks = ks
        self.delta_s = delta_s

        # Section properties
        self.EI = E * np.pi/64 * (OD**4 - (OD - 2*WT)**4)
        A = np.pi/4 * (OD**2 - (OD - 2*WT)**2)
        self.A_steel = A
        self.w_s = (rho_s - self.rho_w) * self.g * A

        # Boundary angles
        self.theta_FP = 2*np.pi # FP (on seabed) — horizontal
        self.theta_LOP = 2*np.pi - phi0 # LOP (at surface) — departure angle

    def compute_xy(self, theta):

        ds = self.delta_s
        sd = np.sin(theta) * ds
        total = np.sum(sd)
        prefix = np.concatenate(([0.], np.cumsum(sd[:-1])))
        x = np.concatenate(([0.], np.cumsum(np.cos(theta[:-1]) * ds)))
        return x, -(total - prefix)

    def residual(self, theta_inner):

        ds = self.delta_s

        theta = np.concatenate(([self.theta_FP], theta_inner, [self.theta_LOP]))

        N = len(theta) - 1

        _, y = self.compute_xy(theta)

        ksb = np.where(y >= self.WD, self.ks * ds, 0.)
        KSB = np.cumsum(ksb)

        W = np.cumsum(np.full(N+1, self.w_s * ds))
        st = np.sin(theta)
        ct = np.cos(theta)

        k = np.arange(1, N)

        cKs = np.cumsum(KSB * ds * st)

        cds = np.cumsum(ds * st)

        ds2 = cKs[k] + KSB[k] * (cds[-1] - cds[k])

        return (self.EI / ds * (-theta[k-1] + 2*theta[k] - theta[k+1])
                + ds * ct[k] * ds2
                + self.WD * KSB[k] * ds * ct[k]
                + W[k] * ds * ct[k]
                + self.P0 * ds * st[k])

    def pretension(self, theta):

        ct = np.cos(theta)
        ct_safe = np.where(np.abs(ct) > 0.01, ct, np.sign(ct + 1e-10) * 0.01)
        return self.P0 / np.abs(ct_safe)

    def solve(self, n_increments=400, verbose=False):
        """
        Convergence criterion: y[FP] oscillation < 0.1% of WD over 6 increments
        (after minimum 400 increments with seabed contact and small residual).

        Returns
        -------
        theta  : ndarray (N_nodes,)   node angles [rad]
        x      : ndarray (N_nodes,)   node x-positions [m]
        y      : ndarray (N_nodes,)   node depths [m]  (positive downward)
        T_node : ndarray (N_nodes,)   pretension at each node [N]
        """
        T = self.P0
        w = self.w_s
        ds = self.delta_s

        # Catenary initial guess
        S = T/w * np.sinh(w/T * (T/w * np.arccosh(1 + self.WD*w/T)))
        s = np.arange(0, S + ds, ds)
        theta0 = 2*np.pi - np.arctan(w/T * s)
        theta0[0] = self.theta_FP
        theta0[-1] = self.theta_LOP
        theta = np.concatenate(([self.theta_FP],
                                 fsolve(self.residual, theta0[1:-1], maxfev=500000),
                                 [self.theta_LOP]))
        y_fp_history = []

        for m in range(n_increments):
            sol, info, _, _ = fsolve(self.residual, theta[1:-1], full_output=True, maxfev=500000)
            theta[1:-1] = sol
            norm_R = np.linalg.norm(info['fvec'])
            x, y = self.compute_xy(theta)
            nc = int(np.sum(y >= self.WD))
            y_fp = y[0]
            y_fp_history.append(y_fp)

            if verbose and m % 20 == 0:
                print(f'm={m+1}: nc={nc}, y[FP]={y_fp:.3f}m, |R|={norm_R:.2e}')

            # Convergence check
            if (m >= 40 and nc >= 5 and norm_R < 1.0
                    and len(y_fp_history) >= 6):
                window = y_fp_history[-6:]
                if max(window) - min(window) < 0.001 * self.WD:
                    if verbose:
                        print(f'Converged after {m+1} increments')
                    break

            L_new = len(theta) * ds
            theta = np.concatenate((
                [self.theta_FP],
                [2*np.pi - np.arctan(w/T * L_new)],
                theta[1:]))

        x, y = self.compute_xy(theta)
        T_node = self.pretension(theta)
        return theta, x, y, T_node
