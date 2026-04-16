"""
Reference:
    Kim, H-S. & Kim, B.W. (2018). "An efficient linearized dynamic analysis
    method for structural safety design of J-lay and S-lay pipeline
    installation." Ships and Offshore Structures.
"""

import numpy as np
import time
from scipy.linalg import lu_factor, lu_solve
from .static_solver import StaticSolver
from .wave_field import WaveField


class DynamicSolver:
    """
    Parameters
    ----------
    static            : StaticSolver  — converged static solution
    CD, CM, CA, CF    : float         — Morison drag, inertia, added-mass, friction coefficients
    zeta_seabed       : float         — seabed spring-damper damping ratio
    delta_penetration : float         — seabed reference penetration depth [m]
    rayleigh_alpha    : float         — Rayleigh mass damping coefficient
    rayleigh_beta     : float         — Rayleigh stiffness damping coefficient
    rho_inf           : float         — generalized-alpha spectral radius at infinite frequency  (0=max, 1=none)
    """

    def __init__(self, static: StaticSolver,
                 CD=1.2, CM=2.0, CA=1.0, CF=0.05,
                 zeta_seabed=0.05, delta_penetration=0.05,
                 rayleigh_alpha=0.0, rayleigh_beta=6.3e-4,
                 rho_inf=0.8):

        self.st = static
        self.CD = CD
        self.CM = CM
        self.CA = CA
        self.CF = CF
        self.zeta_c = zeta_seabed
        self.delta_c = delta_penetration
        self.ra_a = rayleigh_alpha
        self.ra_b = rayleigh_beta

        # Generalized-alpha parameters (Chung & Hulbert 1993)
        ri = rho_inf
        self.am = (2.0*ri - 1.0) / (ri + 1.0)
        self.af = ri / (ri + 1.0)
        self.gamma = 0.5 - self.am + self.af
        self.beta = (1.0 - self.am + self.af)**2 / 4.0

        self.A_ext = np.pi/4 * static.OD**2

        print(f"\n{'='*40}")
        print(f"DynamicSolver")
        print(f"{'='*40}")
        print(f"Solving static equilibrium...")
        self._solve_static()
        print(f"Assembling FEM matrices...")
        self._assemble()
        print(f"Ready.\n")


    # Static solution
    def _solve_static(self):
        t0 = time.time()
        theta, x, y, T_node = self.st.solve(verbose=False)
        self._store_static(theta, x, y, T_node, time.time() - t0)

    def _store_static(self, theta, x, y, T_node, elapsed=0.0):
        self.theta0 = theta
        self.x0 = x
        self.y0 = y
        self.T0_node = T_node
        self.N_nodes = len(theta)
        self.N_el = self.N_nodes - 1
        self.N_dof = 3 * self.N_nodes
        contact = y >= self.st.WD
        self.contact = contact

        self._susp_idx = np.where(~contact)[0]
        tdp_candidates = np.where(~contact)[0]

        if len(tdp_candidates):
            self.tdp_idx = int(tdp_candidates[0])
        else:
            self.tdp_idx = self.N_nodes // 2

        e = np.arange(self.N_el)
        theta_mid = 0.5*(theta[e] + theta[e+1])
        alpha_e = -(theta_mid - 2*np.pi)
        self._cos_e = np.cos(alpha_e)
        self._sin_e = np.sin(alpha_e)
        theta_diff = theta[e+1] - theta[e]
        self._theta_diff = theta_diff
        self._valid_M = np.abs(theta_diff) <= np.radians(5)

        si = self._susp_idx
        alpha_nodes = -(theta[si] - 2*np.pi)
        self._susp_tx = np.cos(alpha_nodes)
        self._susp_tz = np.sin(alpha_nodes)

        print(f"Nodes={self.N_nodes}, DOFs={self.N_dof}, contact={int(np.sum(contact))}, susp={len(si)}, t={elapsed:.1f}s")

    # Element matrices
    def _element_K_M(self, alpha_e, T0_e):
        """6×6 global element stiffness and mass matrices."""
        st = self.st
        L = st.delta_s
        EI = st.EI
        EA = st.E * st.A_steel
        m = st.rho_s * st.A_steel + self.CA * st.rho_w * self.A_ext

        I_ring = np.pi/32 * (st.OD**4 - (st.OD - 2*st.WT)**4)
        Im = st.rho_s * I_ring

        k_ax = EA/L + T0_e/L
        k12 = 12*EI/L**3 + T0_e/L
        k6 = 6*EI/L**2
        k4 = 4*EI/L
        k2 = 2*EI/L

        Ke = np.array([[k_ax, 0, 0, -k_ax, 0, 0],
                       [0, k12, k6, 0, -k12, k6],
                       [0, k6, k4, 0, -k6, k2],
                       [-k_ax, 0, 0, k_ax, 0, 0],
                       [0, -k12, -k6, 0, k12, -k6],
                       [0, k6, k2, 0, -k6, k4]])

        mL = m * L
        Me = np.array([[mL/3, 0, 0, mL/6, 0, 0],
                       [0, 156*mL/420, 22*mL**2/420, 0, 54*mL/420, -13*mL**2/420],
                       [0, 22*mL**2/420, 4*mL**3/420, 0, 13*mL**2/420, -3*mL**3/420],
                       [mL/6, 0, 0, mL/3, 0, 0],
                       [0, 54*mL/420, 13*mL**2/420, 0, 156*mL/420, -22*mL**2/420],
                       [0, -13*mL**2/420, -3*mL**3/420, 0, -22*mL**2/420, 4*mL**3/420]])

        Me[2,2] += Im*L/3
        Me[2,5] += Im*L/6
        Me[5,2] += Im*L/6
        Me[5,5] += Im*L/3

        c = np.cos(alpha_e)
        s = np.sin(alpha_e)

        Tm = np.zeros((6,6))
        Tm[0,0] = c
        Tm[0,1] = s
        Tm[1,0] = -s
        Tm[1,1] = c
        Tm[2,2] = 1.0
        Tm[3,3] = c
        Tm[3,4] = s
        Tm[4,3] = -s
        Tm[4,4] = c
        Tm[5,5] = 1.0

        return Tm.T @ Ke @ Tm, Tm.T @ Me @ Tm

    # System assembly
    def _assemble(self):
        st = self.st
        N = self.N_dof
        K_g = np.zeros((N, N))
        M_g = np.zeros((N, N))

        for e in range(self.N_el):
            theta_mid = 0.5*(self.theta0[e] + self.theta0[e+1])
            alpha_e = -(theta_mid - 2*np.pi)
            T0_e = 0.5*(self.T0_node[e] + self.T0_node[e+1])
            Ke_g, Me_g = self._element_K_M(alpha_e, T0_e)
            idx = np.array([3*e, 3*e+1, 3*e+2, 3*(e+1), 3*(e+1)+1, 3*(e+1)+2])
            K_g[np.ix_(idx, idx)] += Ke_g
            M_g[np.ix_(idx, idx)] += Me_g

        C_g = self.ra_a * M_g + self.ra_b * K_g

        # Seabed spring-damper on contact nodes (Eqs. 24–25)
        for i in range(self.N_nodes):
            if self.contact[i]:
                w_node = st.w_s * st.delta_s
                m_node = (st.rho_s*st.A_steel + self.CA*st.rho_w*self.A_ext) * st.delta_s
                kc = w_node / self.delta_c
                cc = 2 * self.zeta_c * np.sqrt(m_node * kc)
                iz = 3*i + 1
                K_g[iz, iz] += kc
                C_g[iz, iz] += cc

        self.K_g = K_g
        self.M_g = M_g
        self.C_g = C_g

        # Boundary conditions:
        #   FP  (node 0)   : fixed ux  (DOF 0)
        #   LOP (node N-1) : fixed uz  (DOF 3*(N-1)+1)
        fixed = [0, 3*(self.N_nodes-1)+1]
        free = [i for i in range(N) if i not in fixed]
        self.free_dofs = free
        self.K_f = K_g[np.ix_(free, free)]
        self.M_f = M_g[np.ix_(free, free)]
        self.C_f = C_g[np.ix_(free, free)]
        print(f"Free DOFs={len(free)}, Rayleigh α={self.ra_a},  β={self.ra_b}")

    # Morison force
    def _morison_force(self, wave: WaveField, t, v_full, a_full):

        si = self._susp_idx
        if len(si) == 0:
            return np.zeros(self.N_dof)

        st = self.st
        D = st.OD
        A_m = np.pi/4*D**2
        S_F = np.pi*D
        rho = st.rho_w
        ds = st.delta_s

        xi = self.x0[si]
        zi = self.y0[si]
        vx_w, vz_w, ax_w, az_w = wave.kinematics_vec(xi, zi, t)

        vx_s = v_full[3*si]
        vz_s = v_full[3*si+1]

        ax_s = a_full[3*si]
        az_s = a_full[3*si+1]

        tx   = self._susp_tx
        tz   = self._susp_tz

        drx    = vx_w - vx_s
        drz = vz_w - vz_s
        dot_t  = drx*tx + drz*tz

        vr_x   = drx - dot_t*tx
        vr_z = drz - dot_t*tz

        vr_mag = np.hypot(vr_x, vr_z)
        vF_mag = np.abs(dot_t)

        dot_aw = ax_w*tx + az_w*tz
        apn_x  = ax_w - dot_aw*tx
        apn_z = az_w - dot_aw*tz

        fx = ds*(0.5*self.CD*rho*vr_mag*vr_x*D + self.CM*rho*apn_x*A_m + 0.5*self.CF*rho*vF_mag*(dot_t*tx)*S_F)
        fz = ds*(0.5*self.CD*rho*vr_mag*vr_z*D + self.CM*rho*apn_z*A_m + 0.5*self.CF*rho*vF_mag*(dot_t*tz)*S_F)

        F = np.zeros(self.N_dof)
        np.add.at(F, 3*si, fx)
        np.add.at(F, 3*si+1, fz)
        return F

    # Internal forces
    def _internal_forces(self, u_full):

        st = self.st
        ds = st.delta_s
        EI = st.EI
        EA = st.E * st.A_steel
        e = np.arange(self.N_el)

        i0 = 3*e
        i1 = 3*e+1
        i2 = 3*e+2
        i3 = 3*(e+1)
        i4 = 3*(e+1)+1
        i5 = 3*(e+1)+2

        c = self._cos_e
        s = self._sin_e

        u_ax1 = c*u_full[i0] + s*u_full[i1]
        u_tr1 = -s*u_full[i0] + c*u_full[i1]
        rot1 = u_full[i2]
        u_ax2 = c*u_full[i3] + s*u_full[i4]
        u_tr2 = -s*u_full[i3] + c*u_full[i4]
        rot2 = u_full[i5]

        M_dyn = EI/ds**2 * (6*u_tr1/ds - 6*u_tr2/ds + 2*rot1 + 2*rot2)
        T_dyn = EA * (u_ax2 - u_ax1) / ds
        M_static = np.where(self._valid_M, EI/ds * self._theta_diff, 0.0)
        T_static = 0.5*(self.T0_node[e] + self.T0_node[e+1])

        return M_static + M_dyn, T_static + T_dyn


    def run(self, wave: WaveField, dt, T_sim, output_nodes=None, print_interval=0.1):
        """
        Generalized-alpha Newmark time integration with fixed LOP.

        Parameters
        ----------
        wave           : WaveField
        dt             : float   time step [s]
        T_sim          : float   total simulation time [s]
        output_nodes   : list    node indices for full time history (default: TDP + LOP)
        print_interval : float   fraction of T_sim between progress prints

        Returns
        -------
        dict with keys:
            t, u_hist, M_hist, T_hist
            M_max, M_rms, T_max, T_rms
            output_nodes, x_nodes, y_nodes
        """
        n_steps = int(T_sim / dt)
        am = self.am
        af = self.af
        gm = self.gamma
        bt = self.beta

        K_eff = ((1-am)*self.M_f + (1-af)*gm*dt*self.C_f + (1-af)*bt*dt**2*self.K_f)
        K_lu = lu_factor(K_eff)

        u = np.zeros(self.N_dof)
        v = np.zeros(self.N_dof)
        acc = np.zeros(self.N_dof)
        free = self.free_dofs

        if output_nodes is None:
            output_nodes = sorted({self.tdp_idx, self.N_nodes-1})

        t_arr = np.linspace(0, T_sim, n_steps+1)
        u_hist = np.zeros((len(output_nodes)*3, n_steps+1))
        M_hist = np.zeros((self.N_el, n_steps+1))
        T_hist = np.zeros((self.N_el, n_steps+1))

        M0, T0 = self._internal_forces(u)

        M_hist[:,0] = M0
        T_hist[:,0] = T0

        print_step = max(1, int(print_interval*n_steps))
        t_wall = time.time()
        F_prev = self._morison_force(wave, 0., v, acc)

        print(f"Time integration: dt={dt}s, steps={n_steps}, T={T_sim:.0f}s")
        print(f"{'Step':>6}, {'t(s)':>7}, {'uz_TDP(mm)':>12}, {'elapsed':>8}")
        print(f"{'-'*40}")

        for step in range(n_steps):
            t_next = (step+1) * dt
            F_next = self._morison_force(wave, t_next, v, acc)

            u_f = u[free]
            v_f = v[free]
            acc_f = acc[free]

            u_p = u_f + dt*v_f + (0.5-bt)*dt**2*acc_f
            v_p = v_f + (1-gm)*dt*acc_f

            R_eff = ((1-af)*F_next[free] + af*F_prev[free]
                     - (1-af)*self.K_f@u_p  - af*self.K_f@u_f
                     - (1-af)*self.C_f@v_p  - af*self.C_f@v_f
                     - am*self.M_f@acc_f)

            acc_f_new = lu_solve(K_lu, R_eff)
            u[free] = u_p + bt*dt**2*acc_f_new
            v[free] = v_p + gm*dt*acc_f_new
            acc[free] = acc_f_new
            F_prev = F_next

            M_e, T_e = self._internal_forces(u)

            M_hist[:,step+1] = M_e
            T_hist[:,step+1] = T_e

            for k, ni in enumerate(output_nodes):
                u_hist[3*k:3*k+3, step+1] = u[3*ni:3*ni+3]

            if (step+1) % print_step == 0:
                print(f"{step+1:>6}, {t_next:>7.2f}"
                      f"{u[3*self.tdp_idx+1]*1000:>12.2f}"
                      f"{time.time()-t_wall:>7.1f}s")

        print(f"{'-'*40}")
        print(f"Completed in {time.time()-t_wall:.1f}s\n")

        skip = max(1, int(0.2*n_steps))

        M_w = M_hist[:,skip:]
        T_w = T_hist[:,skip:]

        M_max = np.max(np.abs(M_w), axis=1)
        T_max = np.max(T_w, axis=1)

        M_rms = np.sqrt(np.mean((M_w-np.mean(M_w,axis=1,keepdims=True))**2, axis=1))
        T_rms = np.sqrt(np.mean((T_w-np.mean(T_w,axis=1,keepdims=True))**2, axis=1))

        return dict(t=t_arr, u_hist=u_hist, M_hist=M_hist, T_hist=T_hist,
                    M_max=M_max, M_rms=M_rms, T_max=T_max, T_rms=T_rms,
                    output_nodes=output_nodes,
                    x_nodes=self.x0, y_nodes=self.y0)
