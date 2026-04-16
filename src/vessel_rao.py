"""
6-DOF vessel RAO computation and LOP motion functions.

Solves the frequency-domain equation of motion:
    [-ω²(M + A(ω)) + iω·B(ω) + C] X(ω) = F(ω)

DOF convention:
    0 = surge  (x translation)
    1 = sway   (y translation)
    2 = heave  (z translation)
    3 = roll   (x rotation)
    4 = pitch  (y rotation)
    5 = yaw    (z rotation)

Units expected in input files:
    Mass matrix M   : tonnes
    Added mass A    : tonnes
    Damping B       : ts/m  (or tm/rad for rotational DOFs)
    Stiffness C     : t/m   (or t/rad)
    Excitation Fw   : kN    (or kNm for moments)
    Frequencies     : rad/s

Notes on pitch RAO:
    The pitch RAO near resonance right now is unrealistically large.
    A minimum critical damping fraction (pitch_min_zeta) can be added to regularize the resonance peak.
    For now set USE_PITCH = False.
"""

import os
import numpy as np
from scipy.interpolate import interp1d


class VesselRAO:
    """
    Parameters
    ----------
    data_path      : str    folder containing A.txt B.txt C.txt M.txt omega.txt Fw.txt
    X_COG          : float  x-coordinate of CoG from vessel reference [m]
    X_STINGER_TIP  : float  x-coordinate of pipe departure point [m]
    pitch_min_zeta : float  minimum pitch damping as fraction of critical
    use_heave      : bool   include heave contribution at LOP
    use_surge      : bool   include surge contribution at LOP
    use_pitch      : bool   include pitch lever-arm contribution
    """

    def __init__(self, data_path,
                 X_COG=71.25, X_STINGER_TIP=98.0,
                 pitch_min_zeta=0.05,
                 use_heave=True, use_surge=True, use_pitch=False):

        self.data_path = data_path
        self.X_STERN = X_STINGER_TIP - X_COG
        self.pitch_min_zeta = pitch_min_zeta
        self.use_heave = use_heave
        self.use_surge = use_surge
        self.use_pitch = use_pitch

        self.omega_rao = None
        self.Xw_rao = None

        self._load_and_solve()


    def _load_and_solve(self):
        def load(fname):
            return np.loadtxt(os.path.join(self.data_path, fname))

        A_raw = load('A.txt')
        B_raw = load('B.txt')
        C = load('C.txt')
        M = load('M.txt')
        omega = load('omega.txt')
        Fw = load('Fw.txt')

        N = len(omega)
        A_t = np.zeros((6,6,N))
        B_t = np.zeros((6,6,N))

        for i in range(N):
            A_t[:,:,(N-1-i)] = A_raw[(6*i):(6*(i+1)),:]
            B_t[:,:,(N-1-i)] = B_raw[(6*i):(6*(i+1)),:]

        # Complex excitation force
        F = np.zeros((6, N), dtype=complex)
        for i in range(N):
            for j in range(6):
                F[j,i] = Fw[i,j*2] * np.exp(1j * np.deg2rad(Fw[i,j*2+1]))

        # Minimum pitch damping to regularise near-resonance response; still questionable
        if self.pitch_min_zeta > 0:
            for i, w in enumerate(omega):
                M44 = M[4,4] + A_t[4,4,i]
                C44 = C[4,4]
                if C44 > 0 and M44 > 0:
                    B_crit = 2 * self.pitch_min_zeta * np.sqrt(M44 * C44)
                    if B_t[4,4,i] < B_crit:
                        B_t[4,4,i] = B_crit

        Xw = np.zeros((6, N), dtype=complex)
        for i, w in enumerate(omega):
            K = -w**2*(M + A_t[:,:,i]) + 1j*w*B_t[:,:,i] + C
            Xw[:,i] = np.linalg.solve(K, F[:,i])

        self.omega_rao = omega
        self.Xw_rao = Xw

        print(f"RAO: {N} frequency points"
              f"ω={omega.min():.3f}…{omega.max():.3f} rad/s"
              f"X_STERN={self.X_STERN:.2f}m")

    def print_summary(self, Tp):
        wp = 2*np.pi / Tp
        idx = np.argmin(np.abs(self.omega_rao - wp))
        print(f"  RAO at Tp={Tp}s (ω={self.omega_rao[idx]:.3f} rad/s):")
        names = ['surge','sway','heave','roll','pitch','yaw']
        for d, name in enumerate(names):
            amp = abs(self.Xw_rao[d, idx])
            ph = np.degrees(np.angle(self.Xw_rao[d, idx]))
            if amp > 1e-6:
                print(f"{name:6s}: {amp:.4f},  phase={ph:.1f}°")


    def build_lop_functions(self, wave):
        """
        LOP motion is computed as a superposition over wave components:
            u_LOP(t) = Σ_n  ζ_n · |RAO(ω_n)| · cos(ω_n·t + φ_n + arg(RAO(ω_n)))

        Parameters
        ----------
        wave : WaveField   (must be set up before calling this)

        Returns
        -------
        (lop_displacement, lop_velocity, lop_acceleration) each is a callable f(t) → (ux, uz, theta)
        """
        if wave._type == 'regular':
            omega_w = np.array([wave.omega])
            amp_w   = np.array([wave.H / 2])
            phase_w = np.array([0.0])
        else:
            omega_w = wave._omega_c
            amp_w   = wave._amp_c
            phase_w = wave._phase_c

        def _interp(dof):
            ow = np.clip(omega_w, self.omega_rao.min(), self.omega_rao.max())
            re = interp1d(self.omega_rao, self.Xw_rao[dof].real, kind='linear',
                          bounds_error=False,
                          fill_value=(self.Xw_rao[dof].real[0],
                                      self.Xw_rao[dof].real[-1]))
            im = interp1d(self.omega_rao, self.Xw_rao[dof].imag, kind='linear',
                          bounds_error=False,
                          fill_value=(self.Xw_rao[dof].imag[0],
                                      self.Xw_rao[dof].imag[-1]))
            return re(ow) + 1j*im(ow)

        rao_surge = _interp(0)
        rao_heave = _interp(2)
        rao_pitch = _interp(4)

        _amp_surge  = amp_w * np.abs(rao_surge)
        _ang_surge  = np.angle(rao_surge)
        _amp_heave  = amp_w * np.abs(rao_heave)
        _ang_heave  = np.angle(rao_heave)
        _amp_pitch  = amp_w * np.abs(rao_pitch)
        _ang_pitch  = np.angle(rao_pitch)
        _x_stern    = self.X_STERN
        _use_heave  = self.use_heave
        _use_surge  = self.use_surge
        _use_pitch  = self.use_pitch

        def lop_displacement(t):
            pt = omega_w * t + phase_w
            ux = uz = theta = 0.0
            if _use_surge:
                ux = np.sum(_amp_surge * np.cos(pt + _ang_surge))
            if _use_heave:
                uz = np.sum(_amp_heave * np.cos(pt + _ang_heave))
            if _use_pitch:
                theta = np.sum(_amp_pitch * np.cos(pt + _ang_pitch))
                uz += -theta * _x_stern
            return ux, uz, theta

        def lop_velocity(t, dt=0.01):
            up = lop_displacement(t + dt)
            um = lop_displacement(t - dt)
            return tuple((p - m) / (2*dt) for p, m in zip(up, um))

        def lop_acceleration(t, dt=0.01):
            u0 = lop_displacement(t)
            up = lop_displacement(t + dt)
            um = lop_displacement(t - dt)
            return tuple((p - 2*c + m) / dt**2 for p, c, m in zip(up, u0, um))

        return lop_displacement, lop_velocity, lop_acceleration

    def check_lop_motion(self, wave, n_cycles=5):
        lop_d, _, _ = self.build_lop_functions(wave)
        Tp = getattr(wave, 'Tp', getattr(wave, 'T_wave', 10.0))
        t_arr = np.linspace(0, n_cycles*Tp, 1000)
        uz = np.array([lop_d(t)[1] for t in t_arr])
        ux = np.array([lop_d(t)[0] for t in t_arr])

        print(f"LOP motion estimate ({n_cycles} cycles):")
        print(f"Heave: RMS={np.std(uz)*1000:.0f} mm, max={np.max(np.abs(uz))*1000:.0f} mm")
        print(f"Surge: RMS={np.std(ux)*1000:.0f} mm, max={np.max(np.abs(ux))*1000:.0f} mm")

        if np.max(np.abs(uz)) > 5.0:
            print("WARNING: LOP heave > 5m — check pitch RAO / damping settings")
