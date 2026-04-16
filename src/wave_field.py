"""
Airy linear wave kinematics for regular and irregular waves.

Supported spectrum types:
  - Regular      : single frequency (H, T)
  - Irregular    : JONSWAP random-phase superposition
  - Tabain       : single-parameter Adriatic Sea spectrum

Global coordinate convention: x horizontal, z positive downward.
"""

import numpy as np


class WaveField:
    """
    2D Airy wave kinematics.

    Parameters (set via regular_wave / irregular_wave / tabain_wave)
    -----------------------------------------------------------------
    WD   : float   water depth [m]
    g    : float   gravitational acceleration [m/s²]
    """

    def __init__(self, WD, g=9.81):
        self.WD = WD
        self.g = g
        self._type = None   # 'regular' or 'irregular'

    def _dispersion(self, omega):
        g = self.g
        d = self.WD

        k = omega**2 / g
        for _ in range(100):
            k = omega**2 / (g * np.tanh(k * d))
        return k

    def regular_wave(self, H, T_wave):
        """
        Single-frequency regular wave.

        Parameters
        ----------
        H      : float   wave height [m]
        T_wave : float   wave period [s]
        """
        self._type = 'regular'
        self.H = H
        self.T_wave = T_wave
        self.omega = 2*np.pi / T_wave
        self.k_wave = self._dispersion(self.omega)
        print(f"Wave: Regular  H={H}m  T={T_wave} s, λ={2*np.pi/self.k_wave:.1f} m")

    def irregular_wave(self, Hs, Tp, gamma=3.3, n_comp=50, seed=42):
        """
        Irregular JONSWAP wave via random-phase superposition.

        Parameters
        ----------
        Hs     : float   significant wave height [m]
        Tp     : float   peak period [s]
        gamma  : float   peak enhancement factor (default 3.3)
        n_comp : int     number of frequency components
        seed   : int     random seed for reproducibility
        """
        self._type = 'irregular'
        self.Hs = Hs
        self.Tp = Tp
        omega_p = 2*np.pi / Tp
        omega_arr = np.linspace(0.3*omega_p, 4.0*omega_p, n_comp+1)
        d_omega = omega_arr[1] - omega_arr[0]

        sigma = np.where(omega_arr <= omega_p, 0.07, 0.09)

        S_shape = (self.g**2 / omega_arr**5 * np.exp(-1.25*(omega_p/omega_arr)**4)
                   * gamma**np.exp(-0.5*((omega_arr-omega_p)/(sigma*omega_p))**2))

        alpha_norm = (Hs/4)**2 / np.trapezoid(S_shape, omega_arr)
        S = alpha_norm * S_shape

        rng = np.random.default_rng(seed)
        self._omega_c = omega_arr
        self._amp_c = np.sqrt(2.0 * S * d_omega)
        self._k_c = np.array([self._dispersion(w) for w in omega_arr])
        self._phase_c = rng.uniform(0, 2*np.pi, n_comp+1)
        Hs_check = 4*np.sqrt(np.trapezoid(S, omega_arr))

        print(f"Wave: JONSWAP  Hs={Hs}m, Tp={Tp}s, γ={gamma}, n={n_comp}, Hs_check={Hs_check:.3f}m")
        self._precompute_depth_factors()

    def tabain_wave(self, Hs, n_comp=50, seed=42):
        """
        Tabain (1997) single-parameter spectrum for the Adriatic Sea.
        Peak frequency derived from Hs: ωm = 0.32 + 1.8/(Hs + 0.6)

        Parameters
        ----------
        Hs     : float   significant wave height [m]
        n_comp : int     number of frequency components
        seed   : int     random seed
        """
        self._type = 'irregular'
        self.Hs = Hs
        omega_m = 0.32 + 1.8 / (Hs + 0.6)
        self.Tp = 2*np.pi / omega_m

        omega_arr = np.linspace(0.3*omega_m, 4.0*omega_m, n_comp+1)
        d_omega = omega_arr[1] - omega_arr[0]

        sigma = np.where(omega_arr <= omega_m, 0.08, 0.10)

        S = (0.862 * (0.0135 * self.g**2) / omega_arr**5 * np.exp(-5.186 / (omega_arr**4 * Hs**2))
             * 1.63**np.exp(-((omega_arr - omega_m)**2 / (2*sigma**2*omega_m**2))))

        rng = np.random.default_rng(seed)
        self._omega_c = omega_arr
        self._amp_c = np.sqrt(2.0 * S * d_omega)
        self._k_c = np.array([self._dispersion(w) for w in omega_arr])
        self._phase_c = rng.uniform(0, 2*np.pi, n_comp+1)
        Hs_check = 4*np.sqrt(np.trapezoid(S, omega_arr))
        print(f"Wave: Tabain  Hs={Hs} m, ωm={omega_m:.3f} rad/s, Tp={self.Tp:.2f} s  Hs_check={Hs_check:.3f} m")
        self._precompute_depth_factors()

    def _precompute_depth_factors(self):
        kd = self._k_c * self.WD
        self._deep = kd > 10.0
        kd_safe = np.where(self._deep, 1.0, kd)
        self._sinh_kd = np.where(self._deep, 1.0, np.sinh(kd_safe))

    def kinematics_vec(self, x_arr, z_arr, t):
        d    = self.WD
        z_up = -z_arr

        if self._type == 'regular':
            k = self.k_wave
            w = self.omega
            A = self.H / 2

            kd = k * d

            if kd > 10.0:
                cf = np.exp(k*z_up)
                sf = np.exp(k*z_up)
            else:
                cf = np.cosh(k*(z_up+d)) / np.sinh(kd)
                sf = np.sinh(k*(z_up+d)) / np.sinh(kd)
            ph = k*x_arr - w*t
            vx = A*w * cf * np.cos(ph)
            vz = -A*w * sf * np.sin(ph)
            ax = -A*w**2 * cf * np.sin(ph)
            az = A*w**2 * sf * np.cos(ph)
            return vx, vz, ax, az

        k_c = self._k_c[:,  None]
        w_c = self._omega_c[:, None]
        a_c = self._amp_c[:,  None]
        ph0 = self._phase_c[:, None]
        deep = self._deep[:, None]
        sinh_kd = self._sinh_kd[:, None]

        x_r = x_arr[None, :]
        z_r = z_up[None,  :]

        kzd = k_c * (z_r + d)
        cf = np.where(deep, np.exp(k_c*z_r), np.cosh(kzd) / sinh_kd)
        sf = np.where(deep, np.exp(k_c*z_r), np.sinh(kzd) / sinh_kd)

        ph = k_c*x_r - w_c*t + ph0

        vx  = np.sum( a_c*w_c * cf * np.cos(ph), axis=0)
        vz  = np.sum(-a_c*w_c * sf * np.sin(ph), axis=0)
        ax  = np.sum(-a_c*w_c**2 * cf * np.sin(ph), axis=0)
        az  = np.sum( a_c*w_c**2 * sf * np.cos(ph), axis=0)
        return vx, vz, ax, az

    def kinematics(self, x_pos, z_pos, t):
        if z_pos >= self.WD:
            return 0., 0., 0., 0.
        vx, vz, ax, az = self.kinematics_vec(np.array([x_pos]), np.array([z_pos]), t)
        return float(vx[0]), float(vz[0]), float(ax[0]), float(az[0])

    def surface_elevation(self, x_arr, t):

        if self._type == 'regular':
            return (self.H/2) * np.cos(self.k_wave*x_arr - self.omega*t)

        k_c = self._k_c[:, None]
        w_c = self._omega_c[:, None]
        a_c = self._amp_c[:, None]
        ph0 = self._phase_c[:, None]
        ph = k_c*x_arr[None, :] - w_c*t + ph0
        return np.sum(a_c * np.cos(ph), axis=0)
