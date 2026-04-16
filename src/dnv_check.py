"""
DNV ST-F101 (2021) combined loading criterion (LCC) check.

Reference:
    DNV ST-F101, Submarine Pipeline Systems, 2021.
    Section 5.4.6 — Local buckling, combined loading.

Equations implemented:
    Capacity:
        Burst pressure          eq. 5.9
        Collapse pressure       eq. 5.13
        Moment capacity         eq. 5.26
        Axial force capacity    eq. 5.27
    LCC:
        Internal overpressure   eq. 5.24
        External overpressure   eq. 5.34

Design load effects:
    M_sd = M_static × Y_F × Y_c + (M_dyn - M_static) × Y_E    eq. 4.6
    S_sd = T_static × Y_F × Y_c + (T_dyn - T_static) × Y_E    eq. 4.8

Load cases (ULS, Table 4-4):
    LC-A  system effects present     : Y_F=1.2, Y_E=0.7
    LC-B  system effects not present : Y_F=1.1, Y_E=1.3

Units throughout: MPa, kN, kNm, mm (as in DNV equations)
"""

import math
import numpy as np
from scipy.interpolate import interp1d


class DNVCapacities:
    """
    Parameters
    ----------
    OD             : float   outer diameter [m]
    WT             : float   wall thickness [m]
    E              : float   Young's modulus [Pa]
    SMYS           : float   specified minimum yield stress [MPa]
    SMTS           : float   specified minimum tensile stress [MPa]
    steel_grade    : str     'CMn', '13Cr', '22Cr', or '25Cr'
    T_e            : float   design temperature [°C]
    a_u            : float   material strength factor (Table 5-3) 0.96 for 'Other' loading, 1.0 for pressure test
    Y_m            : float   material resistance factor (Table 5-1) 1.15 for ULS
    Safety_class   : str     'Low', 'Medium', or 'High'
    pipe_fab       : str     'Seamless','UO','JCO','TRB','ERW','HFW','UOE','JCOE'
    pipe_type      : str     'SMLS' or 'welded'  (for fabrication tolerance)
    Condition      : str     'Installation' or 'Operation'
    C_a            : float   corrosion allowance [mm]
    E_a            : float   erosion allowance [mm]
    O_o            : float   ovality [%]
    v              : float   Poisson ratio
    p_i            : float   internal pressure [MPa]
    p_min          : float   minimum internal pressure [MPa]
    condition_name : str     installation condition for Y_c (Table 4-5)
    rho_w          : float   seawater density [kg/m³]
    """

    # Temperature de-rating data [°C, MPa]
    _STEEL_DERATE = {'CMn':  [(20,0),(50,0),(100,30),(200,70)],
                     '13Cr': [(20,0),(50,0),(100,30),(200,70)],
                     '22Cr': [(20,0),(100,90),(200,140)],
                     '25Cr': [(20,0),(100,90),(200,140)]}

    _Y_C_TABLE = {'Pipeline resting on uneven seabed': 1.07,
                  'J-tube pull-in': 0.82,
                  'System pressure test': 0.93,
                  'S-lay installation - Local buckling load control check on stinger': 0.80,
                  'Reeling installation - Displacement controlled check - seamless pipes': 0.77,
                  'Reeling installation - Displacement controlled check - welded pipes': 0.82,
                  'Otherwise': 1.00}

    _SC_TABLE = {'Low': {'PD': 1.046, 'LB': 1.04, 'DC': 2.0},
                 'Medium': {'PD': 1.138, 'LB': 1.14, 'DC': 2.5},
                 'High': {'PD': 1.308, 'LB': 1.26, 'DC': 3.3}}

    _FAB_FACTOR = {'Seamless':1.0, 'UO':0.93, 'JCO':0.93, 'TRB':0.93,
                   'ERW':0.93, 'HFW':0.93, 'UOE':0.85, 'JCOE':0.85}

    def __init__(self,
                 OD, WT, E, SMYS, SMTS,
                 steel_grade='CMn',
                 T_e=20.0,
                 a_u=0.96,
                 Y_m=1.15,
                 Safety_class='Medium',
                 pipe_fab='UOE',
                 pipe_type='SMLS',
                 Condition='Installation',
                 C_a=0.0,
                 E_a=0.0,
                 O_o=1.0,
                 v=0.3,
                 p_i=0.0,
                 p_min=0.0,
                 condition_name='Otherwise',
                 rho_w=1025.0):

        self.rho_w = rho_w
        self.p_i = p_i
        self.p_min = p_min

        D_o = OD * 1000
        D_in = (OD - 2*WT) * 1000
        E_GPa = E / 1e9

        # Safety class factors
        sc = self._SC_TABLE[Safety_class]
        self.Y_m = Y_m
        self.Y_SCPD = sc['PD']
        self.Y_SCLB = sc['LB']
        self.Y_SCDC = sc['DC']

        # Condition load effect factor
        self.Y_c = self._Y_C_TABLE.get(condition_name, 1.00)

        # Temperature de-rating
        pts = self._STEEL_DERATE[steel_grade]
        T_arr, S_arr = zip(*pts)
        derate = float(interp1d(T_arr, S_arr, kind='linear')(max(min(T_e, 200), 20)))
        f_y = (SMYS - derate) * a_u
        f_u = (SMTS - derate) * a_u
        self.f_y = f_y
        self.f_u = f_u

        # Fabrication tolerance (Table 7-17/7-18)
        def t_fab_fn(D, pt):
            if pt.lower() == 'smls':
                if D < 66.7:
                    return 0.5
                elif D <= 610:
                    return 0.0075 * D
                else:
                    return 0.01 * D
            else:
                if D < 66.7:
                    return 0.5
                elif D < 426.7:
                    return 0.0075*D
                elif D <= 610:
                    return 3.2
                elif D < 800:
                    return 0.005*D
                else:
                    return 4.0

        t_fab = t_fab_fn(D_o, pipe_type)
        t_nom = (D_o - D_in) / 2

        if Condition == 'Installation':
            t_2 = t_nom
            t_1 = t_nom - t_fab
        else:
            t_2 = t_nom - C_a - E_a
            t_1 = t_nom - t_fab - C_a - E_a

        if D_o / t_2 > 45:
            raise ValueError(f"D/t2 = {D_o/t_2:.1f} > 45 — outside DNV range")

        a_fab = self._FAB_FACTOR[pipe_fab]
        O_oref = O_o / 100.0

        # Burst pressure (eq 5.9) [MPa]
        f_cb = min(f_y, f_u / 1.15)
        p_bt = (2*t_2) / (D_o - t_2) * f_cb * 2/math.sqrt(3)

        # Combined loading factors (eq 5.25)
        beta_t2 = (60 - D_o/t_2) / 90
        a_c_t2 = 1 + beta_t2 * (f_u/f_y - 1)

        # Moment capacity (eq 5.26) [kNm]
        M_p = f_y * (D_o - t_2)**2 * t_2 / 1e6
        M_ct = a_c_t2 * M_p

        # Axial capacity (eq 5.27) [kN]
        S_p = f_y * math.pi * (D_o - t_2) * t_2 / 1e3
        S_ct = a_c_t2 * S_p

        # Collapse pressure (eq 5.13) [MPa]
        p_el = (2.0 * E_GPa*1e3 * (t_2/D_o)**3) / (1.0 - v**2)
        p_p = f_y * a_fab * (2*t_2/D_o)
        p_c = self._solve_collapse(p_el, p_p, O_oref, D_o, t_2)

        self.D_o = D_o
        self.t_nom = t_nom
        self.t_fab = t_fab
        self.t_1 = t_1
        self.t_2 = t_2
        self.a_fab = a_fab
        self.p_bt = p_bt
        self.p_c = p_c
        self.M_ct = M_ct
        self.S_ct = S_ct
        self.beta_t2 = beta_t2
        self.a_c_t2 = a_c_t2

    @staticmethod
    def _solve_collapse(p_el, p_p, O_ov, D, t):
        b = -p_el
        c = -(p_p**2 + p_p*p_el*O_ov*(D/t))
        d = p_el * p_p**2
        u = (1/3)*((-1/3)*b**2 + c)
        vv = 0.5*((2/27)*b**3 - (1/3)*b*c + d)
        phi = math.acos(max(-1.0, min(1.0, -vv/math.sqrt(-u**3))))
        return -2*math.sqrt(-u)*math.cos(phi/3 + math.pi/3) - b/3

    def print_summary(self, WD):
        p_e_sb = WD * self.rho_w * 9.81 / 1e6
        print(f"DNV capacities:")
        print(f"f_y={self.f_y:.1f} MPa, f_u={self.f_u:.1f} MPa")
        print(f"t_nom={self.t_nom:.1f} mm, t_2={self.t_2:.3f}mm, t_fab={self.t_fab:.3f} mm")
        print(f"p_bt={self.p_bt:.3f} MPa, p_c={self.p_c:.3f} MPa")
        print(f"M_ct={self.M_ct:.1f} kNm, S_ct={self.S_ct:.1f} kN")
        print(f"beta={self.beta_t2:.4f}, a_c={self.a_c_t2:.4f}")
        print(f"Y_m={self.Y_m}, Y_SCLB={self.Y_SCLB}, Y_c={self.Y_c}")
        print(f"p_e at {WD:.0f} m = {p_e_sb:.3f} MPa, (p_c/p_e = {self.p_c/p_e_sb:.2f})")


def compute_lcc(M_static_kNm, T_static_kN, M_dyn_kNm, T_dyn_kN,
                y_el_m, Y_F, Y_E, cap: DNVCapacities):
    """
    DNV ST-F101 LCC at every pipe element.

    Parameters
    ----------
    M_static_kNm : ndarray (N_el,)  static bending moment [kNm]
    T_static_kN  : ndarray (N_el,)  static tension [kN]
    M_dyn_kNm    : ndarray (N_el,)  max dynamic bending moment [kNm]
    T_dyn_kN     : ndarray (N_el,)  max dynamic tension [kN]
    y_el_m       : ndarray (N_el,)  element midpoint depth [m]
    Y_F          : float            functional load factor
    Y_E          : float            environmental load factor
    cap          : DNVCapacities

    Returns
    -------
    lcc : ndarray (N_el,) LCC value per element.  Limit = 1.0.
    """
    p_e  = y_el_m * cap.rho_w * 9.81 / 1e6   # MPa
    M_sd = M_static_kNm * Y_F * cap.Y_c + (M_dyn_kNm - M_static_kNm) * Y_E
    S_sd = T_static_kN  * Y_F * cap.Y_c + (T_dyn_kN  - T_static_kN)  * Y_E

    lcc = np.zeros(len(p_e))
    for i in range(len(p_e)):
        pe_i = p_e[i]
        pi_pe = cap.p_i - pe_i      # positive = internal overpressure
        Msd_i = float(np.abs(M_sd[i]))
        Ssd_i = float(S_sd[i])

        # Moment + axial interaction (common to both eqs 5.24 and 5.34)
        term_MS = (cap.Y_m * cap.Y_SCLB * Msd_i / cap.M_ct
                   + (cap.Y_m * cap.Y_SCLB * Ssd_i / cap.S_ct)**2)

        if pi_pe >= 0:
            # Internal overpressure — eq 5.24
            ratio = pi_pe / cap.p_bt
            Y_p_i = (1 - cap.beta_t2) if ratio <= 2/3 else \
                     1 - 3*cap.beta_t2*(1 - ratio)*ratio
            term_P = Y_p_i * (cap.p_i - pe_i) / (cap.a_c_t2 * cap.p_bt)
        else:
            # External overpressure — eq 5.34
            term_P = cap.Y_m * cap.Y_SCLB * (pe_i - cap.p_min) / cap.p_c

        lcc[i] = term_MS**2 + term_P**2

    return lcc
