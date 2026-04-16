"""
Test case 1
"""

import os
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

from utils import (build_solvers, build_wave, build_vessel, build_dnv,
                   run_analysis, run_dnv_check, run_scatter, save_scatter,
                   make_plots, make_animation, print_summary)


# Folder paths
RAO_DATA_DIR = 'ship_data' # folder with A.txt B.txt C.txt M.txt omega.txt Fw.txt
OUTPUT_DIR  = 'results' # folder for output figures and .npy files

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Run options:

# COMPARE_MODE:
# 'waves_only' — fixed LOP, no vessel motion
# 'rao_only' — waves + vessel RAO at LOP
# 'compare' — run both, show side-by-side comparison
COMPARE_MODE = 'compare'

# DNV_SCATTER:
# False — single sea state DNV check only
# True — also sweep over Hs_scatter / Tp_scatter grid
DNV_SCATTER = False

# Scatter grid (only used when DNV_SCATTER = True)
Hs_scatter = np.arange(0.5, 7.0, 0.5) # m
Tp_scatter = np.arange(3.0, 15.0, 1.0)  # s

# Plots and animation
MAKE_PLOTS = True
RUN_ANIMATION = True
ANIM_EVERY = 2
ANIM_FPS = 20

# Derived mode flags — do not edit
RUN_NO_RAO = COMPARE_MODE in ('waves_only', 'compare')
RUN_RAO = COMPARE_MODE in ('rao_only', 'compare')


# Vessel data
X_COG = 71.25 # m from bow — vessel center of gravity
X_STINGER_TIP = 155.1 # m from bow — pipe departure point

PITCH_MIN_ZETA = 0.05
USE_HEAVE = True
USE_SURGE = True
USE_PITCH = False


# Pipeline data
OD = 0.25
WT = 0.02
E = 2.07e11
rho_w = 1025.0
w_s = 631.13
A_steel = np.pi/4 * (OD**2 - (OD - 2*WT)**2)
rho_s_eff = rho_w + w_s / (9.81 * A_steel)

PHI0_DEG = 5.1
P0 = 50e3
WD = 250.0
ks = 4e3
ds = 1.0
beta_ray = 4.1e-4

PIPE_PARAMS = dict(OD=OD, WT=WT, E=E, rho_s=rho_s_eff,
                   phi0=np.radians(PHI0_DEG), P0=P0,
                   WD=WD, ks=ks, delta_s=ds)

DYN_KWARGS = dict(CD=1.2, CM=2.0, CA=1.0, CF=0.05,
                  zeta_seabed=0.05, delta_penetration=0.05,
                  rayleigh_alpha=0.0, rayleigh_beta=beta_ray,
                  rho_inf=0.8)

# Stinger mask: zero M in the last ~15m near LOP (current fix for numerical BC artifact)
N_STINGER = max(3, int(15.0 / ds))


# Wave data
WAVE_TYPE = 'irregular'
Hs = 6.0
Tp = 12.0
N_COMP = 50
SEED = 42

DT = 0.25
T_SIM = max(600.0, 10.0 * Tp)

# DNV ST-F101 data
SMYS = 450.0
SMTS = 535.0
steel_grade = 'CMn'
T_e = 20.0
a_u = 0.96
Y_m = 1.15
Safety_class = 'Medium'
pipe_fab = 'UOE'
pipe_type = 'SMLS'
Condition = 'Installation'
cond_name = 'Otherwise'
C_a = 0.0
E_a = 0.0
O_o = 1.0
v = 0.3
p_i = 0.0
p_min = 0.0

lay_type = 'J-lay' if PHI0_DEG >= 50 else 'S-lay'

PARAMS = dict(PHI0_DEG=PHI0_DEG, _P0=P0, _WD=WD,
              Hs=Hs, Tp=Tp, WAVE_TYPE=WAVE_TYPE,
              COMPARE_MODE=COMPARE_MODE,
              USE_HEAVE=USE_HEAVE, USE_SURGE=USE_SURGE, USE_PITCH=USE_PITCH,
              lay_type=lay_type, Safety_class=Safety_class)

if __name__ == '__main__':

    # 1. Pipe capacities
    cap = build_dnv(OD=OD, WT=WT, E=E, SMYS=SMYS, SMTS=SMTS, WD=WD,
                    steel_grade=steel_grade, T_e=T_e,
                    a_u=a_u, Y_m=Y_m, Safety_class=Safety_class,
                    pipe_fab=pipe_fab, pipe_type=pipe_type,
                    Condition=Condition, C_a=C_a, E_a=E_a, O_o=O_o, v=v,
                    p_i=p_i, p_min=p_min,condition_name=cond_name, rho_w=rho_w)

    # 2. Vessel RAOs
    vessel = build_vessel(rao_data_dir=RAO_DATA_DIR,
                          X_COG=X_COG, X_STINGER_TIP=X_STINGER_TIP,
                          pitch_min_zeta=PITCH_MIN_ZETA,
                          use_heave=USE_HEAVE, use_surge=USE_SURGE, use_pitch=USE_PITCH)

    # 3. Wave
    wave = build_wave(WAVE_TYPE, Hs, Tp, WD, n_comp=N_COMP, seed=SEED)

    # 4. Dynamic FEM solver
    dyn = build_solvers(PIPE_PARAMS, DYN_KWARGS)

    # 5. Dynamic analysis
    res_no_rao, res_rao = run_analysis(dyn, wave, vessel,
                                       run_no_rao=RUN_NO_RAO, run_rao=RUN_RAO,
                                       dt=DT, T_sim=T_SIM, n_stinger=N_STINGER)

    # 6. DNV LCC check
    lcc_a, lcc_b, gov_label = run_dnv_check(dyn, res_no_rao, res_rao, cap,
                                            run_no_rao=RUN_NO_RAO, run_rao=RUN_RAO,
                                            n_stinger=N_STINGER)

    # 7. Console summary
    print_summary(dyn, res_no_rao, res_rao,lcc_a, lcc_b, gov_label, COMPARE_MODE, PARAMS)

    # 8. Plots
    if MAKE_PLOTS:
        make_plots(dyn, res_no_rao, res_rao, lcc_a, lcc_b, gov_label,
                   vessel, PARAMS, OUTPUT_DIR)

    # 9. Scatter sweep
    if DNV_SCATTER:
        LCC_A_nrao, LCC_B_nrao, LCC_A_rao, LCC_B_rao = run_scatter(dyn, vessel, cap,
                                                                   Hs_scatter, Tp_scatter,
                                                                   run_no_rao=RUN_NO_RAO, run_rao=RUN_RAO,
                                                                   dt=DT, n_stinger=N_STINGER,
                                                                   n_comp=N_COMP, seed=SEED)

        save_scatter(OUTPUT_DIR, Hs_scatter, Tp_scatter, LCC_A_nrao, LCC_B_nrao, LCC_A_rao, LCC_B_rao)

        if MAKE_PLOTS:
            make_plots(dyn, res_no_rao, res_rao,
                       lcc_a, lcc_b, gov_label,
                       vessel, PARAMS, OUTPUT_DIR,
                       dnv_scatter=True,
                       Hs_scatter=Hs_scatter, Tp_scatter=Tp_scatter,
                       LCC_A_nrao=LCC_A_nrao, LCC_B_nrao=LCC_B_nrao,
                       LCC_A_rao=LCC_A_rao, LCC_B_rao=LCC_B_rao)

    # 10. Animation
    if RUN_ANIMATION:
        make_animation(dyn, wave, vessel, RUN_RAO,
                       dt=DT, T_sim=T_SIM, n_stinger=N_STINGER,
                       fps=ANIM_FPS, every=ANIM_EVERY,
                       output_dir=OUTPUT_DIR)
