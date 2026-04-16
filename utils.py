"""
Execution helpers for the pipeline installation analysis.

All print statements, solver construction, dynamic runs, DNV checks,
scatter sweeps and plot calls live here. run_case.py only sets
parameters and calls these functions.

Functions
---------
build_solvers()          Build StaticSolver + DynamicSolver, return dyn
build_wave()             Build WaveField from parameters
build_vessel()           Build VesselRAO from parameters
build_dnv()              Build DNVCapacities from parameters
run_analysis()           Run waves-only and/or RAO dynamic analysis
run_dnv_check()          Compute LCC on analysis results
run_scatter()            Sweep Hs/Tp grid, return LCC grids
save_scatter()           Save scatter .npy files
make_plots()             Call all plot functions
print_summary()          Print results table to console
"""

import os
import time
import numpy as np

from src import (StaticSolver, WaveField, DynamicSolver,
                 VesselRAO, run_with_rao, run_animation_with_rao,
                 DNVCapacities, compute_lcc, plot_summary,
                 plot_lcc_profile, plot_lcc_scatter)

# Helper functions
def build_solvers(pipe_params, dyn_kwargs):
    print(f"\n{'='*40}")
    print(f"Building static solver and FEM model...")
    print(f"{'='*40}")
    t0 = time.time()
    static = StaticSolver(**pipe_params)
    dyn = DynamicSolver(static, **dyn_kwargs)
    print(f"Ready in {time.time()-t0:.1f}s\n")
    return dyn


def build_wave(wave_type, Hs, Tp, WD, n_comp=50, seed=42):
    print(f"Building wave: {wave_type.upper()}, Hs/H={Hs}m, Tp/T={Tp}s ...")
    wave = WaveField(WD=WD)
    if wave_type == 'irregular':
        wave.irregular_wave(Hs=Hs, Tp=Tp, gamma=3.3, n_comp=n_comp, seed=seed)
    elif wave_type == 'regular':
        wave.regular_wave(H=Hs, T_wave=Tp)
    elif wave_type == 'tabain':
        wave.tabain_wave(Hs=Hs, n_comp=n_comp, seed=seed)
    else:
        raise ValueError(f"Unknown wave_type '{wave_type}'. Use 'irregular','regular', or 'tabain'.")
    return wave


def build_vessel(rao_data_dir, X_COG, X_STINGER_TIP,
                 pitch_min_zeta=0.05,
                 use_heave=True, use_surge=True, use_pitch=False):

    print("Loading vessel RAOs...")
    vessel = VesselRAO(data_path=rao_data_dir,
                       X_COG=X_COG, X_STINGER_TIP=X_STINGER_TIP,
                       pitch_min_zeta=pitch_min_zeta,
                       use_heave=use_heave, use_surge=use_surge, use_pitch=use_pitch)
    return vessel


def build_dnv(OD, WT, E, SMYS, SMTS, WD,
              steel_grade='CMn', T_e=20.0,
              a_u=0.96, Y_m=1.15,
              Safety_class='Medium',
              pipe_fab='UOE', pipe_type='SMLS',
              Condition='Installation',
              C_a=0.0, E_a=0.0, O_o=1.0, v=0.3,
              p_i=0.0, p_min=0.0,
              condition_name='Otherwise',
              rho_w=1025.0):

    print("Computing DNV ST-F101 pipe capacities...")

    cap = DNVCapacities(OD=OD, WT=WT, E=E, SMYS=SMYS, SMTS=SMTS,
                        steel_grade=steel_grade, T_e=T_e,
                        a_u=a_u, Y_m=Y_m,
                        Safety_class=Safety_class,
                        pipe_fab=pipe_fab, pipe_type=pipe_type,
                        Condition=Condition, C_a=C_a, E_a=E_a, O_o=O_o, v=v,
                        p_i=p_i, p_min=p_min,
                        condition_name=condition_name,
                        rho_w=rho_w)
    cap.print_summary(WD)
    return cap


# Dynamic analysis
def run_analysis(dyn, wave, vessel, run_no_rao, run_rao, dt, T_sim, n_stinger):

    res_no_rao = None
    res_rao = None

    if run_no_rao:
        print("\n--- Waves only (fixed LOP) ---")
        res_no_rao = dyn.run(wave, dt=dt, T_sim=T_sim,output_nodes=[dyn.tdp_idx, dyn.N_nodes-1],
                             print_interval=0.25)

        _apply_mask(res_no_rao, n_stinger)

    if run_rao:
        print("\n--- Waves + vessel RAO at LOP ---")
        lop_disp, lop_vel, lop_acc = vessel.build_lop_functions(wave)
        vessel.check_lop_motion(wave)
        res_rao = run_with_rao(
            dyn, wave, lop_disp, lop_vel, lop_acc,
            dt=dt, T_sim=T_sim,
            n_stinger=n_stinger, print_interval=0.25)

    return res_no_rao, res_rao


def _apply_mask(res, n_stinger):
    if n_stinger > 0:
        res['M_max'][-n_stinger:] = 0.0
        res['M_hist'][-n_stinger:,:] = 0.0


# DNV LCC
def run_dnv_check(dyn, res_no_rao, res_rao, cap, run_no_rao, run_rao, n_stinger):

    u_zero = np.zeros(dyn.N_dof)
    M_static, T_static = dyn._internal_forces(u_zero)
    if n_stinger > 0:
        M_static[-n_stinger:] = 0.0
    M_static_kNm = M_static / 1e3
    T_static_kN  = T_static / 1e3
    y_el = 0.5*(dyn.y0[:-1]+dyn.y0[1:])

    res_gov = res_rao if run_rao else res_no_rao
    gov_label = 'Waves+RAO' if run_rao else 'Waves only'
    M_dyn_kNm = res_gov['M_max'] / 1e3
    T_dyn_kN = res_gov['T_max'] / 1e3

    lcc_a = compute_lcc(M_static_kNm, T_static_kN, M_dyn_kNm, T_dyn_kN,
                        y_el, Y_F=1.2, Y_E=0.7, cap=cap)

    lcc_b = compute_lcc(M_static_kNm, T_static_kN, M_dyn_kNm, T_dyn_kN,
                        y_el, Y_F=1.1, Y_E=1.3, cap=cap)
    if n_stinger > 0:
        lcc_a[-n_stinger:] = 0.0
        lcc_b[-n_stinger:] = 0.0

    return lcc_a, lcc_b, gov_label


# Wave scatter build for DNV LCC calculation
def run_scatter(dyn, vessel, cap, Hs_scatter, Tp_scatter,
                run_no_rao, run_rao, dt, n_stinger, n_comp, seed):

    WD = dyn.st.WD
    n_hs = len(Hs_scatter)
    n_tp = len(Tp_scatter)

    LCC_A_nrao = np.full((n_tp, n_hs), np.nan) if run_no_rao else None
    LCC_B_nrao = np.full((n_tp, n_hs), np.nan) if run_no_rao else None
    LCC_A_rao = np.full((n_tp, n_hs), np.nan) if run_rao    else None
    LCC_B_rao = np.full((n_tp, n_hs), np.nan) if run_rao    else None

    u_zero = np.zeros(dyn.N_dof)
    M_stat, T_stat = dyn._internal_forces(u_zero)
    if n_stinger > 0:
        M_stat[-n_stinger:] = 0.0
    M_stat_kNm = M_stat / 1e3
    T_stat_kN = T_stat / 1e3
    y_el = 0.5*(dyn.y0[:-1]+dyn.y0[1:])

    n_total = n_hs * n_tp
    n_done = 0
    t_start = time.time()
    _print_scatter_header(run_no_rao, run_rao)

    for j, Hs_s in enumerate(Hs_scatter):
        for i, Tp_s in enumerate(Tp_scatter):
            wave_s = WaveField(WD=WD)
            wave_s.irregular_wave(Hs=float(Hs_s), Tp=float(Tp_s), gamma=3.3, n_comp=n_comp, seed=seed)
            T_sim_s = max(300.0, 10.0 * Tp_s)

            if run_no_rao:
                r_nr = dyn.run(wave_s, dt=dt, T_sim=T_sim_s,
                               output_nodes=[dyn.tdp_idx, dyn.N_nodes-1],
                               print_interval=999.0)

                _apply_mask(r_nr, n_stinger)

                a_nr = compute_lcc(M_stat_kNm, T_stat_kN,
                                   r_nr['M_max']/1e3, r_nr['T_max']/1e3,
                                   y_el, 1.2, 0.7, cap)

                b_nr = compute_lcc(M_stat_kNm, T_stat_kN,
                                   r_nr['M_max']/1e3, r_nr['T_max']/1e3,
                                   y_el, 1.1, 1.3, cap)
                if n_stinger > 0:
                    a_nr[-n_stinger:]=0.0
                    b_nr[-n_stinger:]=0.0

                LCC_A_nrao[i,j]=np.max(a_nr)
                LCC_B_nrao[i,j]=np.max(b_nr)

            if run_rao:
                ld_s, lv_s, la_s = vessel.build_lop_functions(wave_s)
                r_r = run_with_rao(dyn, wave_s, ld_s, lv_s, la_s,
                                   dt=dt, T_sim=T_sim_s,
                                   n_stinger=n_stinger, print_interval=999.0)

                a_r = compute_lcc(M_stat_kNm, T_stat_kN,
                                  r_r['M_max']/1e3, r_r['T_max']/1e3,
                                  y_el, 1.2, 0.7, cap)

                b_r = compute_lcc(M_stat_kNm, T_stat_kN,
                                  r_r['M_max']/1e3, r_r['T_max']/1e3,
                                  y_el, 1.1, 1.3, cap)
                if n_stinger > 0:
                    a_r[-n_stinger:]=0.0
                    b_r[-n_stinger:]=0.0

                LCC_A_rao[i,j]=np.max(a_r)
                LCC_B_rao[i,j]=np.max(b_r)

            n_done += 1
            elapsed = time.time()-t_start
            eta = elapsed/n_done*(n_total-n_done)
            _print_scatter_row(Hs_s, Tp_s,
                               LCC_A_nrao[i,j] if run_no_rao else None,
                               LCC_B_nrao[i,j] if run_no_rao else None,
                               LCC_A_rao[i,j] if run_rao else None,
                               LCC_B_rao[i,j] if run_rao else None,
                               elapsed, eta)

    print(f"\n  Total: {time.time()-t_start:.0f}s for {n_total} runs\n")
    return LCC_A_nrao, LCC_B_nrao, LCC_A_rao, LCC_B_rao


def save_scatter(output_dir, Hs_scatter, Tp_scatter,
                 LCC_A_nrao, LCC_B_nrao, LCC_A_rao, LCC_B_rao):

    np.save(os.path.join(output_dir, 'Hs_scatter.npy'), Hs_scatter)
    np.save(os.path.join(output_dir, 'Tp_scatter.npy'), Tp_scatter)

    for arr, name in [(LCC_A_nrao,'LCC_A_norao'), (LCC_B_nrao,'LCC_B_norao'),
                      (LCC_A_rao, 'LCC_A_rao'), (LCC_B_rao, 'LCC_B_rao')]:
        if arr is not None:
            np.save(os.path.join(output_dir, f'{name}.npy'), arr)
    print(f"Scatter results saved to {output_dir}/")


# Plotting
def make_plots(dyn, res_no_rao, res_rao,
               lcc_a, lcc_b, gov_label,
               vessel, params, output_dir,
               dnv_scatter=False,
               Hs_scatter=None, Tp_scatter=None,
               LCC_A_nrao=None, LCC_B_nrao=None,
               LCC_A_rao=None, LCC_B_rao=None):

    params = dict(params)
    params['gov_label'] = gov_label

    print("\nSaving plots...")
    plot_summary(dyn, res_no_rao, res_rao, lcc_a, lcc_b,
                vessel.omega_rao, vessel.Xw_rao,
                params, out_path=os.path.join(output_dir, 'case_results.png'))

    plot_lcc_profile(dyn, lcc_a, lcc_b, params,
                     gov_label=gov_label,
                     out_path=os.path.join(output_dir, 'case_lcc.png'))

    if dnv_scatter and Hs_scatter is not None:
        compare = params.get('COMPARE_MODE', 'compare')

        if compare == 'compare':
            datasets = [(LCC_A_nrao, 'LC-A Waves only\nγ_F=1.2 γ_E=0.7'),
                        (LCC_B_nrao, 'LC-B Waves only\nγ_F=1.1 γ_E=1.3'),
                        (LCC_A_rao, 'LC-A Waves + RAO\nγ_F=1.2 γ_E=0.7'),
                        (LCC_B_rao, 'LC-B Waves + RAO\nγ_F=1.1 γ_E=1.3')]

        elif compare == 'waves_only':
            datasets = [(LCC_A_nrao,'LC-A Waves only\nγ_F=1.2 γ_E=0.7'),
                        (LCC_B_nrao,'LC-B Waves only\nγ_F=1.1 γ_E=1.3')]

        else:
            datasets = [(LCC_A_rao,'LC-A Waves + RAO\nγ_F=1.2 γ_E=0.7'),
                        (LCC_B_rao,'LC-B Waves + RAO\nγ_F=1.1 γ_E=1.3')]

        plot_lcc_scatter(datasets, Hs_scatter, Tp_scatter, params,
                         out_path=os.path.join(output_dir, 'case_scatter_lcc.png'))


def make_animation(dyn, wave, vessel, run_rao,
                   dt, T_sim, n_stinger,
                   fps, every, output_dir):

    out_path = os.path.join(output_dir, 'case_animation.gif')

    if run_rao:
        lop_disp, lop_vel, lop_acc = vessel.build_lop_functions(wave)
        run_animation_with_rao(dyn, wave, lop_disp, lop_vel, lop_acc,
                               dt=dt, T_sim=T_sim, n_stinger=n_stinger,
                               fps=fps, every=every, out_path=out_path)

    else:
        dyn.animate(wave, dt=dt, T_sim=T_sim, fps=fps, every=every, out_path=out_path)


# Summary print in console
def print_summary(dyn, res_no_rao, res_rao,
                  lcc_a, lcc_b, gov_label,
                  compare_mode, params):

    u_zero = np.zeros(dyn.N_dof)
    M_static, T_static = dyn._internal_forces(u_zero)
    x_el = dyn.x0[-1] - 0.5*(dyn.x0[:-1]+dyn.x0[1:])

    lay_type = params.get('lay_type','')
    PHI0_DEG = params.get('PHI0_DEG', 0)
    P0 = params.get('_P0', 0)
    WD = params.get('_WD', 0)
    Hs = params.get('Hs', 0)
    Tp = params.get('Tp', 0)
    WAVE_TYPE = params.get('WAVE_TYPE','')

    print(f"\n{'='*40}")
    print(f"RESULTS SUMMARY")
    print(f"{lay_type}, φ₀={PHI0_DEG:.0f}°, P₀={P0/1e3:.0f} kN, WD={WD:.0f}m")
    print(f"{WAVE_TYPE.upper()}, Hs={Hs}m, Tp={Tp}s")
    print(f"{'='*40}")

    if compare_mode == 'compare':
        print(f"{'':30s} {'Static':>10} {'Waves only':>10} {'Waves+RAO':>10}")
        print(f"{'-'*40}")

        print(f"{'Max |M| (kNm)':30s}  "
              f"{np.max(np.abs(M_static))/1e3:>10.1f}"
              f"{res_no_rao['M_max'].max()/1e3:>10.1f}"
              f"{res_rao['M_max'].max()/1e3:>10.1f}")

        print(f"{'Max T (kN)':30s}"
              f"{np.max(T_static)/1e3:>10.1f}"
              f"{res_no_rao['T_max'].max()/1e3:>10.1f}"
              f"{res_rao['T_max'].max()/1e3:>10.1f}")

    elif compare_mode == 'waves_only':
        print(f"{'':30s} {'Static':>10} {'Waves only':>10}")
        print(f"{'-'*40}")

        print(f"{'Max |M| (kN·m)':30s}"
              f"{np.max(np.abs(M_static))/1e3:>10.1f}"
              f"{res_no_rao['M_max'].max()/1e3:>10.1f}")

        print(f"{'Max T (kN)':30s}"
              f"{np.max(T_static)/1e3:>10.1f}"
              f"{res_no_rao['T_max'].max()/1e3:>10.1f}")

    else:   # rao_only
        print(f"{'':30s} {'Static':>10} {'Waves+RAO':>10}")
        print(f"{'-'*40}")

        print(f"{'Max |M| (kNm)':30s}"
              f"{np.max(np.abs(M_static))/1e3:>10.1f}"
              f"{res_rao['M_max'].max()/1e3:>10.1f}")

        print(f"{'Max T (kN)':30s}"
              f"{np.max(T_static)/1e3:>10.1f}"
              f"{res_rao['T_max'].max()/1e3:>10.1f}")

    print(f"\n DNV ST-F101 LCC ({gov_label}):")
    for lcc, label, YF, YE in [(lcc_a,'LC-A',1.2,0.7),(lcc_b,'LC-B',1.1,1.3)]:
        idx = np.argmax(lcc)
        status = 'PASS' if np.max(lcc) <= 1.0 else 'FAIL'
        print(f"{label} (γ_F={YF} γ_E={YE}):  "
              f"max = {np.max(lcc):.4f}  [{status}]  "
              f"at x={x_el[idx]:.0f}m from LOP")
    print(f"{'='*40}\n")


def _print_scatter_header(run_no_rao, run_rao):
    hdr = (f"{'Hs':>5} {'Tp':>5}"
           + (f"{'A_norao':>9} {'B_norao':>9} " if run_no_rao else "")
           + (f"{'A_rao':>9} {'B_rao':>9}" if run_rao else "")
           + f"{'elapsed':>7} {'ETA':>7}")
    print(hdr)
    print(f"{'-'*len(hdr)}")


def _print_scatter_row(Hs_s, Tp_s, a_nr, b_nr, a_r, b_r, elapsed, eta):
    row = f"{Hs_s:>5.1f}, {Tp_s:>5.1f}"
    if a_nr is not None:
        row += f"{a_nr:>9.3f}, {b_nr:>9.3f}"
    if a_r is not None:
        row += f"{a_r:>9.3f}, {b_r:>9.3f}"
    row += f"{elapsed:>6.0f}s, {eta:>6.0f}s"
    print(row)
