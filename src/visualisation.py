"""
All plotting functions for pipeline installation analysis results.

Functions
---------
plot_summary()        — 9-panel results overview
plot_lcc_profile()    — DNV LCC along the pipe (single sea state)
plot_lcc_scatter()    — DNV LCC Hs/Tp heatmap (scatter diagram)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm


def plot_summary(dyn, res_no_rao, res_rao,
                 lcc_a, lcc_b,
                 omega_rao, Xw_rao,
                 params, out_path='case_results.png'):
    """
    9-panel summary figure.

    Parameters
    ----------
    dyn         : DynamicSolver
    res_no_rao  : dict or None   result from dyn.run()
    res_rao     : dict or None   result from run_with_rao()
    lcc_a, lcc_b: ndarray (N_el,)  LCC profiles
    omega_rao   : ndarray   RAO frequencies
    Xw_rao      : ndarray   complex RAO (6, N_omega)
    params      : dict      display parameters (PHI0_DEG, _P0, _WD, Hs, Tp,
                            WAVE_TYPE, COMPARE_MODE, USE_HEAVE, USE_SURGE,
                            USE_PITCH, lay_type, gov_label)
    out_path    : str
    """
    u_zero = np.zeros(dyn.N_dof)
    M_static, T_static = dyn._internal_forces(u_zero)
    x0 = dyn.x0
    y0 = dyn.y0
    xL = x0[-1] - x0
    x_el = x0[-1] - 0.5*(x0[:-1]+x0[1:])

    res_gov = res_rao if res_rao is not None else res_no_rao
    tdp_out_idx = res_gov['output_nodes'].index(dyn.tdp_idx)
    t_arr = res_gov['t']
    gov_label = params.get('gov_label','')

    fig = plt.figure(figsize=(18,12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

    # 1 — Static configuration
    ax1 = fig.add_subplot(gs[0,0])
    ax1.plot(xL, y0, 'b-', lw=2, label='Catenary')
    ax1.axhline(params['_WD'], color='saddlebrown', lw=2, label='Seabed')
    ax1.scatter([xL[dyn.tdp_idx]], [y0[dyn.tdp_idx]], c='r', zorder=5, label='TDP')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.4)
    ax1.legend(fontsize=8)
    ax1.set_xlabel('x from LOP (m)')
    ax1.set_ylabel('Depth (m)')
    ax1.set_title('Static configuration')

    # 2 — Max bending moment envelope
    ax2 = fig.add_subplot(gs[0,1])
    ax2.plot(x_el, np.abs(M_static)/1e3, 'b--', lw=1.5, label='Static')
    if res_no_rao is not None:
        ax2.plot(x_el, res_no_rao['M_max']/1e3, 'g-', lw=2, label='Waves only')
    if res_rao is not None:
        ax2.plot(x_el, res_rao['M_max']/1e3, 'r-', lw=2, label='Waves + RAO')
    ax2.invert_xaxis()
    ax2.grid(True, alpha=0.4)
    ax2.legend(fontsize=8)
    ax2.set_xlabel('x from LOP (m)')
    ax2.set_ylabel('|M| (kN·m)')
    ax2.set_title('Max bending moment envelope')

    # 3 — Max tension envelope
    ax3 = fig.add_subplot(gs[0,2])
    ax3.plot(x_el, T_static/1e3, 'b--', lw=1.5, label='Static')
    if res_no_rao is not None:
        ax3.plot(x_el, res_no_rao['T_max']/1e3, 'g-', lw=2, label='Waves only')
    if res_rao is not None:
        ax3.plot(x_el, res_rao['T_max']/1e3, 'r-', lw=2, label='Waves + RAO')
    ax3.invert_xaxis()
    ax3.grid(True, alpha=0.4)
    ax3.legend(fontsize=8)
    ax3.set_xlabel('x from LOP (m)')
    ax3.set_ylabel('T (kN)')
    ax3.set_title('Max tension envelope')

    # 4 — M time history at TDP
    ax4 = fig.add_subplot(gs[1,0])
    if res_no_rao is not None:
        ax4.plot(t_arr, res_no_rao['M_hist'][dyn.tdp_idx,:]/1e3,
                 'g-', lw=1, alpha=0.7, label='Waves only')
    if res_rao is not None:
        ax4.plot(t_arr, res_rao['M_hist'][dyn.tdp_idx,:]/1e3,
                 'r-', lw=1, alpha=0.8, label='Waves + RAO')
    ax4.axhline(M_static[dyn.tdp_idx]/1e3, color='k', ls='--', lw=1,
                label=f'Static {M_static[dyn.tdp_idx]/1e3:.1f} kNm')
    ax4.grid(True, alpha=0.4)
    ax4.legend(fontsize=8)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('M (kN·m)')
    ax4.set_title(f'M at TDP (node {dyn.tdp_idx})')

    # 5 — TDP vertical displacement
    ax5 = fig.add_subplot(gs[1,1])
    if res_no_rao is not None:
        ax5.plot(t_arr, res_no_rao['u_hist'][3*tdp_out_idx+1,:]*1000,
                 'g-', lw=1, alpha=0.7, label='Waves only')
    if res_rao is not None:
        ax5.plot(t_arr, res_rao['u_hist'][3*tdp_out_idx+1,:]*1000,
                 'r-', lw=1, alpha=0.8, label='Waves + RAO')
    ax5.axhline(0, color='k', lw=0.8, ls='--')
    ax5.grid(True, alpha=0.4)
    ax5.legend(fontsize=8)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('u_z (mm)')
    ax5.set_title('TDP vertical displacement')

    # 6 — Vessel LOP motion
    ax6 = fig.add_subplot(gs[1,2])
    if res_rao is not None:
        ax6.plot(t_arr, res_rao['lop_uz']*1000, 'b-', lw=1, label='Heave')
        ax6.plot(t_arr, res_rao['lop_ux']*1000, 'm-', lw=1, alpha=0.7, label='Surge')
        ax6.axhline(0, color='k', lw=0.8, ls='--')
        ax6.legend(fontsize=8)
    else:
        ax6.text(0.5,0.5,'RAO not run',ha='center',va='center',
                 transform=ax6.transAxes,fontsize=11,color='grey')
    ax6.grid(True, alpha=0.4)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Displacement (mm)')
    ax6.set_title('Vessel LOP motion (from RAO)')

    # 7 — RAO curves
    ax7 = fig.add_subplot(gs[2,0])
    Tp_plot = 2*np.pi / omega_rao
    ax7.plot(Tp_plot, np.abs(Xw_rao[2,:]), 'b-', lw=2, label='Heave (m/m)')
    ax7.plot(Tp_plot, np.abs(Xw_rao[0,:]), 'g-', lw=2, label='Surge (m/m)')
    ax7.axvline(params['Tp'], color='r', ls='--', lw=1.5,
                label=f"Tp={params['Tp']}s")
    ax7.set_xlim([2,30])
    ax7.grid(True, alpha=0.4)
    ax7.legend(fontsize=8)
    ax7.set_xlabel('Tp (s)')
    ax7.set_ylabel('RAO amplitude')
    ax7.set_title('Vessel RAO — Heave & Surge')

    # 8 — DNV LCC profile
    ax8 = fig.add_subplot(gs[2,1])
    ax8.plot(x_el, lcc_a, 'b-', lw=2, label=f'LC-A γ_F=1.2 γ_E=0.7')
    ax8.plot(x_el, lcc_b, 'r-', lw=2, label=f'LC-B γ_F=1.1 γ_E=1.3')
    ax8.axhline(1.0, color='k', ls='--', lw=1.5, label='Limit = 1.0')
    ax8.fill_between(x_el, 0, np.maximum(lcc_a,lcc_b),
                     where=np.maximum(lcc_a,lcc_b)>1.0,
                     alpha=0.25, color='red', label='Fail zone')
    ax8.invert_xaxis()
    ax8.grid(True, alpha=0.4)
    ax8.legend(fontsize=8)
    ax8.set_xlabel('x from LOP (m)')
    ax8.set_ylabel('LCC')
    ax8.set_title('DNV ST-F101 LCC')
    ax8.set_ylim(bottom=0)

    # 9 — DAF
    ax9 = fig.add_subplot(gs[2,2])
    M_abs = np.abs(M_static)
    M_abs[M_abs < 1.0] = 1.0
    if res_no_rao is not None:
        ax9.plot(x_el, res_no_rao['M_max']/M_abs, 'g-', lw=2, label='Waves only')
    if res_rao is not None:
        ax9.plot(x_el, res_rao['M_max']/M_abs, 'r-', lw=2, label='Waves + RAO')
    ax9.axhline(1.0, color='k', ls='--', lw=1)
    ax9.invert_xaxis()
    ax9.grid(True, alpha=0.4)
    ax9.legend(fontsize=8)
    ax9.set_xlabel('x from LOP (m)')
    ax9.set_ylabel('DAF')
    ax9.set_title('Dynamic amplification factor')

    fig.suptitle(
        f"{params['lay_type']}  φ₀={params['PHI0_DEG']:.0f}°  "
        f"P₀={params['_P0']/1e3:.0f} kN  WD={params['_WD']:.0f}m  |  "
        f"{params['WAVE_TYPE'].upper()}  Hs={params['Hs']}m  Tp={params['Tp']}s  |  "
        f"Mode: {params['COMPARE_MODE']}",
        fontsize=10)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_lcc_profile(dyn, lcc_a, lcc_b, params,
                     gov_label='', out_path='case_lcc.png'):
    """
    2-panel DNV LCC profile along the pipe (single sea state).

    Parameters
    ----------
    dyn          : DynamicSolver
    lcc_a, lcc_b : ndarray (N_el,)
    params       : dict  (PHI0_DEG, _P0, _WD, Hs, Tp, Safety_class, lay_type)
    gov_label    : str   label for governing result ('Waves only' or 'Waves+RAO')
    """
    x_el = dyn.x0[-1] - 0.5*(dyn.x0[:-1]+dyn.x0[1:])
    tdp_x = x_el[dyn.tdp_idx] if dyn.tdp_idx < len(x_el) else x_el[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"DNV ST-F101 LCC — {params['lay_type']}  "
        f"φ₀={params['PHI0_DEG']:.0f}°  P₀={params['_P0']/1e3:.0f} kN  "
        f"WD={params['_WD']:.0f}m  |  "
        f"Hs={params['Hs']}m  Tp={params['Tp']}s  |  "
        f"{params['Safety_class']}  |  {gov_label}",
        fontsize=11)

    for ax, lcc, title in [
            (axes[0], lcc_a,
             'Load Case A  (system effects present)\nγ_F = 1.2,  γ_E = 0.7'),
            (axes[1], lcc_b,
             'Load Case B  (system effects not present)\nγ_F = 1.1,  γ_E = 1.3')]:
        ax.plot(x_el, lcc, color='navy', lw=2, label='LCC')
        ax.axhline(1.0, color='red', lw=1.5, ls='--', label='Limit = 1.0')
        ax.fill_between(x_el, 0, lcc, where=lcc<=1.0,
                        alpha=0.2, color='green', label='Pass')
        ax.fill_between(x_el, 0, lcc, where=lcc>1.0,
                        alpha=0.3, color='red', label='Fail')
        idx_max = np.argmax(lcc)
        ax.plot(x_el[idx_max], lcc[idx_max], 'k*', ms=13, zorder=5,
                label=f'Max = {lcc[idx_max]:.3f}\nx = {x_el[idx_max]:.0f}m')
        ax.axvline(tdp_x, color='orange', lw=1.2, ls=':', alpha=0.8, label='TDP')
        ax.set_xlabel('x from LOP (m)', fontsize=11)
        ax.set_ylabel('LCC', fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.invert_xaxis()
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_lcc_scatter(datasets, Hs_scatter, Tp_scatter, params,
                     out_path='case_scatter_lcc.png'):
    """
    Heatmap of max LCC over the Hs/Tp scatter grid.

    Parameters
    ----------
    datasets : list of (LCC_array, title_string)
               e.g. [(LCC_A_norao, 'LC-A Waves only'), ...]
    Hs_scatter, Tp_scatter : 1D arrays
    params   : dict  (PHI0_DEG, _P0, _WD, Safety_class, COMPARE_MODE, lay_type)
    """
    Hs_grid, Tp_grid = np.meshgrid(Hs_scatter, Tp_scatter)
    n = len(datasets)
    if n == 4:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes_flat = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes_flat = axes.flatten()

    fig.suptitle(
        f"DNV ST-F101 LCC Scatter — {params['lay_type']}  "
        f"φ₀={params['PHI0_DEG']:.0f}°  P₀={params['_P0']/1e3:.0f} kN  "
        f"WD={params['_WD']:.0f}m  |  {params['Safety_class']}  |  "
        f"Mode: {params['COMPARE_MODE']}",
        fontsize=11)

    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=3.0)

    for ax, (LCC, title) in zip(axes_flat, datasets):
        cf = ax.contourf(Hs_grid, Tp_grid, np.clip(LCC, 0, 3),
                         levels=np.linspace(0, 3, 31),
                         cmap='RdYlGn_r', norm=norm)
        cb = plt.colorbar(cf, ax=ax, label='Max DNV LCC (≤1.0 = pass)')
        cb.ax.axhline(1.0, color='black', lw=2, ls='--')
        try:
            cs = ax.contour(Hs_grid, Tp_grid, LCC,
                            levels=[1.0], colors='red', linewidths=2.5)
            ax.clabel(cs, fmt='LCC=1.0', fontsize=9, colors='red')
        except Exception:
            pass
        max_lcc = float(np.nanmax(LCC))
        idx = np.unravel_index(np.nanargmax(LCC), LCC.shape)
        ax.plot(Hs_scatter[idx[1]], Tp_scatter[idx[0]],
                'k*', ms=14, zorder=5,
                label=f'Max={max_lcc:.3f}\n'
                      f'Hs={Hs_scatter[idx[1]]:.1f}m '
                      f'Tp={Tp_scatter[idx[0]]:.0f}s')
        ax.set_xlabel('Hs (m)', fontsize=11)
        ax.set_ylabel('Tp (s)', fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.25, color='white', lw=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
