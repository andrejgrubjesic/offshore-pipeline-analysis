"""
In the standard DynamicSolver, the LOP is fixed (uz = 0).
Here the LOP becomes a time-varying prescribed displacement computed from the vessel RAO:

    uz_LOP(t), ux_LOP(t), θ_LOP(t)  ← from VesselRAO.build_lop_functions()

This is the support excitation formulation. The equation of motion for the free DOFs becomes:

    [K_ff]{u_f} = {F_Morison} - [K_fp]{u_p(t)} - [M_fp]{ü_p(t)}

where fp = coupling submatrix between free and prescribed DOFs.

Also provides run_animation_with_rao() for generating GIF output.
"""

import numpy as np
import time
from scipy.linalg import lu_factor, lu_solve
from .dynamic_solver import DynamicSolver
from .wave_field import WaveField


def run_with_rao(dyn: DynamicSolver, wave: WaveField,
                 lop_displacement, lop_velocity, lop_acceleration,
                 dt, T_sim, n_stinger=0, print_interval=0.1):
    """
    Generalized-alpha time integration with prescribed LOP motion.

    The LOP DOFs (ux, uz, θ) are driven by the vessel RAO time series.
    The seabed FP node (ux) is fixed as usual.

    Parameters
    ----------
    dyn               : DynamicSolver  — built from static solution
    wave              : WaveField
    lop_displacement  : callable  f(t) → (ux, uz, theta)  from VesselRAO
    lop_velocity      : callable  f(t) → (vux, vuz, vtheta)
    lop_acceleration  : callable  f(t) → (aux, auz, atheta)
    dt                : float     time step [s]
    T_sim             : float     simulation time [s]
    n_stinger         : int       number of elements near LOP to mask (set M=0)
    print_interval    : float     fraction of T_sim between progress prints

    Returns
    -------
    dict with keys:
        t, u_hist, M_hist, T_hist
        M_max, M_rms, T_max, T_rms
        output_nodes, x_nodes, y_nodes
        lop_uz, lop_ux    — LOP displacement time histories [m]
    """
    n_steps = int(T_sim / dt)
    am = dyn.am
    af = dyn.af
    gm = dyn.gamma
    bt = dyn.beta

    # Prescribed DOFs: FP ux, LOP uz, LOP ux, LOP theta
    lop_node = dyn.N_nodes - 1
    dof_ux_lop = 3 * lop_node
    dof_uz_lop = 3 * lop_node + 1
    dof_theta_lop = 3 * lop_node + 2
    prescribed = sorted({0, dof_uz_lop, dof_ux_lop, dof_theta_lop})
    free = [d for d in range(dyn.N_dof) if d not in prescribed]

    # Sub-matrices coupling free and prescribed DOFs
    K_ff = dyn.K_g[np.ix_(free, free)]
    M_ff = dyn.M_g[np.ix_(free, free)]
    C_ff = dyn.C_g[np.ix_(free, free)]
    K_fp = dyn.K_g[np.ix_(free, prescribed)]
    M_fp = dyn.M_g[np.ix_(free, prescribed)]
    C_fp = dyn.C_g[np.ix_(free, prescribed)]

    K_eff = (1-am)*M_ff + (1-af)*gm*dt*C_ff + (1-af)*bt*dt**2*K_ff
    K_lu = lu_factor(K_eff)

    u = np.zeros(dyn.N_dof)
    v = np.zeros(dyn.N_dof)
    acc = np.zeros(dyn.N_dof)

    out_nodes = sorted({dyn.tdp_idx, dyn.N_nodes-1})
    t_arr = np.linspace(0, T_sim, n_steps+1)
    u_hist = np.zeros((len(out_nodes)*3, n_steps+1))
    M_hist = np.zeros((dyn.N_el, n_steps+1))
    T_hist = np.zeros((dyn.N_el, n_steps+1))
    lop_uz_hist = np.zeros(n_steps+1)
    lop_ux_hist = np.zeros(n_steps+1)

    M0, T0 = dyn._internal_forces(u)
    M_hist[:,0] = M0
    T_hist[:,0] = T0
    F_prev = dyn._morison_force(wave, 0., v, acc)
    print_step = max(1, int(print_interval * n_steps))
    t_wall = time.time()

    print(f"run_with_rao: dt={dt}s, steps={n_steps}, T={T_sim:.0f}s")
    print(f"{'Step':>6}, {'t(s)':>7}, {'uz_TDP(mm)':>12}, {'uz_LOP(mm)':>12}, {'elapsed':>8}")
    print(f"{'-'*40}")

    for step in range(n_steps):
        t_now  = step * dt
        t_next = (step+1) * dt

        ux_n, uz_n, th_n = lop_displacement(t_next)
        vux_n,vuz_n,vth_n = lop_velocity(t_next)
        aux_n,auz_n,ath_n = lop_acceleration(t_next)
        ux_c, uz_c, th_c = lop_displacement(t_now)
        vux_c,vuz_c,vth_c = lop_velocity(t_now)

        u_p_next = np.array([0., uz_n,  ux_n,  th_n ])
        u_p_now = np.array([0., uz_c,  ux_c,  th_c ])
        v_p_next = np.array([0., vuz_n, vux_n, vth_n])
        a_p_next = np.array([0., auz_n, aux_n, ath_n])

        for k_d, dof in enumerate(prescribed):
            u[dof] = u_p_now[k_d]

        F_next = dyn._morison_force(wave, t_next, v, acc)
        u_f=u[free]
        v_f=v[free]
        acc_f=acc[free]
        u_fp = u_f + dt*v_f + (0.5-bt)*dt**2*acc_f
        v_fp = v_f + (1-gm)*dt*acc_f

        R_eff = ((1-af)*F_next[free] + af*F_prev[free]
                 - (1-af)*K_ff@u_fp - af*K_ff@u_f
                 - (1-af)*C_ff@v_fp - af*C_ff@v_f
                 - am*M_ff@acc_f
                 - (1-af)*K_fp@u_p_next - af*K_fp@u_p_now
                 - (1-af)*C_fp@v_p_next
                 - (1-af)*M_fp@a_p_next)

        acc_fn = lu_solve(K_lu, R_eff)
        u[free] = u_fp + bt*dt**2*acc_fn
        v[free] = v_fp + gm*dt*acc_fn
        acc[free] = acc_fn

        for k_d, dof in enumerate(prescribed):
            u[dof] = u_p_next[k_d]

        F_prev = F_next

        M_e, T_e = dyn._internal_forces(u)
        M_hist[:,step+1] = M_e
        T_hist[:,step+1] = T_e
        lop_uz_hist[step+1] = uz_n
        lop_ux_hist[step+1] = ux_n
        for k, ni in enumerate(out_nodes):
            u_hist[3*k:3*k+3, step+1] = u[3*ni:3*ni+3]

        if (step+1) % print_step == 0:
            print(f"{step+1:>6}  {t_next:>7.2f}"
                  f"{u[3*dyn.tdp_idx+1]*1000:>12.2f}"
                  f"{uz_n*1000:>12.2f}"
                  f"{time.time()-t_wall:>7.1f}s")

    print(f"{'-'*40}")
    print(f"Completed in {time.time()-t_wall:.1f}s\n")

    # TODO: find a solution to fix artifacts near LOP; this will suffice for now
    if n_stinger > 0:
        M_hist[-n_stinger:, :] = 0.0

    skip = max(1, int(0.2*n_steps))
    M_w = M_hist[:,skip:]
    T_w = T_hist[:,skip:]

    M_max = np.max(np.abs(M_w), axis=1)
    T_max = np.max(T_w, axis=1)

    M_rms = np.sqrt(np.mean((M_w-np.mean(M_w,axis=1,keepdims=True))**2, axis=1))
    T_rms = np.sqrt(np.mean((T_w-np.mean(T_w,axis=1,keepdims=True))**2, axis=1))

    return dict(t=t_arr, u_hist=u_hist, M_hist=M_hist, T_hist=T_hist,
                M_max=M_max, M_rms=M_rms, T_max=T_max, T_rms=T_rms,
                output_nodes=out_nodes,
                x_nodes=dyn.x0, y_nodes=dyn.y0,
                lop_uz=lop_uz_hist, lop_ux=lop_ux_hist)

# Animation build
def run_animation_with_rao(dyn: DynamicSolver, wave: WaveField,
                           lop_displacement, lop_velocity, lop_acceleration,
                           dt, T_sim, n_stinger=0,
                           fps=20, every=2, out_path='animation.gif'):

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    n_steps = int(T_sim / dt)
    am = dyn.am
    af = dyn.af
    gm = dyn.gamma
    bt = dyn.beta

    lop_node = dyn.N_nodes - 1
    prescribed = sorted({0, 3*lop_node+1, 3*lop_node, 3*lop_node+2})
    free = [d for d in range(dyn.N_dof) if d not in prescribed]

    K_ff = dyn.K_g[np.ix_(free,free)]
    M_ff = dyn.M_g[np.ix_(free,free)]
    C_ff = dyn.C_g[np.ix_(free,free)]
    K_fp = dyn.K_g[np.ix_(free,prescribed)]
    M_fp = dyn.M_g[np.ix_(free,prescribed)]
    C_fp = dyn.C_g[np.ix_(free,prescribed)]

    K_eff = (1-am)*M_ff+(1-af)*gm*dt*C_ff+(1-af)*bt*dt**2*K_ff
    K_lu = lu_factor(K_eff)

    u = np.zeros(dyn.N_dof)
    v = np.zeros(dyn.N_dof)
    acc = np.zeros(dyn.N_dof)
    frame_idx = list(range(0, n_steps+1, every))
    n_frames = len(frame_idx)
    x_frames = np.zeros((n_frames, dyn.N_nodes))
    y_frames = np.zeros((n_frames, dyn.N_nodes))
    t_frames = np.zeros(n_frames)

    x_frames[0] = dyn.x0
    y_frames[0] = dyn.y0

    F_prev = dyn._morison_force(wave, 0., v, acc)
    fi = 1
    t_wall = time.time()
    print(f"Animating {n_steps} steps → {n_frames} frames ...")

    for step in range(n_steps):
        t_now = step*dt
        t_next = (step+1)*dt

        ux_n,uz_n,th_n = lop_displacement(t_next)

        vux_n,vuz_n,vth_n = lop_velocity(t_next)

        aux_n,auz_n,ath_n = lop_acceleration(t_next)

        ux_c,uz_c,th_c = lop_displacement(t_now)

        vux_c,vuz_c,vth_c = lop_velocity(t_now)

        u_p_next = np.array([0.,uz_n,ux_n,th_n])

        u_p_now = np.array([0.,uz_c,ux_c,th_c])

        v_p_next = np.array([0.,vuz_n,vux_n,vth_n])

        a_p_next = np.array([0.,auz_n,aux_n,ath_n])

        for k_d,dof in enumerate(prescribed):
            u[dof] = u_p_now[k_d]

        F_next = dyn._morison_force(wave,t_next,v,acc)
        u_f = u[free]
        v_f = v[free]
        acc_f = acc[free]

        u_fp = u_f+dt*v_f+(0.5-bt)*dt**2*acc_f
        v_fp = v_f+(1-gm)*dt*acc_f

        R_eff=((1-af)*F_next[free]+af*F_prev[free]
               -(1-af)*K_ff@u_fp-af*K_ff@u_f-(1-af)*C_ff@v_fp-af*C_ff@v_f
               -am*M_ff@acc_f-(1-af)*K_fp@u_p_next-af*K_fp@u_p_now
               -(1-af)*C_fp@v_p_next-(1-af)*M_fp@a_p_next)

        acc_fn = lu_solve(K_lu,R_eff)

        u[free] = u_fp+bt*dt**2*acc_fn
        v[free] = v_fp+gm*dt*acc_fn
        acc[free] = acc_fn

        for k_d,dof in enumerate(prescribed):
            u[dof] = u_p_next[k_d]

        F_prev = F_next
        if fi < n_frames and (step+1) == frame_idx[fi]:
            x_frames[fi] = dyn.x0+u[0::3]
            y_frames[fi] = dyn.y0+u[1::3]
            t_frames[fi] = t_next
            fi+=1

    print(f"Integration done in {time.time()-t_wall:.1f}s, building animation ...")

    WD = dyn.st.WD
    x0 = dyn.x0
    y0 = dyn.y0
    xL0 = x0[-1]-x0

    xmin = np.min(x0[-1]-x_frames)-20
    xmax = np.max(x0[-1]-x_frames)+20

    x_wg = np.linspace(x0[0]-50, x0[-1]+50, 400)
    x_wlop = x0[-1]-x_wg

    fig, axes = plt.subplots(1,1,figsize=(14,6))
    ax1 = axes

    ax1.set_xlim(xmin,xmax)
    ax1.set_ylim(WD+20,-10)
    ax1.axhline(WD,color='saddlebrown',lw=2,label='Seabed')
    ax1.axhline(0,color='steelblue',lw=0.8,ls='--',alpha=0.4)

    eta0=wave.surface_elevation(x_wg,0.)

    ax1.fill_between(x_wlop, WD, eta0, color='steelblue',alpha=0.35,zorder=1)
    wave_line,=ax1.plot(x_wlop, eta0, color='deepskyblue',lw=1.5,label='Wave')
    ax1.plot(xL0,y0,color='blue',lw=2,ls='--',alpha=0.4,label='Static')

    dyn_line,=ax1.plot([],[],color='red',lw=1.5,zorder=5,label='Dynamic')
    tdp_dot,=ax1.plot([],[],'ko',ms=6,zorder=6,label='TDP')
    lop_dot,=ax1.plot([],'go',ms=6,zorder=6,label='Vessel LOP')
    ax1.set_xlabel('x from LOP (m)')
    ax1.set_ylabel('Depth (m)')

    ax1.set_title('Dynamic model')

    ax1.legend(fontsize = 8, loc="center", bbox_to_anchor=(0.95, 0.2))

    time_txt=ax1.text(-0.13,0.97,'',transform=ax1.transAxes,fontsize=10,va='top')
    lop_txt =ax1.text(-0.13,0.91,'',transform=ax1.transAxes,fontsize=9,va='top')

    u0v = np.zeros(dyn.N_dof)
    M0, _ = dyn._internal_forces(u0v)
    if n_stinger>0:
        M0[-n_stinger:] = 0.0

    x_el0=x0[-1]-0.5*(x0[:-1]+x0[1:])

    M_frames=np.zeros((n_frames,dyn.N_el))
    M_frames[0]=M0
    u_full=np.zeros(dyn.N_dof)

    for f2 in range(1,n_frames):
        u_full[0::3]=x_frames[f2]-dyn.x0
        u_full[1::3]=y_frames[f2]-dyn.y0
        Mf,_=dyn._internal_forces(u_full)
        if n_stinger>0: Mf[-n_stinger:]=0.0
        M_frames[f2]=Mf

    def update(f2):
        eta=wave.surface_elevation(x_wg,t_frames[f2])
        wave_line.set_ydata(eta)

        for coll in ax1.collections:
            coll.remove()

        ax1.fill_between(x_wlop,WD,eta,color='steelblue',alpha=0.35,zorder=1)
        xL_dyn=x_frames[f2,-1]-x_frames[f2]

        dyn_line.set_data(xL_dyn,y_frames[f2])

        tdp_dot.set_data([xL_dyn[dyn.tdp_idx]],[y_frames[f2,dyn.tdp_idx]])
        lop_dot.set_data([0.0],[y_frames[f2,-1]])

        time_txt.set_text(f't = {t_frames[f2]:.1f} s')
        lop_txt.set_text(f'LOP Δuz = {(y_frames[f2,-1]-dyn.y0[-1])*1000:+.0f} mm')

        x_el_dyn=x_frames[f2,-1]-0.5*(x_frames[f2,:-1]+x_frames[f2,1:])

        return wave_line,dyn_line,tdp_dot,lop_dot,time_txt,lop_txt

    anim=FuncAnimation(fig,update,frames=n_frames,interval=1000//fps,blit=True)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    print(f"Animation saved: {out_path} ({n_frames} frames @ {fps} fps)")
