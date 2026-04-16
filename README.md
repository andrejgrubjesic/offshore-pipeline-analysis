# Pipeline Installation Analysis

Dynamic analysis of offshore pipeline installation (S-lay and J-lay)
with vessel motion excitation and DNV ST-F101 code check.


<img width="1400" height="600" alt="case_animation" src="https://github.com/user-attachments/assets/3588f0d9-76b0-4851-98af-988619602572" />


## References
- **Static solver**: Trapper, P.A. (2019), *Feasible numerical method for analysis of offshore pipeline in installation, Applied Ocean Research* 88, 48–62
    - https://doi.org/10.1016/j.apor.2019.04.018
    
- **Dynamic solver**: Kim & Kim (2018), *An efficient linearised dynamic analysis method for structural safety design of J-lay and S-lay
                                         pipeline installation Ships and Offshore Structures*
    - https://doi.org/10.1080/17445302.2018.1493906
      
- **Code check**: DNV ST-F101 (2021), Submarine Pipeline Systems

- **Generalized-α Method**: Chung & Hulbert (1993), *A Time Integration Algorithm for Structural Dynamics With Improved Numerical Dissipation: The Generalized-α Method*
    - https://doi.org/10.1115/1.2900803

- **Adriatic spectrum**: Tabain, T. (1997), *Standard wind wave spectrum for the Adriatic Sea revisited (1977-1997). Brodogradnja* 45, 303-313


## Quick start

1. Copy your RAO data files (`A.txt`, `B.txt`, `C.txt`, `M.txt`, `omega.txt`, `Fw.txt`) into `data/`
2. Open `run_case.py` and edit the parameter blocks at the top
3. Run:

```bash
python run_case.py
```

## Module overview

| Module | Exports | Description |
|---|---|---|
| `static_solver` | `StaticSolver` | FDM energy minimisation catenary |
| `wave_field` | `WaveField` | Airy kinematics, JONSWAP, Tabain |
| `dynamic_solver` | `DynamicSolver` | FEM assembly + `run()` |
| `vessel_rao` | `VesselRAO` | Load RAO matrices, build LOP motion |
| `rao_dynamic_solver` | `run_with_rao()` | Newmark with prescribed LOP |
| `dnv_check` | `DNVCapacities`, `compute_lcc()` | ST-F101 LCC check |
| `visualisation` | `plot_results()`, `plot_lcc()`, etc. | All figures |


