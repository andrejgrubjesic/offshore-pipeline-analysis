# Pipeline Installation Analysis

Dynamic analysis of offshore pipeline installation (S-lay and J-lay)
with vessel motion excitation and DNV ST-F101 code check.

## References
- **Static solver**: Trapper, P.A. (2019), *Applied Ocean Research* 88, 48–62
- **Dynamic solver**: Kim & Kim (2018), *Ships and Offshore Structures*
- **Code check**: DNV ST-F101 (2021), Submarine Pipeline Systems
- **Adriatic spectrum**: Tabain (1997)

---

## Repository structure

```
pipeline/
├── static_solver.py       # Trapper (2019) FDM catenary equilibrium
├── wave_field.py          # Airy wave kinematics (regular / JONSWAP / Tabain)
├── dynamic_solver.py      # Kim & Kim (2018) linearised FEM + Newmark
├── vessel_rao.py          # 6-DOF vessel RAO + LOP motion functions
├── rao_dynamic_solver.py  # Generalised-alpha integrator with RAO support excitation
├── dnv_check.py           # DNV ST-F101 LCC combined loading criterion
├── visualisation.py       # All plots and animation
run_case.py                # ← EDIT THIS to run your analysis
data/                      # Put A.txt B.txt C.txt M.txt omega.txt Fw.txt here
results/                   # Output figures saved here
```

---

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

## Dependencies

```
numpy
scipy
matplotlib
```

Install with:
```bash
pip install numpy scipy matplotlib
```
