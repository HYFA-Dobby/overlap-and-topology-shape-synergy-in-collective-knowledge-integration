# overlap-and-topology-shape-synergy-in-collective-knowledge-integration
This repository contains the simulation code, curated visualization data, and figure-reproduction scripts used for the study **“Overlap and Topology Shape Synergy in Collective Knowledge Integration.”**

The repository separates three roles:

- `simulation/` generates raw numerical results from network simulations.
- `visualization_data/` contains the curated CSV files that are actually read by the plotting scripts.
- `visualization/` reproduces the main and supplementary figures from those curated CSV files.

---

## Repository structure

```text
.
├── src/
│   ├── __init__.py
│   ├── subnetwork.py
│   └── integnetwork.py
│
├── simulation/
│   ├── Fig2ab_2agent_internal_synergy_classification.py
│   ├── Fig2c_Nagent_internal_synergy_classification.py
│   ├── Fig3_Nagent_external_synergy_classification.py
│   ├── Fig4a_rewiring_synergy_cal.py
│   └── Fig4bc_rewiring_synergy_boundary_cal.py
│
└── visualization/
    ├── Fig2/Fig2AB.py, Fig2C.py
    ├── Fig3/Fig3.py
    ├── Fig4/Fig4A.py, Fig4BC.py
    ├── S1-S3/FigS1_S3.py
    ├── S4-S5/FigS4_S5.py
    ├── S6/FigS6.py
    ├── S7/FigS7.py
    ├── S8/FigS8.py
    ├── S9/FigS9.py
    ├── S10/FigS10.py
    └── S11/FigS11.py

```

If the data are distributed as `visualization_data.zip`, extract the archive in the repository root before running the visualization scripts, so that paths such as `./visualization_data/Fig.2/...` exist.

---

## Requirements

The scripts use the following Python packages:

```bash
pip install numpy pandas networkx matplotlib seaborn scipy
```

Several simulation scripts use Python multiprocessing and can be computationally expensive because the background networks contain `N = 100,000` nodes.

Run scripts from the repository root. For scripts importing `src`, the safest form is:

```bash
PYTHONPATH=. python simulation/Fig2ab_2agent_internal_synergy_classification.py
```

and similarly for `visualization/S10/FigS10.py`, which also imports `src`.

---

## Network and integration conventions

The background networks used in the simulation scripts are:

| Code name | Model | Main parameters |
|---|---|---|
| `random` | Erdős–Rényi (ER) | `N=10^5`, `p=20/(N-1)` |
| `scale_free` | Barabási–Albert (BA) | `N=10^5`, `m=10` |
| `modular` | 10-block SBM | block size `10^4`, `p_in=0.00182`, `p_out=0.000022` |
| `small_world` | Watts–Strogatz (WS) | `N=10^5`, degree `20`, default rewiring `p=1e-4` unless varied |

Network labels used in filenames are:

```text
random      -> ER
scale_free  -> BA
modular     -> SBM
small_world -> WS
```

`src/subnetwork.py` generates partially overlapping individual subnetworks. It first creates the prescribed shared node region and then grows mutually exclusive individual-specific regions. Additional background edges inside each individual subnetwork are retained according to the within-individual wiring probability `p_w`.

`src/integnetwork.py` constructs the integrated network as the **union of the nodes and edges already present in the individual subnetworks**. It does not add new integration edges.

---

## Internal and external synergy outputs

### Internal synergy

Internal synergy is based on shortest-path changes after integration. For two individuals, the simulation separately records:

- synergistic node pairs,
- pairs unique to individual A,
- pairs unique to individual B,
- redundant pairs.

For multiple individuals, the supplied simulation records the strict internal-synergy count.

### External synergy

External synergy is measured with first-exit random walks and/or by direct enumeration of the external boundary.

For `n` individuals, a discovered external node is represented by a binary pattern

```text
[U1, U2, ..., Un, R]
```

where the first `n` bits indicate whether the external node is connected to each individual-specific region and the final bit indicates a connection to the shared region `R`.

For two individuals:

```text
100 : U_A only
010 : U_B only
001 : R only
101 : U_A + R
011 : U_B + R
110 : U_A + U_B
111 : strict external synergy (U_A + U_B + R)
```

For `n` individuals, strict external synergy is the all-ones pattern of length `n+1`.

---

# Simulation scripts

## 1. `simulation/Fig2ab_2agent_internal_synergy_classification.py`

This script performs the two-individual internal-synergy calculation over a grid of overlap `D_ov` and within-individual wiring probability `p_w`.

It generates eight matrices for the selected background network:

```text
{label}_synergy_mean.csv
{label}_synergy_sem.csv
{label}_unique_A_mean.csv
{label}_unique_A_sem.csv
{label}_unique_B_mean.csv
{label}_unique_B_sem.csv
{label}_redundancy_mean.csv
{label}_redundancy_sem.csv
```

Raw output directory:

```text
results/internal_synergy/two_agent/
```

These outputs feed several figures, not only Fig. 2:

```text
synergy_mean      -> Fig. 2A/B
synergy_sem       -> Figs. S1-S3
unique_A_*        -> Figs. S4-S5
redundancy_*      -> Figs. S4-S5
unique_B_*        -> not required by the current visualization scripts
```

The distributed visualization files are therefore split across figure-specific folders rather than duplicated in one place.

---

## 2. `simulation/Fig2c_Nagent_internal_synergy_classification.py`

This script generalizes internal synergy to `N_agents` individuals and has two modes.

### `single_pw`

At fixed `p_w=0.5`, it saves the mean and SEM of internal synergy across overlap:

```text
{label}_synergy_mean_{N_agents}agent_pw_0.500.csv
{label}_synergy_sem_{N_agents}agent_pw_0.500.csv
```

Raw output directory:

```text
results/internal_synergy/multi_agent/{network_type}/
```

These data are used by **Fig. 2C**. The distributed `visualization_data/Fig.2/C/` contains files for group sizes `n=2,...,10` for each background network.

### `matrix`

This mode computes a `D_ov x p_w` matrix for a specified group size:

```text
{label}_synergy_matrix_mean_{N_agents}agent.csv
{label}_synergy_matrix_sem_{N_agents}agent.csv
```

The 5-agent matrices are used by **Fig. S6**.

---

## 3. `simulation/Fig3_Nagent_external_synergy_classification.py`

This script performs first-exit random-walk exploration and classifies the discovered external node by its connection pattern `[U1,...,Un,R]`.

Each repeat uses one independently generated background network, and multiple first-exit discoveries are performed on that background realization. Means and SEMs are then computed across repeats.

### `single_pw`

At fixed `p_w=0.5`, it generates:

```text
{label}_syn_result_{N_agents}agent.csv
```

with columns:

```text
ov, pattern, mean, sem
```

Raw output directory:

```text
results/external_synergy/{network_type}/{N_agents}agent/
```

These same pattern files are reused by several figures:

- **Fig. 3A:** 2-agent patterns for all four background networks.
- **Fig. 3B:** group-size dependence. The distributed data contain `n=2,...,5` for ER/BA/SBM and `n=2,...,10` for WS.
- **Fig. S7:** all seven 2-agent connection patterns. No separate S7 data directory is needed.
- **Fig. S9:** WS group-size data for `n=2,...,10`. No separate S9 data directory is needed.

### `matrix`

This mode computes strict external synergy over `D_ov x p_w` and produces mean/SEM heatmaps used by **Fig. S8**.

The visualization-ready names are:

```text
{label}_mean_2agent_heatmap.csv
{label}_sem_2agent_heatmap.csv
```


---

## 4. `simulation/Fig4a_rewiring_synergy_cal.py`

This script repeats the first-exit random-walk calculation while varying the WS rewiring probability

```text
p = 0 plus 10 logarithmically spaced values from 1e-4 to 1.
```

For each `p`, it saves a 2-agent connection-pattern summary with columns

```text
ov, pattern, mean, sem
```


---

## 5. `simulation/Fig4bc_rewiring_synergy_boundary_cal.py`

This script directly enumerates external boundary nodes around the integrated network for each WS rewiring probability and overlap condition.

For each realization it records:

- total number of external boundary nodes,
- number of structurally synergistic boundary nodes,
- fraction of synergistic nodes among the boundary.

Each CSV contains:

```text
p_w
p_s
ov
D_ov
outside_mean
outside_sem
syn_count_mean
syn_count_sem
syn_ratio_mean
syn_ratio_sem
```

These files are used by:

- **Fig. 4B:** `syn_count_mean` and `syn_count_sem`.
- **Fig. 4C:** `syn_ratio_mean` and `syn_ratio_sem`.
- **Fig. S11:** `D_ov` and `syn_ratio_mean` are reused for the phenomenological effective-contact fit.

Thus there is no second copy of the Fig. 4C data under `visualization_data/Fig.S11/`.

---

# Visualization data: why some files are present and others are absent

`visualization_data/` is organized around the **inputs actually consumed by each visualization script**, rather than around a one-to-one copy of `results/`.

The dependency structure is:

```text
Fig2ab simulation
├── synergy_mean ----------------------> visualization_data/Fig.2/AB/ ------> Fig. 2A/B
├── synergy_sem -----------------------> visualization_data/Fig.S1-S3/ ----> Figs. S1-S3
├── unique_A_mean / redundancy_mean ---> visualization_data/Fig.S4-S5/ ----> Figs. S4-S5
├── unique_A_sem / redundancy_sem -----> archived with S4-S5 for completeness
└── unique_B_* ------------------------> not copied; not used by current plots

Fig2c simulation
├── fixed-p_w mean/SEM ----------------> visualization_data/Fig.2/C/ -------> Fig. 2C
└── 5-agent matrices ------------------> visualization_data/Fig.S6/ --------> Fig. S6

Fig3 simulation
├── single-p_w pattern CSVs -----------> visualization_data/Fig.3/ ---------> Fig. 3
│                                                               ├----------> Fig. S7
│                                                               └----------> Fig. S9 (WS only)
└── matrix-mode strict heatmaps --------> visualization_data/Fig.S8/ -------> Fig. S8

Fig4A simulation -----------------------> visualization_data/Fig.4/random_walk/ -> Fig. 4A

Fig4BC simulation ----------------------> visualization_data/Fig.4/boundary/ ---> Fig. 4B/C
                                                                            └-> Fig. S11

FigS10.py ------------------------------> candidate/selected realization CSVs + Fig. S10

FigS11.py + Fig.4/boundary data --------> fitted_effective_contact_parameters.csv + Fig. S11
```

This explains the intentionally missing figure-specific data folders:

- There is **no `visualization_data/Fig.S7/`** because Fig. S7 reuses the 2-agent files in `visualization_data/Fig.3/`.
- There is **no `visualization_data/Fig.S9/`** because Fig. S9 reuses the WS `n=2,...,10` files in `visualization_data/Fig.3/small_world/`.
- Figs. S1-S3 do not contain their own mean matrices; they reuse the Fig. 2 synergy means and store only the additional SEM matrices under `Fig.S1-S3/`.
- Fig. S11 does not duplicate the Fig. 4C boundary curves; it reads them directly from `visualization_data/Fig.4/boundary/`.

---

# Figure-by-figure reproduction map

| Figure | Visualization script | Required visualization data | Source calculation |
|---|---|---|---|
| Fig. 2A/B | `visualization/Fig2/Fig2AB.py` | `Fig.2/AB/{network}/{label}_synergy_mean.csv` | `Fig2ab_2agent_internal_synergy_classification.py` |
| Fig. 2C | `visualization/Fig2/Fig2C.py` | `Fig.2/C/{network}/{label}_synergy_{mean,sem}_{n}agent_pw_0.500.csv` | `Fig2c_Nagent_internal_synergy_classification.py`, `single_pw` |
| Fig. 3A/B | `visualization/Fig3/Fig3.py` | `Fig.3/{network}/{label}_syn_result_{n}agent.csv` | `Fig3_Nagent_external_synergy_classification.py`, `single_pw` |
| Fig. 4A | `visualization/Fig4/Fig4a.py` | `Fig.4/random_walk/syn_result_2agent_ps{p}.csv` | `Fig4a_rewiring_synergy_cal.py` |
| Fig. 4B/C | `visualization/Fig4/Fig4bc.py` | `Fig.4/boundary/syn_result_ps{p}_boundary.csv` | `Fig4bc_rewiring_synergy_boundary_cal.py` |
| Fig. S1 | `visualization/S1-S3/FigS1_S3.py` | S1-S3 synergy SEM | Fig2ab simulation |
| Fig. S2 | same | Fig. 2 synergy mean + S1-S3 synergy SEM | Fig2ab simulation |
| Fig. S3 | same | Fig. 2 synergy mean + S1-S3 synergy SEM | Fig2ab simulation |
| Fig. S4/S5 | `visualization/S4-S5/FigS4_S5.py` | unique-A/redundancy mean matrices | Fig2ab simulation |
| Fig. S6 | `visualization/S6/FigS6.py` | 5-agent synergy matrix mean | Fig2c simulation, `matrix` |
| Fig. S7 | `visualization/S7/FigS7.py` | **reuses Fig. 3 2-agent CSVs** | Fig3 simulation, `single_pw` |
| Fig. S8 | `visualization/S8/FigS8.py` | strict external-synergy mean/SEM heatmaps | Fig3 simulation, `matrix` |
| Fig. S9 | `visualization/S9/FigS9.py` | **reuses Fig. 3 WS group-size CSVs** | Fig3 simulation, `single_pw` |
| Fig. S10 | `visualization/S10/FigS10.py` | no aggregate input required; script generates realizations itself | S10 script itself |
| Fig. S11 | `visualization/S11/FigS11.py` | **reuses Fig. 4 boundary CSVs** | Fig4bc data + phenomenological fitting |

Fig. 1 is a conceptual schematic and is not generated by the simulation/visualization scripts in this archive.

The repository also does not contain a separate Fig. 4D plotting script. The representative network snapshots are generated by the Fig. S10 workflow; the intermediate-overlap condition is the source of the corresponding representative visualization.

---

# CSV formats

## Internal-synergy matrices

Used for Fig. 2A/B and several supplementary heatmaps.

```text
rows    : overlap conditions D_ov
columns : p_w values
cells   : mean or SEM of classified node-pair counts
```

The curated matrix files used by `Fig2AB.py`, `FigS1_S3.py`, `FigS4_S5.py`, and `FigS6.py` do **not** contain an additional row-index column; the plotting scripts reconstruct the coordinate grids from their configured `p_w` and `D_ov` arrays.

## External pattern summaries

Used for Figs. 3, 4A, S7, and S9.

```text
ov,pattern,mean,sem
```

Because CSV readers may interpret patterns such as `001` as the integer `1`, the visualization scripts restore leading zeros to the expected pattern length before classification.

## External strict-synergy heatmaps

Used for Fig. S8. These CSVs retain `D_ov` as the row index and are read with `index_col=0`.

## Boundary statistics

Used for Fig. 4B/C and Fig. S11:

```text
p_w,p_s,ov,D_ov,
outside_mean,outside_sem,
syn_count_mean,syn_count_sem,
syn_ratio_mean,syn_ratio_sem
```

---

# Reproducing figures from the included visualization data

Once `visualization_data/` is available, the figures can be generated without rerunning the expensive simulations.

```bash
python visualization/Fig2/Fig2AB.py
python visualization/Fig2/Fig2C.py
python visualization/Fig3/Fig3.py
python visualization/Fig4/Fig4a.py
python visualization/Fig4/Fig4bc.py

python visualization/S1-S3/FigS1_S3.py
python visualization/S4-S5/FigS4_S5.py
python visualization/S6/FigS6.py
python visualization/S7/FigS7.py
python visualization/S8/FigS8.py
python visualization/S9/FigS9.py
PYTHONPATH=. python visualization/S10/FigS10.py
python visualization/S11/FigS11.py
```

The scripts create their corresponding subdirectories under `figures/` automatically.

Note that Fig. S10 regenerates large WS networks and is therefore substantially more expensive than the other visualization scripts. Fig. S11 performs fitting but does not rerun the network simulation.

---

# Regenerating the simulation data

The simulation scripts expose `network_type`, `simulation_mode`, `N_agents`, grid sizes, and iteration counts near the top of their main functions. To reconstruct all distributed data, the scripts must be run repeatedly for the required networks and group sizes.

For example:

- run the two-agent internal simulation once for each of `random`, `scale_free`, `modular`, and `small_world`;
- run the multi-agent internal `single_pw` mode for the required group sizes;
- run the multi-agent internal `matrix` mode with `N_agents=5` for Fig. S6;
- run the external `single_pw` mode for `n=2,...,5` in ER/BA/SBM and `n=2,...,10` in WS;
- run the external `matrix` mode with `N_agents=2` for Fig. S8;
- run both WS rewiring simulations for Fig. 4.

Raw simulation outputs are written to `results/`. Before replacing files in `visualization_data/`, check the filename and CSV-layout notes above. The distributed plotting data should be treated as the canonical examples of the formats expected by the visualization scripts.

---

# Important compatibility notes

1. **Some data are shared among figures rather than duplicated.**  
   In particular, S1-S3 reuse Fig. 2 mean data, S7 and S9 reuse Fig. 3 pattern data, and S11 reuses Fig. 4C boundary data.


2. **Run scripts from the repository root.**  
   Relative paths in the visualization scripts assume the repository root as the working directory.

---

## Citation
If you use this code or data, please cite the corresponding preprint:

> Daeseong Kim and Masato S. Abe, **“Overlap and Topology Shape Synergy in Collective Knowledge Integration.”** *arXiv preprint* arXiv:2609.18140 (2026).  
> [https://arxiv.org/abs/2609.18140](https://arxiv.org/abs/2609.18140)

BibTeX:

```bibtex
@misc{kim2026overlap,
  title         = {Overlap and Topology Shape Synergy in Collective Knowledge Integration},
  author        = {Kim, Daeseong and Abe, Masato S.},
  year          = {2026},
  eprint        = {2609.18140},
  archivePrefix = {arXiv},
  primaryClass  = {physics.soc-ph},
  url           = {https://arxiv.org/abs/2609.18140}
}
