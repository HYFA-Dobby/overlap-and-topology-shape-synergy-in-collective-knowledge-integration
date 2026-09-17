import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from pathlib import Path
from matplotlib.lines import Line2D


"""
Visualization of Fig. 4B and Fig. 4C.

Fig. 4B:
    Number of structurally synergistic external nodes, N_syn.

Fig. 4C:
    Boundary concentration rho_syn.

Input:
    visualization_data/Fig.4/boundary/
        syn_result_ps{p_s}.csv

Output:
    figures/Fig.4/
"""


# ============================================================
# Paths
# ============================================================

data_dir = Path(
    "./visualization_data/Fig.4/boundary"
)

figure_dir = Path(
    "./figures/Fig.4"
)

figure_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Rewiring probabilities
# ============================================================

prob_list = np.logspace(
    -4,
    0,
    10
)

prob_list = np.concatenate(
    [
        [0],
        prob_list
    ]
)

positive_prob = prob_list[
    prob_list > 0
]


# ============================================================
# Color scale
# ============================================================

norm = mcolors.LogNorm(
    vmin=positive_prob.min(),
    vmax=positive_prob.max()
)

cmap = plt.colormaps[
    "viridis"
]


# ============================================================
# Utility
# ============================================================

def get_color(p_s):

    if np.isclose(
        p_s,
        0.0
    ):
        return "black"

    return cmap(
        norm(p_s)
    )


def add_rewiring_colorbar(
    fig,
    ax
):

    sm = plt.cm.ScalarMappable(
        norm=norm,
        cmap=cmap
    )

    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=ax,
        pad=0.02
    )

    cbar.set_label(
        r"Rewiring probability, $p$",
        fontsize=20
    )

    cbar.ax.tick_params(
        labelsize=15
    )

    # p = 0
    p0_handle = Line2D(
        [0],
        [0],
        color="black",
        linewidth=2.5,
        label=r"$p=0$"
    )

    ax.legend(
        handles=[p0_handle],
        loc="upper left",
        frameon=False,
        fontsize=16
    )


# ============================================================
# Fig. 4B
# Number of synergistic external nodes
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 6)
)


for p_s in prob_list:

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    path = (
        data_dir
        / f"syn_result_ps{p_s}.csv"
    )

    df = pd.read_csv(
        path
    )

    df = df.sort_values(
        "D_ov"
    )

    x = df[
        "D_ov"
    ]

    y = df[
        "syn_count_mean"
    ]

    sem = df[
        "syn_count_sem"
    ]

    color = get_color(
        p_s
    )

    # --------------------------------------------------------
    # Mean
    # --------------------------------------------------------

    ax.plot(
        x,
        y,
        color=color,
        linewidth=2.5
    )

    # --------------------------------------------------------
    # SEM
    # --------------------------------------------------------

    ax.fill_between(
        x,
        np.maximum(
            y - sem,
            0
        ),
        y + sem,
        color=color,
        alpha=0.15,
        linewidth=0
    )


# ------------------------------------------------------------
# Axis
# ------------------------------------------------------------

ax.set_xlim(
    0,
    1
)

ax.set_ylim(
    bottom=0
)

ax.set_xlabel(
    r"$D_{ov}$",
    fontsize=30
)

ax.set_ylabel(
    "Synergy nodes",
    fontsize=30
)

ax.tick_params(
    axis="both",
    labelsize=25
)

ax.locator_params(
    axis="x",
    nbins=3
)

ax.locator_params(
    axis="y",
    nbins=4
)


# ------------------------------------------------------------
# Colorbar
# ------------------------------------------------------------

add_rewiring_colorbar(
    fig,
    ax
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

plt.tight_layout()

output_path = (
    figure_dir
    / "Fig4B_boundary_count.pdf"
)

plt.savefig(
    output_path,
    dpi=100,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {output_path}"
)


# ============================================================
# Fig. 4C
# Boundary concentration
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 6)
)


for p_s in prob_list:

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    path = (
        data_dir
        / f"syn_result_ps{p_s}.csv"
    )

    df = pd.read_csv(
        path
    )

    df = df.sort_values(
        "D_ov"
    )

    x = df[
        "D_ov"
    ]

    y = df[
        "syn_ratio_mean"
    ]

    sem = df[
        "syn_ratio_sem"
    ]

    color = get_color(
        p_s
    )

    # --------------------------------------------------------
    # Mean
    # --------------------------------------------------------

    ax.plot(
        x,
        y,
        color=color,
        linewidth=2.5
    )

    # --------------------------------------------------------
    # SEM
    # --------------------------------------------------------

    ax.fill_between(
        x,
        np.maximum(
            y - sem,
            0
        ),
        y + sem,
        color=color,
        alpha=0.15,
        linewidth=0
    )


# ------------------------------------------------------------
# Axis
# ------------------------------------------------------------

ax.set_xlim(
    0,
    1
)

ax.set_ylim(
    bottom=0
)

ax.set_xlabel(
    r"$D_{ov}$",
    fontsize=30
)

ax.set_ylabel(
    "Synergy node ratio",
    fontsize=30
)

ax.tick_params(
    axis="both",
    labelsize=25
)

ax.locator_params(
    axis="x",
    nbins=3
)

ax.locator_params(
    axis="y",
    nbins=4
)


# ------------------------------------------------------------
# Colorbar
# ------------------------------------------------------------

add_rewiring_colorbar(
    fig,
    ax
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

plt.tight_layout()

output_path = (
    figure_dir
    / "Fig4C_boundary_ratio.pdf"
)

plt.savefig(
    output_path,
    dpi=100,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {output_path}"
)