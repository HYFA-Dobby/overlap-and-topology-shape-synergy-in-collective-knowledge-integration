import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.ticker import MaxNLocator


"""
Visualization of Figs. S1-S3.

Fig. S1:
    Absolute SEM of internal synergy.

Fig. S2:
    Relative SEM = SEM / mean.
    Cells whose mean is below 1% of the maximum mean for the
    corresponding network are omitted.

Fig. S3:
    Cross-sections of mean internal synergy at representative
    p_w values, with ±1 SEM shown as shaded regions.

Input:
    visualization_data/Fig.S1-S3/{network}/{label}_synergy_sem.csv

Mean data are reused from Fig. 2:
    visualization_data/Fig.2/AB/{network}/{label}_synergy_mean.csv

Output:
    figures/Fig.S1-S3/
"""


# ============================================================
# Paths
# ============================================================

sem_data_dir = Path(
    "./visualization_data/Fig.S1-S3"
)

mean_data_dir = Path(
    "./visualization_data/Fig.2/AB"
)

figure_dir = Path(
    "./figures/Fig.S1-S3"
)

figure_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Settings
# ============================================================

networks = [
    "random",
    "scale_free",
    "modular",
    "small_world"
]

network_labels = {
    "random": "ER",
    "scale_free": "BA",
    "modular": "SBM",
    "small_world": "WS"
}

network_titles = {
    "random": "Random",
    "scale_free": "Scale-free",
    "modular": "Modular",
    "small_world": "Small-world"
}

panel_labels = [
    "A",
    "B",
    "C",
    "D"
]

sub_network_size = 100

prob_wire = np.linspace(
    0,
    1,
    100
)

ov_node = np.arange(
    1,
    sub_network_size + 1
)

ov_degree = (
    ov_node
    / sub_network_size
)


# ============================================================
# Load data
# ============================================================

mean_data = {}
sem_data = {}

for network in networks:

    label = network_labels[
        network
    ]

    mean_path = (
        mean_data_dir
        / network
        / f"{label}_synergy_mean.csv"
    )

    sem_path = (
        sem_data_dir
        / network
        / f"{label}_synergy_sem.csv"
    )

    mean_data[network] = pd.read_csv(
        mean_path
    ).values

    sem_data[network] = pd.read_csv(
        sem_path
    ).values


# ============================================================
# Fig. S1
# Absolute SEM heatmaps
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 10)
)

axes = axes.flatten()

for i, network in enumerate(
    networks
):

    ax = axes[i]

    Z = sem_data[
        network
    ]

    heatmap = sns.heatmap(
        Z[::-1],
        cmap="viridis",
        cbar=True,
        ax=ax,
        xticklabels=False,
        yticklabels=False
    )

    # --------------------------------------------------------
    # Axis
    # --------------------------------------------------------

    ax.set_xlabel(
        r"$p_w$",
        fontsize=20
    )

    ax.set_ylabel(
        r"$D_{ov}$",
        fontsize=20
    )

    ax.set_title(
        network_titles[network],
        fontsize=18
    )

    ax.set_xticks(
        [
            0.5,
            50.5,
            99.5
        ]
    )

    ax.set_xticklabels(
        [
            "0",
            "0.5",
            "1"
        ],
        fontsize=15,
        rotation=0
    )

    ax.set_yticks(
        [
            0.5,
            50.5,
            99.5
        ]
    )

    ax.set_yticklabels(
        [
            "1",
            "0.5",
            "0"
        ],
        fontsize=15,
        rotation=0
    )

    # --------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------

    cbar = (
        heatmap.collections[0]
        .colorbar
    )

    cbar.ax.tick_params(
        labelsize=13
    )

    cbar.locator = MaxNLocator(
        nbins=5
    )

    cbar.update_ticks()

    cbar.set_label(
        "SEM of synergistic node pairs",
        fontsize=14
    )

    # --------------------------------------------------------
    # Panel label
    # --------------------------------------------------------

    ax.text(
        -0.14,
        1.06,
        panel_labels[i],
        transform=ax.transAxes,
        fontsize=22
    )

    ax.collections[0].set_rasterized(
        True
    )


plt.tight_layout()

plt.savefig(
    figure_dir / "Fig.S1.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Fig. S2
# Relative SEM heatmaps
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 10)
)

axes = axes.flatten()

for i, network in enumerate(
    networks
):

    ax = axes[i]

    mean = mean_data[
        network
    ]

    sem = sem_data[
        network
    ]

    # --------------------------------------------------------
    # Relative SEM
    # --------------------------------------------------------

    with np.errstate(
        divide="ignore",
        invalid="ignore"
    ):

        rsem = (
            sem
            / mean
        )

    # --------------------------------------------------------
    # Omit cells with mean < 1% of maximum mean
    # --------------------------------------------------------

    threshold = (
        0.01
        * np.nanmax(mean)
    )

    mask = (
        mean
        < threshold
    )

    rsem = rsem.copy()

    rsem[mask] = np.nan

    rsem[
        ~np.isfinite(rsem)
    ] = np.nan

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    heatmap = sns.heatmap(
        rsem[::-1],
        cmap="viridis",
        cbar=True,
        ax=ax,
        xticklabels=False,
        yticklabels=False
    )

    # --------------------------------------------------------
    # Axis
    # --------------------------------------------------------

    ax.set_xlabel(
        r"$p_w$",
        fontsize=20
    )

    ax.set_ylabel(
        r"$D_{ov}$",
        fontsize=20
    )

    ax.set_title(
        network_titles[network],
        fontsize=18
    )

    ax.set_xticks(
        [
            0.5,
            50.5,
            99.5
        ]
    )

    ax.set_xticklabels(
        [
            "0",
            "0.5",
            "1"
        ],
        fontsize=15,
        rotation=0
    )

    ax.set_yticks(
        [
            0.5,
            50.5,
            99.5
        ]
    )

    ax.set_yticklabels(
        [
            "1",
            "0.5",
            "0"
        ],
        fontsize=15,
        rotation=0
    )

    # --------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------

    cbar = (
        heatmap.collections[0]
        .colorbar
    )

    cbar.ax.tick_params(
        labelsize=13
    )

    cbar.locator = MaxNLocator(
        nbins=5
    )

    cbar.update_ticks()

    cbar.set_label(
        "Relative SEM (SEM / mean)",
        fontsize=14
    )

    # --------------------------------------------------------
    # Panel label
    # --------------------------------------------------------

    ax.text(
        -0.14,
        1.06,
        panel_labels[i],
        transform=ax.transAxes,
        fontsize=22
    )

    ax.collections[0].set_rasterized(
        True
    )


plt.tight_layout()

plt.savefig(
    figure_dir / "Fig.S2.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Fig. S3
# Representative p_w cross-sections
# ============================================================

target_pw = [
    0.1,
    0.5,
    0.9
]

# Use the nearest available p_w value
pw_indices = [
    np.argmin(
        np.abs(
            prob_wire - value
        )
    )
    for value in target_pw
]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 9),
    sharex=True
)

axes = axes.flatten()


for i, network in enumerate(
    networks
):

    ax = axes[i]

    mean = mean_data[
        network
    ]

    sem = sem_data[
        network
    ]

    for pw_index in pw_indices:

        actual_pw = prob_wire[
            pw_index
        ]

        mean_curve = mean[
            :,
            pw_index
        ]

        sem_curve = sem[
            :,
            pw_index
        ]

        # ----------------------------------------------------
        # Mean
        # ----------------------------------------------------

        line = ax.plot(
            ov_degree,
            mean_curve,
            linewidth=2,
            label=rf"$p_w = {actual_pw:.2f}$"
        )[0]

        # ----------------------------------------------------
        # ±1 SEM
        # ----------------------------------------------------

        ax.fill_between(
            ov_degree,
            mean_curve - sem_curve,
            mean_curve + sem_curve,
            alpha=0.18,
            color=line.get_color(),
            linewidth=0
        )

    # --------------------------------------------------------
    # Axis
    # --------------------------------------------------------

    ax.set_xlim(
        0,
        1
    )

    ax.set_xticks(
        [
            0,
            0.5,
            1
        ]
    )

    ax.set_xlabel(
        r"$D_{ov}$",
        fontsize=20
    )

    ax.set_ylabel(
        "Synergistic node pairs",
        fontsize=18
    )

    ax.set_title(
        network_titles[network],
        fontsize=18
    )

    ax.tick_params(
        axis="both",
        labelsize=14
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    ax.legend(
        frameon=False,
        fontsize=12
    )

    # --------------------------------------------------------
    # Panel label
    # --------------------------------------------------------

    ax.text(
        -0.14,
        1.06,
        panel_labels[i],
        transform=ax.transAxes,
        fontsize=22
    )


plt.tight_layout()

plt.savefig(
    figure_dir / "Fig.S3.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved Fig. S1, Fig. S2, and Fig. S3."
)