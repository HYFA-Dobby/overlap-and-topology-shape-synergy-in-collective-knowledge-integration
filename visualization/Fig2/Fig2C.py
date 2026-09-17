import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


"""
Visualization of Fig. 2C.

For each background-network structure, synergistic node pairs per
individual are plotted as a function of overlap degree D_ov.

Each line corresponds to a different number of agents.
Shaded regions indicate the standard error of the mean (SEM).

Input data:
    visualization_data/Fig.2/C/{network}/

Example:
    visualization_data/Fig.2/C/small_world/
        WS_synergy_mean_2agent_pw_0.500.csv
        WS_synergy_sem_2agent_pw_0.500.csv
        ...
        WS_synergy_mean_10agent_pw_0.500.csv
        WS_synergy_sem_10agent_pw_0.500.csv

Output:
    figures/Fig.2/Fig2C.pdf
"""


# ============================================================
# Paths
# ============================================================

data_dir = Path(
    "./visualization_data/Fig.2/C"
)

figure_dir = Path(
    "./figures/Fig.2"
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
    "scale_free": "Scale-Free",
    "modular": "Modular",
    "small_world": "Small-world"
}

prob_wire = 0.5

sub_network_size = 100

agent_numbers = list(
    range(2, 11)
)

# Same color is used for the same group size
# across all four network panels.
cmap = plt.get_cmap(
    "tab10"
)

agent_colors = {
    n: cmap(i)
    for i, n in enumerate(agent_numbers)
}


# ============================================================
# Figure
# ============================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(20, 5),
    sharex=True
)


# ============================================================
# Network panels
# ============================================================

for ax, network in zip(
    axes,
    networks
):

    label = network_labels[
        network
    ]

    network_dir = (
        data_dir
        / network
    )

    # ========================================================
    # Group size
    # ========================================================

    for n_agents in agent_numbers:

        mean_path = (
            network_dir
            / (
                f"{label}_synergy_"
                f"mean_{n_agents}agent_pw_{prob_wire:.3f}.csv"
            )
        )

        sem_path = (
            network_dir
            / (
                f"{label}_synergy_"
                f"sem_{n_agents}agent_pw_{prob_wire:.3f}.csv"
            )
        )

        # Skip group sizes for which data are not available.
        if (
            not mean_path.exists()
            or not sem_path.exists()
        ):
            continue

        # ----------------------------------------------------
        # Load data
        # ----------------------------------------------------

        mean_df = pd.read_csv(
            mean_path
        )

        sem_df = pd.read_csv(
            sem_path
        )

        # First column: overlap node number
        ov = mean_df.iloc[
            :,
            0
        ].to_numpy()

        # Second column: mean / SEM
        synergy_mean = mean_df.iloc[
            :,
            1
        ].to_numpy()

        synergy_sem = sem_df.iloc[
            :,
            1
        ].to_numpy()

        # ----------------------------------------------------
        # Convert overlap-node number to D_ov
        # ----------------------------------------------------

        dov = (
            ov
            / sub_network_size
        )

        # ----------------------------------------------------
        # Fig. 2C reports synergy per individual
        # ----------------------------------------------------

        synergy_mean_per_agent = (
            synergy_mean
            / n_agents
        )

        synergy_sem_per_agent = (
            synergy_sem
            / n_agents
        )

        # ----------------------------------------------------
        # Mean
        # ----------------------------------------------------

        ax.plot(
            dov,
            synergy_mean_per_agent,
            linewidth=2,
            color=agent_colors[n_agents],
            label=str(n_agents)
        )

        # ----------------------------------------------------
        # SEM
        # ----------------------------------------------------

        ax.fill_between(
            dov,
            synergy_mean_per_agent
            - synergy_sem_per_agent,
            synergy_mean_per_agent
            + synergy_sem_per_agent,
            color=agent_colors[n_agents],
            alpha=0.15,
            linewidth=0
        )

    # ========================================================
    # Panel formatting
    # ========================================================

    ax.set_title(
        network_titles[network],
        fontsize=20
    )

    ax.set_xlabel(
        r"$D_{ov}$",
        fontsize=20
    )

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

    ax.tick_params(
        axis="both",
        labelsize=16
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)


# ============================================================
# Y axis
# ============================================================

axes[0].set_ylabel(
    "Node pairs\nper individual",
    fontsize=20
)


# ============================================================
# Legend
# ============================================================

handles, labels = axes[-1].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    title="Group size",
    title_fontsize=16,
    fontsize=14,
    frameon=False,
    loc="center right",
    bbox_to_anchor=(1.02, 0.5)
)


# ============================================================
# Layout
# ============================================================

plt.tight_layout(
    rect=[
        0,
        0,
        0.95,
        1
    ]
)


# ============================================================
# Save
# ============================================================

output_path = (
    figure_dir
    / "Fig2C.pdf"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {output_path}"
)