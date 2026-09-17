import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


"""
Visualization of Fig. S8.

Fig. S8A:
    Mean number of strict external-synergy discovery events.

Fig. S8B:
    SEM of strict external-synergy discovery events.

Input:
    visualization_data/Fig.S8/{network}/
        {network_label}_mean_2agent_heatmap.csv
        {network_label}_sem_2agent_heatmap.csv

Output:
    figures/Fig.S8/
"""


# ============================================================
# Paths
# ============================================================

data_dir = Path(
    "./visualization_data/Fig.S8"
)

figure_dir = Path(
    "./figures/Fig.S8"
)

figure_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Settings
# ============================================================

N_agent = 2

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

value_types = [
    "mean",
    "sem"
]


# ============================================================
# Visualization
# ============================================================

for network in networks:

    label = network_labels[
        network
    ]

    for value_type in value_types:

        # ----------------------------------------------------
        # Input
        # ----------------------------------------------------

        path = (
            data_dir
            / network
            / f"{label}_{value_type}_{N_agent}agent_heatmap.csv"
        )

        df = pd.read_csv(
            path,
            index_col=0
        )

        dov = (
            df.index
            .astype(float)
            .values
        )

        pw = (
            df.columns
            .astype(float)
            .values
        )

        Z = df.values


        # ====================================================
        # Plot
        # ====================================================

        fig, ax = plt.subplots(
            figsize=(5, 4)
        )

        im = ax.pcolormesh(
            pw,
            dov,
            Z,
            shading="auto"
        )

        # ----------------------------------------------------
        # Axis
        # ----------------------------------------------------

        ax.set_xlabel(
            r"$p_w$",
            fontsize=18
        )

        ax.set_ylabel(
            r"$D_{ov}$",
            fontsize=18
        )

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=15
        )


        # ----------------------------------------------------
        # Colorbar
        # ----------------------------------------------------

        cbar = fig.colorbar(
            im,
            ax=ax
        )

        cbar.ax.tick_params(
            labelsize=15
        )


        # ====================================================
        # Save
        # ====================================================

        plt.tight_layout()

        output_path = (
            figure_dir
            / f"{label}_FigS8_{value_type}.pdf"
        )

        plt.savefig(
            output_path,
            dpi=100,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Finished Fig. S8: "
            f"{network} | {value_type}"
        )


print(
    "Saved all Fig. S8 panels."
)