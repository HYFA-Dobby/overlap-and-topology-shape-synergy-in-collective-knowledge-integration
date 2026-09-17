import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


"""
Visualization of Fig. S4 and Fig. S5.

Fig. S4:
    Absolute numbers of unique and redundant node pairs.

Fig. S5:
    Values normalized by the maximum across D_ov for each p_w.

A separate figure is generated for each network and value type.

Input:
    visualization_data/Fig.S4-S5/{network}/
        {network_label}_unique_A_mean.csv
        {network_label}_redundancy_mean.csv

Output:
    figures/Fig.S4-S5/
"""


# ============================================================
# Paths
# ============================================================

data_dir = Path(
    "./visualization_data/Fig.S4-S5"
)

figure_dir = Path(
    "./figures/Fig.S4-S5"
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

val_types = [
    "unique_A",
    "redundancy"
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
# Visualization
# ============================================================

for network in networks:

    label = network_labels[
        network
    ]

    for val in val_types:

        # ----------------------------------------------------
        # Input
        # ----------------------------------------------------

        mean_path = (
            data_dir
            / network
            / f"{label}_{val}_mean.csv"
        )

        df_original = pd.read_csv(
            mean_path
        )


        # ====================================================
        # Fig. S4
        # Absolute values
        #
        # Same visualization as Fig. 2A
        # ====================================================

        X, Y = np.meshgrid(
            prob_wire,
            ov_degree
        )

        Z = df_original.values

        # Smoothing is used only for contour lines.
        Z_smooth = gaussian_filter(
            Z,
            sigma=1
        )

        fig, ax = plt.subplots(
            figsize=(12, 10)
        )

        # ----------------------------------------------------
        # Filled heatmap
        # ----------------------------------------------------

        ax.contourf(
            X,
            Y,
            Z,
            levels=30,
            cmap="viridis"
        )

        # ----------------------------------------------------
        # White contours
        # ----------------------------------------------------

        contour = ax.contour(
            X,
            Y,
            Z_smooth,
            levels=5,
            colors="white",
            linewidths=1.2
        )

        ax.clabel(
            contour,
            inline=True,
            fontsize=30
        )

        # ----------------------------------------------------
        # Colorbar
        # ----------------------------------------------------

        norm = Normalize(
            vmin=np.min(Z),
            vmax=np.max(Z)
        )

        sm = ScalarMappable(
            norm=norm,
            cmap="viridis"
        )

        sm.set_array([])

        cbar = plt.colorbar(
            sm,
            ax=ax
        )

        cbar.ax.tick_params(
            labelsize=40
        )

        cbar.locator = MaxNLocator(
            nbins=4
        )

        cbar.update_ticks()

        # ----------------------------------------------------
        # Axis
        # ----------------------------------------------------

        ax.set_xlabel(
            r"$p_w$",
            fontsize=45,
            labelpad=10
        )

        ax.set_ylabel(
            r"$D_{ov}$",
            fontsize=45,
            labelpad=10
        )

        ax.tick_params(
            axis="both",
            labelsize=40
        )

        ax.locator_params(
            axis="x",
            nbins=3
        )

        ax.locator_params(
            axis="y",
            nbins=3
        )

        # ----------------------------------------------------
        # Save Fig. S4
        # ----------------------------------------------------

        output_path_S4 = (
            figure_dir
            / f"{label}_FigS4_{val}.pdf"
        )

        plt.savefig(
            output_path_S4,
            dpi=100,
            bbox_inches="tight"
        )

        plt.close()


        # ====================================================
        # Fig. S5
        # Normalized values
        #
        # Same visualization as Fig. 2B
        # ====================================================

        df_normalized = (
            df_original.copy()
        )

        # ----------------------------------------------------
        # Normalize each p_w column by its maximum over D_ov
        # ----------------------------------------------------

        for col in range(
            df_normalized.shape[1]
        ):

            column_values = (
                df_normalized.iloc[
                    :,
                    col
                ]
            )

            max_value = (
                column_values.max()
            )

            if max_value != 0:

                df_normalized.iloc[
                    :,
                    col
                ] = (
                    column_values
                    / max_value
                )

        # ----------------------------------------------------
        # Reverse D_ov direction for seaborn heatmap
        # ----------------------------------------------------

        df_normalized = (
            df_normalized[::-1]
        )

        df_normalized.index = [
            f"{value:.3f}"
            for value in ov_degree[::-1]
        ]

        df_normalized.columns = [
            f"{value:.3f}"
            for value in prob_wire
        ]

        # ----------------------------------------------------
        # Tick locations
        # ----------------------------------------------------

        tick_vals = [
            0,
            0.5,
            1
        ]

        xtick_indices = [
            np.argmin(
                np.abs(
                    prob_wire - value
                )
            )
            for value in tick_vals
        ]

        reversed_ov_degree = (
            ov_degree[::-1]
        )

        ytick_indices = [
            np.argmin(
                np.abs(
                    reversed_ov_degree - value
                )
            )
            for value in tick_vals
        ]

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(12, 10)
        )

        heatmap = sns.heatmap(
            df_normalized,
            square=True,
            cmap="viridis",
            linewidths=0,
            vmin=0,
            vmax=1,
            cbar=True,
            cbar_kws={
                "ticks": [
                    0,
                    0.5,
                    1
                ]
            },
            ax=ax
        )

        # ----------------------------------------------------
        # Axis labels
        # ----------------------------------------------------

        ax.set_xlabel(
            r"$p_w$",
            fontsize=45,
            labelpad=10
        )

        ax.set_ylabel(
            r"$D_{ov}$",
            fontsize=45,
            labelpad=10
        )

        # ----------------------------------------------------
        # X ticks
        # ----------------------------------------------------

        ax.set_xticks(
            [
                index + 0.5
                for index in xtick_indices
            ]
        )

        ax.set_xticklabels(
            np.round(
                prob_wire[
                    xtick_indices
                ],
                1
            ),
            fontsize=40,
            rotation=0
        )

        # ----------------------------------------------------
        # Y ticks
        # ----------------------------------------------------

        ax.set_yticks(
            [
                index + 0.5
                for index in ytick_indices
            ]
        )

        ax.set_yticklabels(
            np.round(
                reversed_ov_degree[
                    ytick_indices
                ],
                1
            ),
            fontsize=40,
            rotation=0
        )

        # ----------------------------------------------------
        # Colorbar
        # ----------------------------------------------------

        cbar = (
            heatmap.collections[0]
            .colorbar
        )

        cbar.ax.tick_params(
            labelsize=40
        )

        # Rasterize heatmap only.
        ax.collections[0].set_rasterized(
            True
        )

        # ----------------------------------------------------
        # Save Fig. S5
        # ----------------------------------------------------

        output_path_S5 = (
            figure_dir
            / f"{label}_FigS5_{val}.pdf"
        )

        plt.savefig(
            output_path_S5,
            dpi=100,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Finished: "
            f"{network} | {val}"
        )