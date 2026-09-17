import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from pathlib import Path
from itertools import product
from matplotlib.lines import Line2D


"""
Visualization of Fig. 4A.

Strict external-synergy discovery events are shown as a
function of overlap degree D_ov for different Watts-Strogatz
rewiring probabilities p.

Input:
    visualization_data/Fig.4/random_walk/
        syn_result_2agent_p0.0.csv
        syn_result_2agent_p0.0001.csv
        ...

CSV format:
    ov, pattern, mean, sem

Output:
    figures/Fig.4/Fig4A_random_walk.pdf
"""


# ============================================================
# Paths
# ============================================================

data_dir = Path(
    "./visualization_data/Fig.4/random_walk"
)

figure_dir = Path(
    "./figures/Fig.4"
)

figure_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Settings
# ============================================================

N_agent = 2
sub_network_size = 100

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

def fill_missing_patterns(
    df,
    n_person
):

    """
    Fill missing Boolean patterns with mean = 0 and sem = 0.

    For n individuals, the pattern length is n + 1.
    The final bit corresponds to the common region.
    """

    bit_length = (
        n_person + 1
    )

    df = df.copy()

    # --------------------------------------------------------
    # Restore leading zeros
    # --------------------------------------------------------

    df["pattern"] = (
        df["pattern"]
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True
        )
        .str.zfill(
            bit_length
        )
    )

    # --------------------------------------------------------
    # All possible non-zero patterns
    # --------------------------------------------------------

    patterns = [
        "".join(bits)
        for bits in product(
            "01",
            repeat=bit_length
        )
        if any(
            bit == "1"
            for bit in bits
        )
    ]

    ov_values = sorted(
        df["ov"].unique()
    )

    full = (
        pd.MultiIndex
        .from_product(
            [
                ov_values,
                patterns
            ],
            names=[
                "ov",
                "pattern"
            ]
        )
        .to_frame(
            index=False
        )
    )

    out = full.merge(
        df,
        how="left",
        on=[
            "ov",
            "pattern"
        ]
    )

    out["mean"] = (
        out["mean"]
        .fillna(0)
    )

    out["sem"] = (
        out["sem"]
        .fillna(0)
    )

    return out


# ============================================================
# Fig. 4A
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 6)
)


for p_s in prob_list:

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    data_path = (
        data_dir
        / f"syn_result_2agent_p{p_s}.csv"
    )

    df = pd.read_csv(
        data_path,
        dtype={
            "pattern": str
        }
    )

    df = fill_missing_patterns(
        df,
        N_agent
    )

    # --------------------------------------------------------
    # Strict synergy
    #
    # For 2 agents:
    # U_A = 1, U_B = 1, R = 1
    # -> 111
    # --------------------------------------------------------

    target = (
        "1"
        * (N_agent + 1)
    )

    df_syn = (
        df[
            df["pattern"] == target
        ]
        .sort_values(
            "ov"
        )
    )

    x = (
        df_syn["ov"]
        / sub_network_size
    )

    y = (
        df_syn["mean"]
    )

    sem = (
        df_syn["sem"]
    )

    # --------------------------------------------------------
    # Color
    # --------------------------------------------------------

    if np.isclose(
        p_s,
        0.0
    ):

        color = "black"

    else:

        color = cmap(
            norm(p_s)
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
        y - sem,
        y + sem,
        color=color,
        alpha=0.15,
        linewidth=0
    )


# ============================================================
# Axis
# ============================================================

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
    "Discovery events",
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

# ============================================================
# Colorbar for p > 0
# ============================================================

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


# ============================================================
# Legend for p = 0
# ============================================================

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
# Layout
# ============================================================

plt.tight_layout()


# ============================================================
# Save
# ============================================================

output_path = (
    figure_dir
    / "Fig4A_random_walk.pdf"
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