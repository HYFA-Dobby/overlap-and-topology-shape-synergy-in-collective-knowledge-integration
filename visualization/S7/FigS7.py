import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from itertools import product


"""
Visualization of Fig. S7.

All possible non-zero connection patterns for two individuals
are shown as functions of overlap degree D_ov.

Patterns:
    100 : Unq(A)
    010 : Unq(B)
    001 : Rnd(A,B)
    101 : Unq(A) & Rnd(A,B)
    011 : Unq(B) & Rnd(A,B)
    110 : Unq(A) & Unq(B)
    111 : Syn(A,B)

Input:
    visualization_data/Fig.3/{network}/
        {label}_syn_result_2agent.csv

CSV format:
    ov, pattern, mean, sem

Output:
    figures/Fig.S7/
"""


# ============================================================
# Paths
# ============================================================

data_dir = Path(
    "./visualization_data/Fig.3"
)

figure_dir = Path(
    "./figures/Fig.S7"
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

sub_network_size = 100
n_agents = 2


# ============================================================
# Pattern labels
# ============================================================

pattern_labels = {
    "100": "Unq(A)",
    "010": "Unq(B)",
    "001": "Rnd(A,B)",
    "101": "Unq(A) & Rnd(A,B)",
    "011": "Unq(B) & Rnd(A,B)",
    "110": "Unq(A) & Unq(B)",
    "111": "Syn(A,B)"
}

pattern_order = [
    "100",
    "010",
    "001",
    "101",
    "011",
    "110",
    "111"
]


# ============================================================
# Utility
# ============================================================

def fill_missing_patterns(df, n_person):
    """
    Fill missing connection patterns with mean = 0 and sem = 0.

    Parameters
    ----------
    df:
        columns = [ov, pattern, mean, sem]

    n_person:
        number of individuals

    For two individuals, the pattern length is 3:
        [U_A, U_B, R]
    """

    bit_length = n_person + 1

    df = df.copy()

    # --------------------------------------------------------
    # Preserve leading zeros
    # --------------------------------------------------------

    df["pattern"] = (
        df["pattern"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(bit_length)
    )

    # --------------------------------------------------------
    # All non-zero patterns
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

    full = pd.MultiIndex.from_product(
        [
            ov_values,
            patterns
        ],
        names=[
            "ov",
            "pattern"
        ]
    ).to_frame(
        index=False
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
# Fig. S7
# ============================================================

for network in networks:

    label = network_labels[
        network
    ]

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    data_path = (
        data_dir
        / network
        / f"{label}_syn_result_{n_agents}agent.csv"
    )

    df = pd.read_csv(
        data_path,
        dtype={
            "pattern": str
        }
    )

    df = fill_missing_patterns(
        df,
        n_agents
    )


    # ========================================================
    # Plot
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )


    # --------------------------------------------------------
    # Plot all seven patterns
    # --------------------------------------------------------

    for pattern in pattern_order:

        df_pattern = (
            df[
                df["pattern"] == pattern
            ]
            .sort_values("ov")
        )

        line, = ax.plot(
            df_pattern["ov"] / sub_network_size,
            df_pattern["mean"],
            linewidth=2.5,
            label=pattern_labels[
                pattern
            ]
        )

        color = line.get_color()

        ax.fill_between(
            df_pattern["ov"] / sub_network_size,
            df_pattern["mean"] - df_pattern["sem"],
            df_pattern["mean"] + df_pattern["sem"],
            color=color,
            alpha=0.15,
            linewidth=0
        )


    # ========================================================
    # Axis
    # ========================================================

    ax.set_xlim(
        0,
        1
    )

    ax.set_xlabel(
        r"$D_{ov}$",
        fontsize=30
    )

    ax.set_ylabel(
        "Discovery events",
        fontsize=30
    )

    ax.set_title(
        network_titles[
            network
        ],
        fontsize=25
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


    # ========================================================
    # Legend
    # ========================================================

    ax.legend(
        loc="center left",
        bbox_to_anchor=(
            1.02,
            0.5
        ),
        frameon=False,
        fontsize=15
    )


    # ========================================================
    # Save
    # ========================================================

    plt.tight_layout()

    output_path = (
        figure_dir
        / f"{label}_FigS7.pdf"
    )

    plt.savefig(
        output_path,
        dpi=100,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Finished Fig. S7: {network}"
    )


print(
    "Saved all Fig. S7 panels."
)