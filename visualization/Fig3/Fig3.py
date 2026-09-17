import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from itertools import product


"""
Visualization of Fig. 3A and Fig. 3B.

Fig. 3A:
    First-exit discovery events classified as:
        Unq(A)   = 100
        Rnd(A,B) = 001
        Syn(A,B) = 111

    In addition, for random, scale-free, and modular networks,
    a separate synergy-only plot is also generated.

Fig. 3B:
    Strict synergistic discovery events for increasing group size.
    Strict synergy corresponds to the all-ones pattern.

Input:
    visualization_data/Fig.3/{network}/
        {label}_syn_result_{n}agent.csv

CSV format:
    ov, pattern, mean, sem

Output:
    figures/Fig.3/
"""


# ============================================================
# Paths
# ============================================================

data_dir = Path(
    "./visualization_data/Fig.3"
)

figure_dir = Path(
    "./figures/Fig.3"
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

# Fig. 3B
agent_numbers = {
    "random": [2, 3, 4, 5],
    "scale_free": [2, 3, 4, 5],
    "modular": [2, 3, 4, 5],
    "small_world": list(range(2, 11))
}

sub_network_size = 100


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

    Pattern length:
        n_person + 1

    The last bit corresponds to the common region R.
    """

    bit_length = n_person + 1

    df = df.copy()

    # Convert patterns to zero-padded strings
    df["pattern"] = (
        df["pattern"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(bit_length)
    )

    # All non-zero patterns
    patterns = [
        "".join(bits)
        for bits in product(
            "01",
            repeat=bit_length
        )
        if any(bit == "1" for bit in bits)
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
    ).to_frame(index=False)

    out = full.merge(
        df,
        how="left",
        on=[
            "ov",
            "pattern"
        ]
    )

    out["mean"] = out["mean"].fillna(0)
    out["sem"] = out["sem"].fillna(0)

    return out


# ============================================================
# Fig. 3A
# Unq(A), Rnd(A,B), Syn(A,B)
# ============================================================

for network in networks:

    label = network_labels[network]

    # Fig. 3A always uses 2 agents
    n_agents = 2

    data_path = (
        data_dir
        / network
        / f"{label}_syn_result_{n_agents}agent.csv"
    )

    df = pd.read_csv(
        data_path
    )

    df = fill_missing_patterns(
        df,
        n_agents
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    df_unq = (
        df[df["pattern"] == "100"]
        .sort_values("ov")
    )

    df_rnd = (
        df[df["pattern"] == "001"]
        .sort_values("ov")
    )

    df_syn = (
        df[df["pattern"] == "111"]
        .sort_values("ov")
    )

    # ========================================================
    # Main Fig. 3A
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    # Unq(A)
    line, = ax.plot(
        df_unq["ov"] / sub_network_size,
        df_unq["mean"],
        linewidth=2.5,
        label="Unq(A)"
    )
    color = line.get_color()

    ax.fill_between(
        df_unq["ov"] / sub_network_size,
        df_unq["mean"] - df_unq["sem"],
        df_unq["mean"] + df_unq["sem"],
        color=color,
        alpha=0.15,
        linewidth=0
    )

    # Rnd(A,B)
    line, = ax.plot(
        df_rnd["ov"] / sub_network_size,
        df_rnd["mean"],
        linewidth=2.5,
        label="Rnd(A,B)"
    )
    color = line.get_color()

    ax.fill_between(
        df_rnd["ov"] / sub_network_size,
        df_rnd["mean"] - df_rnd["sem"],
        df_rnd["mean"] + df_rnd["sem"],
        color=color,
        alpha=0.15,
        linewidth=0
    )

    # Syn(A,B)
    line, = ax.plot(
        df_syn["ov"] / sub_network_size,
        df_syn["mean"],
        linewidth=2.5,
        label="Syn(A,B)"
    )
    color = line.get_color()

    ax.fill_between(
        df_syn["ov"] / sub_network_size,
        df_syn["mean"] - df_syn["sem"],
        df_syn["mean"] + df_syn["sem"],
        color=color,
        alpha=0.15,
        linewidth=0
    )

    # Axis
    ax.set_xlim(0, 1)

    ax.set_xlabel(
        r"$D_{ov}$",
        fontsize=30
    )

    ax.set_ylabel(
        "Discovery events",
        fontsize=30
    )

    ax.set_title(
        network_titles[network],
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

    ax.legend(
        frameon=False,
        fontsize=18
    )

    plt.tight_layout()

    output_path = (
        figure_dir
        / f"{label}_Fig3A.pdf"
    )

    plt.savefig(
        output_path,
        dpi=100,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Finished Fig. 3A: {network}"
    )

    # ========================================================
    # Additional synergy-only plot
    # for networks where synergy is hard to see
    # ========================================================

    if network in ["random", "scale_free", "modular"]:

        fig, ax = plt.subplots(
            figsize=(7, 6)
        )

        ax.plot(
            df_syn["ov"] / sub_network_size,
            df_syn["mean"],
            linewidth=2.5,
            label="Syn(A,B)",
            color="green"
        )

        ax.fill_between(
            df_syn["ov"] / sub_network_size,
            df_syn["mean"] - df_syn["sem"],
            df_syn["mean"] + df_syn["sem"],
            color="green",
            alpha=0.15,
            linewidth=0
        )

        ymax = np.max(
            df_syn["mean"] + df_syn["sem"]
        )

        if ymax > 0:
            ax.set_ylim(0, ymax * 1.1)

        ax.set_xlim(0, 1)

        ax.set_xlabel(
            r"$D_{ov}$",
            fontsize=30
        )

        ax.set_ylabel(
            "Synergistic\ndiscovery events",
            fontsize=30
        )

        ax.set_title(
            f"{network_titles[network]} (Synergy only)",
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

        ax.legend(
            frameon=False,
            fontsize=18
        )

        plt.tight_layout()

        output_path_zoom = (
            figure_dir
            / f"{label}_Fig3A_synergy_only.pdf"
        )

        plt.savefig(
            output_path_zoom,
            dpi=100,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Finished Fig. 3A synergy-only: {network}"
        )


# ============================================================
# Fig. 3B
# Strict synergy vs group size
# ============================================================

for network in networks:

    label = network_labels[network]

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    for n_agents in agent_numbers[network]:

        data_path = (
            data_dir
            / network
            / f"{label}_syn_result_{n_agents}agent.csv"
        )

        df = pd.read_csv(
            data_path
        )

        df = fill_missing_patterns(
            df,
            n_agents
        )

        # Strict synergy = all-ones pattern
        target = "1" * (n_agents + 1)

        df_syn = (
            df[df["pattern"] == target]
            .sort_values("ov")
        )

        line, = ax.plot(
            df_syn["ov"] / sub_network_size,
            df_syn["mean"],
            linewidth=2.5,
            label=str(n_agents)
        )

        color = line.get_color()

        ax.fill_between(
            df_syn["ov"] / sub_network_size,
            df_syn["mean"] - df_syn["sem"],
            df_syn["mean"] + df_syn["sem"],
            color=color,
            alpha=0.15,
            linewidth=0
        )

    # Axis
    ax.set_xlim(0, 1)

    ax.set_xlabel(
        r"$D_{ov}$",
        fontsize=30
    )

    ax.set_ylabel(
        "Synergistic\ndiscovery events",
        fontsize=30
    )

    ax.set_title(
        network_titles[network],
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

    ax.legend(
        title="Group size",
        frameon=False,
        fontsize=15,
        title_fontsize=16
    )

    plt.tight_layout()

    output_path = (
        figure_dir
        / f"{label}_Fig3B.pdf"
    )

    plt.savefig(
        output_path,
        dpi=100,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Finished Fig. 3B: {network}"
    )


print(
    "Saved all Fig. 3A and Fig. 3B panels."
)