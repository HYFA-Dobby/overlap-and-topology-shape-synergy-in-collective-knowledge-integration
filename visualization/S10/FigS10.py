from pathlib import Path
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from multiprocessing import Pool, cpu_count

from src import subnetwork as sn
from src import integnetwork as ig


"""
Reproduction of Figure S10.

Representative boundary organization across knowledge overlap
and Watts-Strogatz rewiring.

Columns:
    D_ov = 0.20, 0.69, 0.88

Rows:
    p = 1e-4, 1.668100537e-2, 1.0

For each (p, D_ov) condition, multiple independent realizations
are generated. The realization whose boundary concentration is
closest to the median is selected for visualization.

Outputs:
    visualization_data/Fig.S10/
        candidate_boundary_statistics.csv
        selected_representative_realizations.csv

    figures/Fig.S10/
        Fig.S10.pdf
"""


# ============================================================
# Paths
# ============================================================

DATA_DIR = Path(
    "./visualization_data/Fig.S10"
)

FIGURE_DIR = Path(
    "./figures/Fig.S10"
)

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Global parameters
# ============================================================

N = 10**5
K = 20

SUB_NETWORK_SIZE = 100
P_W = 0.5


# ============================================================
# Figure S10 parameter combinations
# ============================================================

OV_VALUES = [
    20,
    69,
    88
]

P_VALUES = [
    1e-4,              # weak rewiring
    1.668100537e-2,    # moderate rewiring
    1.0                # strong rewiring
]


# Number of realizations generated for each condition
N_CANDIDATES = 50

# Limit memory use because each WS graph contains 10^5 nodes
N_PROCESSES = min(
    4,
    cpu_count()
)

MASTER_SEED = 12345


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    """
    Set NumPy and Python random seeds.
    """

    np.random.seed(seed)
    random.seed(seed)


# ============================================================
# Generate one realization
# ============================================================

def generate_realization(
    p,
    ov,
    seed
):
    """
    Generate one pair of individual knowledge networks and
    classify the external boundary.

    A structurally synergistic external node is connected to
    all three regions:

        U_A
        U_B
        R

    Returns
    -------
    dict
        Background network, individual/integrated networks,
        structural regions, external boundary, and boundary
        statistics.
    """

    set_seed(seed)


    # --------------------------------------------------------
    # Background network
    # --------------------------------------------------------

    G = nx.watts_strogatz_graph(
        N,
        k=K,
        p=p,
        seed=seed
    )

    nodes = np.asarray(
        G.nodes()
    )


    # --------------------------------------------------------
    # Individual subnetworks
    # --------------------------------------------------------

    attempt = 0

    while True:

        base_node = np.random.choice(
            nodes
        )

        sub_nodes, sub_networks = sn.generate_subnetworks(
            [
                SUB_NETWORK_SIZE,
                SUB_NETWORK_SIZE
            ],
            G,
            base_node,
            ov,
            P_W
        )

        if (
            len(sub_nodes[0]) == SUB_NETWORK_SIZE
            and len(sub_nodes[1]) == SUB_NETWORK_SIZE
        ):
            break

        attempt += 1

        if attempt > 1000:

            raise RuntimeError(
                f"Could not generate subnetworks: "
                f"p={p}, ov={ov}, seed={seed}"
            )


    # --------------------------------------------------------
    # Integrated network
    #
    # Pure union of individual knowledge networks.
    # No additional edges are introduced.
    # --------------------------------------------------------

    integ_network = ig.generate_integrated_network(
        G,
        *sub_networks
    )


    # --------------------------------------------------------
    # Structural regions
    # --------------------------------------------------------

    A = set(
        sub_networks[0].nodes()
    )

    B = set(
        sub_networks[1].nodes()
    )

    R = A & B
    UA = A - B
    UB = B - A

    integrated_nodes = (
        A | B
    )


    # --------------------------------------------------------
    # External one-step boundary
    # --------------------------------------------------------

    neighbors = set()

    for u in integrated_nodes:

        neighbors.update(
            G.neighbors(u)
        )

    neighbors -= integrated_nodes


    # --------------------------------------------------------
    # Structural synergy classification
    #
    # x is synergistic iff
    #
    # k_UA(x) > 0
    # k_UB(x) > 0
    # k_R(x)  > 0
    # --------------------------------------------------------

    synergy_neighbors = set()

    for x in neighbors:

        x_neighbors = set(
            G.neighbors(x)
        )

        has_A = not UA.isdisjoint(
            x_neighbors
        )

        has_B = not UB.isdisjoint(
            x_neighbors
        )

        has_R = not R.isdisjoint(
            x_neighbors
        )

        if (
            has_A
            and has_B
            and has_R
        ):
            synergy_neighbors.add(
                x
            )


    normal_neighbors = (
        neighbors
        - synergy_neighbors
    )


    # --------------------------------------------------------
    # Boundary concentration
    # --------------------------------------------------------

    if len(neighbors) > 0:

        boundary_concentration = (
            len(synergy_neighbors)
            / len(neighbors)
        )

    else:

        boundary_concentration = np.nan


    return {

        "G":
            G,

        "sub_networks":
            sub_networks,

        "integ_network":
            integ_network,

        "UA":
            UA,

        "UB":
            UB,

        "R":
            R,

        "neighbors":
            neighbors,

        "synergy_neighbors":
            synergy_neighbors,

        "normal_neighbors":
            normal_neighbors,

        "boundary_size":
            len(neighbors),

        "synergy_count":
            len(synergy_neighbors),

        "boundary_concentration":
            boundary_concentration
    }


# ============================================================
# Candidate measurement
# ============================================================

def measure_candidate(args):

    p, ov, seed = args

    res = generate_realization(
        p=p,
        ov=ov,
        seed=seed
    )

    return {

        "p":
            p,

        "ov":
            ov,

        "D_ov":
            ov / SUB_NETWORK_SIZE,

        "seed":
            seed,

        "boundary_size":
            res["boundary_size"],

        "synergy_count":
            res["synergy_count"],

        "boundary_concentration":
            res["boundary_concentration"]
    }


# ============================================================
# Generate independent candidate seeds
# ============================================================

def make_candidate_tasks():

    seed_sequence = (
        np.random.SeedSequence(
            MASTER_SEED
        )
    )

    total = (
        len(P_VALUES)
        * len(OV_VALUES)
        * N_CANDIDATES
    )

    child_sequences = (
        seed_sequence.spawn(
            total
        )
    )

    seeds = [
        int(
            s.generate_state(1)[0]
        )
        for s in child_sequences
    ]

    tasks = []

    count = 0

    for p in P_VALUES:

        for ov in OV_VALUES:

            for _ in range(
                N_CANDIDATES
            ):

                tasks.append(
                    (
                        p,
                        ov,
                        seeds[count]
                    )
                )

                count += 1

    return tasks


# ============================================================
# Select representative realization
# ============================================================

def select_median_representatives(
    df
):

    """
    Select one realization for each (p, D_ov) condition.

    Primary criterion:
        boundary concentration closest to the median.

    Tie breaker:
        boundary size closest to the median.
    """

    selected_records = []

    for p in P_VALUES:

        for ov in OV_VALUES:

            sub = df[
                np.isclose(
                    df["p"],
                    p
                )
                & (
                    df["ov"]
                    == ov
                )
            ].copy()

            sub = sub.dropna(
                subset=[
                    "boundary_concentration"
                ]
            )


            # ------------------------------------------------
            # Median boundary concentration
            # ------------------------------------------------

            median_c = (
                sub[
                    "boundary_concentration"
                ]
                .median()
            )

            sub[
                "distance_to_median"
            ] = np.abs(
                sub[
                    "boundary_concentration"
                ]
                - median_c
            )


            # ------------------------------------------------
            # Tie breaker:
            # median boundary size
            # ------------------------------------------------

            median_boundary_size = (
                sub[
                    "boundary_size"
                ]
                .median()
            )

            sub[
                "boundary_size_distance"
            ] = np.abs(
                sub[
                    "boundary_size"
                ]
                - median_boundary_size
            )


            # ------------------------------------------------
            # Representative
            # ------------------------------------------------

            selected = (
                sub
                .sort_values(
                    [
                        "distance_to_median",
                        "boundary_size_distance",
                        "seed"
                    ]
                )
                .iloc[0]
            )


            selected_records.append({

                "p":
                    p,

                "ov":
                    ov,

                "D_ov":
                    ov
                    / SUB_NETWORK_SIZE,

                "median_boundary_concentration":
                    median_c,

                "selected_boundary_concentration":
                    selected[
                        "boundary_concentration"
                    ],

                "selected_boundary_size":
                    selected[
                        "boundary_size"
                    ],

                "selected_synergy_count":
                    selected[
                        "synergy_count"
                    ],

                "selected_seed":
                    int(
                        selected["seed"]
                    )
            })


    return pd.DataFrame(
        selected_records
    )


# ============================================================
# Layout
# ============================================================

def make_boundary_layout(
    G,
    integ_network,
    external_nodes,
    seed,
    external_offset=0.10
):
    """
    Run Kamada-Kawai only on the integrated network.

    External boundary nodes are placed close to the centroid of
    their neighbors in the integrated network.

    Positions are used only for visualization and do not
    represent metric distances in the knowledge space.
    """

    integrated_nodes = set(
        integ_network.nodes()
    )


    # --------------------------------------------------------
    # Integrated network layout
    # --------------------------------------------------------

    pos = nx.kamada_kawai_layout(
        integ_network
    )

    rng = np.random.default_rng(
        seed
    )


    # --------------------------------------------------------
    # External nodes
    # --------------------------------------------------------

    for x in external_nodes:

        internal_neighbors = [

            u

            for u in G.neighbors(x)

            if u in integrated_nodes
        ]


        if len(
            internal_neighbors
        ) > 0:

            centroid = np.mean(
                [
                    pos[u]
                    for u
                    in internal_neighbors
                ],
                axis=0
            )

            jitter = rng.normal(
                loc=0.0,
                scale=external_offset,
                size=2
            )

            pos[x] = (
                centroid
                + jitter
            )

        else:

            pos[x] = rng.normal(
                size=2
            )


    return pos


# ============================================================
# Edge normalization
# ============================================================

def normalize_edge(
    u,
    v
):
    """
    Represent an undirected edge in a consistent orientation.
    """

    return (
        (u, v)
        if u < v
        else (v, u)
    )


# ============================================================
# Draw one representative snapshot
# ============================================================

def draw_snapshot(
    ax,
    realization,
    p,
    ov,
    seed
):

    G = realization[
        "G"
    ]

    integ_network = realization[
        "integ_network"
    ]

    subA = realization[
        "sub_networks"
    ][0]

    subB = realization[
        "sub_networks"
    ][1]

    UA = realization[
        "UA"
    ]

    UB = realization[
        "UB"
    ]

    R = realization[
        "R"
    ]

    synergy_neighbors = realization[
        "synergy_neighbors"
    ]

    normal_neighbors = realization[
        "normal_neighbors"
    ]

    external_nodes = (
        synergy_neighbors
        | normal_neighbors
    )


    # ========================================================
    # Layout
    # ========================================================

    pos = make_boundary_layout(
        G,
        integ_network,
        external_nodes,
        seed
    )


    # ========================================================
    # Internal edges by knowledge ownership
    # ========================================================

    edges_A = {
        normalize_edge(
            u,
            v
        )
        for u, v
        in subA.edges()
    }

    edges_B = {
        normalize_edge(
            u,
            v
        )
        for u, v
        in subB.edges()
    }


    internal_edges_A_only = []
    internal_edges_B_only = []
    internal_edges_shared = []


    for u, v in integ_network.edges():

        e = normalize_edge(
            u,
            v
        )

        in_A = (
            e in edges_A
        )

        in_B = (
            e in edges_B
        )


        if in_A and in_B:

            internal_edges_shared.append(
                (u, v)
            )

        elif in_A:

            internal_edges_A_only.append(
                (u, v)
            )

        elif in_B:

            internal_edges_B_only.append(
                (u, v)
            )


    # ========================================================
    # Draw internal edges
    # ========================================================

    nx.draw_networkx_edges(
        integ_network,
        pos,
        edgelist=internal_edges_A_only,
        ax=ax,
        edge_color="red",
        width=0.9,
        alpha=0.75
    )

    nx.draw_networkx_edges(
        integ_network,
        pos,
        edgelist=internal_edges_B_only,
        ax=ax,
        edge_color="dodgerblue",
        width=0.9,
        alpha=0.75
    )

    nx.draw_networkx_edges(
        integ_network,
        pos,
        edgelist=internal_edges_shared,
        ax=ax,
        edge_color="purple",
        width=1.0,
        alpha=0.80
    )


    # ========================================================
    # Normal external-boundary edges
    # ========================================================

    integrated_nodes = set(
        integ_network.nodes()
    )

    normal_edges = []

    for x in normal_neighbors:

        for u in G.neighbors(x):

            if u in integrated_nodes:

                normal_edges.append(
                    (
                        x,
                        u
                    )
                )


    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=normal_edges,
        ax=ax,
        edge_color="lightgray",
        width=0.35,
        alpha=0.12
    )


    # ========================================================
    # Edges from synergistic external nodes
    # ========================================================

    syn_edges_A = []
    syn_edges_B = []
    syn_edges_R = []


    for x in synergy_neighbors:

        for u in G.neighbors(x):

            if u in UA:

                syn_edges_A.append(
                    (
                        x,
                        u
                    )
                )

            elif u in UB:

                syn_edges_B.append(
                    (
                        x,
                        u
                    )
                )

            elif u in R:

                syn_edges_R.append(
                    (
                        x,
                        u
                    )
                )


    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=syn_edges_A,
        ax=ax,
        edge_color="red",
        width=1.0,
        alpha=0.75
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=syn_edges_B,
        ax=ax,
        edge_color="dodgerblue",
        width=1.0,
        alpha=0.75
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=syn_edges_R,
        ax=ax,
        edge_color="purple",
        width=1.0,
        alpha=0.75
    )


    # ========================================================
    # Nodes
    # ========================================================

    # Normal external boundary
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(
            normal_neighbors
        ),
        node_color="lightgray",
        node_size=9,
        edgecolors="gray",
        linewidths=0.15,
        ax=ax
    )


    # U_A
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(
            UA
        ),
        node_color="red",
        node_size=23,
        edgecolors="black",
        linewidths=0.25,
        ax=ax
    )


    # U_B
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(
            UB
        ),
        node_color="dodgerblue",
        node_size=23,
        edgecolors="black",
        linewidths=0.25,
        ax=ax
    )


    # R
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(
            R
        ),
        node_color="purple",
        node_size=23,
        edgecolors="black",
        linewidths=0.25,
        ax=ax
    )


    # Structurally synergistic external nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(
            synergy_neighbors
        ),
        node_color="limegreen",
        node_size=30,
        edgecolors="black",
        linewidths=0.70,
        ax=ax
    )


    # ========================================================
    # Statistics
    # ========================================================

    rho_syn = realization[
        "boundary_concentration"
    ]

    ax.text(
        0.03,
        0.04,
        (
            rf"$\rho_{{syn}}={rho_syn:.3f}$"
            "\n"
            rf"$N_{{syn}}={len(synergy_neighbors)}$"
        ),
        transform=ax.transAxes,
        fontsize=7,
        ha="left",
        va="bottom"
    )

    ax.axis(
        "off"
    )


# ============================================================
# Plot full 3 x 3 Figure S10
# ============================================================

def plot_representatives(
    selection_df
):

    fig, axes = plt.subplots(
        len(P_VALUES),
        len(OV_VALUES),
        figsize=(
            9,
            9
        )
    )


    # --------------------------------------------------------
    # Column titles
    # --------------------------------------------------------

    column_titles = [
        "Low overlap",
        "Intermediate overlap",
        "High overlap"
    ]


    for col, ov in enumerate(
        OV_VALUES
    ):

        axes[
            0,
            col
        ].set_title(
            column_titles[col]
            + "\n"
            + rf"$D_{{ov}}={ov / SUB_NETWORK_SIZE:.2f}$",
            fontsize=11
        )


    # --------------------------------------------------------
    # Row titles
    # --------------------------------------------------------

    row_titles = [
        "Weak rewiring",
        "Moderate rewiring",
        "Strong rewiring"
    ]


    # --------------------------------------------------------
    # Regenerate the selected realizations
    # --------------------------------------------------------

    for row, p in enumerate(
        P_VALUES
    ):

        for col, ov in enumerate(
            OV_VALUES
        ):

            selected = selection_df[
                np.isclose(
                    selection_df["p"],
                    p
                )
                & (
                    selection_df["ov"]
                    == ov
                )
            ].iloc[0]


            seed = int(
                selected[
                    "selected_seed"
                ]
            )


            realization = (
                generate_realization(
                    p=p,
                    ov=ov,
                    seed=seed
                )
            )


            draw_snapshot(
                axes[
                    row,
                    col
                ],
                realization,
                p=p,
                ov=ov,
                seed=seed
            )


        # ----------------------------------------------------
        # Row label
        # ----------------------------------------------------

        p_text = (
            rf"$p={p:.0e}$"
            if p < 0.1
            else rf"$p={p:g}$"
        )

        axes[
            row,
            0
        ].text(
            -0.18,
            0.5,
            row_titles[row]
            + "\n"
            + p_text,
            transform=axes[
                row,
                0
            ].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=10
        )


    # ========================================================
    # Global legend
    # ========================================================

    legend_handles = [

        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="red",
            markeredgecolor="black",
            markersize=7,
            label=r"$U_A$"
        ),

        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="dodgerblue",
            markeredgecolor="black",
            markersize=7,
            label=r"$U_B$"
        ),

        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="purple",
            markeredgecolor="black",
            markersize=7,
            label=r"$R$"
        ),

        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="lightgray",
            markeredgecolor="gray",
            markersize=7,
            label="External boundary"
        ),

        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="limegreen",
            markeredgecolor="black",
            markersize=8,
            label="Synergistic external node"
        )
    ]


    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            0.985
        ),
        ncol=5,
        frameon=False,
        fontsize=9
    )


    plt.subplots_adjust(
        left=0.10,
        right=0.98,
        top=0.91,
        bottom=0.04,
        wspace=0.06,
        hspace=0.10
    )


    # ========================================================
    # Save
    # ========================================================

    pdf_path = (
        FIGURE_DIR
        / "Fig.S10.pdf"
    )



    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )


    plt.close(
        fig
    )


    print(
        f"Saved: {pdf_path}"
    )



# ============================================================
# Main
# ============================================================

if __name__ == "__main__":


    # ========================================================
    # Step 1.
    # Generate candidate realizations
    # ========================================================

    tasks = make_candidate_tasks()

    print(
        f"Generating "
        f"{len(tasks)} "
        f"candidate realizations..."
    )


    with Pool(
        processes=N_PROCESSES
    ) as pool:

        results = list(
            pool.imap_unordered(
                measure_candidate,
                tasks
            )
        )


    candidate_df = pd.DataFrame(
        results
    )


    candidate_path = (
        DATA_DIR
        / "candidate_boundary_statistics.csv"
    )


    candidate_df.to_csv(
        candidate_path,
        index=False
    )


    print(
        f"Saved candidate statistics: "
        f"{candidate_path}"
    )


    # ========================================================
    # Step 2.
    # Select median representatives
    # ========================================================

    selection_df = (
        select_median_representatives(
            candidate_df
        )
    )


    selection_path = (
        DATA_DIR
        / "selected_representative_realizations.csv"
    )


    selection_df.to_csv(
        selection_path,
        index=False
    )


    print(
        "\nSelected realizations:"
    )

    print(
        selection_df
    )


    # ========================================================
    # Step 3.
    # Regenerate selected realizations and draw Figure S10
    # ========================================================


    plot_representatives(
        selection_df
    )