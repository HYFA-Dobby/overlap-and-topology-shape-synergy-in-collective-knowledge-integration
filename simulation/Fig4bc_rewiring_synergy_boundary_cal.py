import os
import numpy as np
import pandas as pd
import networkx as nx

from multiprocessing import Pool, cpu_count

from src import subnetwork as sn
from src import integnetwork as ig


"""
Boundary synergy calculation for Fig. 4B and 4C.

Structural external-synergy analysis across
Watts-Strogatz rewiring probabilities.

For each rewiring probability p_s, a separate CSV file is generated.

Each CSV contains:
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
"""


# ============================================================
# One network realization
# ============================================================

def task(args):

    p_w, p_s, size, ov, num = args

    N_agent = len(size)

    print(
        f"Calculation start | "
        f"p_s : {p_s} | "
        f"ov : {ov} | "
        f"num : {num}"
    )

    # ========================================================
    # Background network
    # ========================================================

    G = nx.watts_strogatz_graph(
        n=10**5,
        k=20,
        p=p_s
    )

    nodes = np.array(
        G.nodes()
    )

    # ========================================================
    # Individual knowledge networks
    # ========================================================

    while True:

        base_node = np.random.choice(
            nodes
        )

        sub_nodes, sub_networks = sn.generate_subnetworks(
            size,
            G,
            base_node,
            ov,
            p_w
        )

        if all(
            len(sub_nodes[i]) == size[i]
            for i in range(N_agent)
        ):
            break

    # ========================================================
    # Integrated network
    # ========================================================

    integ_network = ig.generate_integrated_network(
        G,
        *sub_networks
    )

    integ_nodes = set(
        integ_network.nodes()
    )

    # ========================================================
    # Shared and individual-specific regions
    # ========================================================

    sub_sets = [
        set(net.nodes())
        for net in sub_networks
    ]

    redundant_nodes = set.intersection(
        *sub_sets
    )

    unique_sets = [
        s - redundant_nodes
        for s in sub_sets
    ]

    # ========================================================
    # External boundary
    #
    # Unknown nodes adjacent to at least one node
    # in the integrated knowledge network
    # ========================================================

    outside_nodes = set()

    for node in integ_nodes:

        outside_nodes.update(
            neighbor
            for neighbor in G.neighbors(node)
            if neighbor not in integ_nodes
        )

    outside_count = len(
        outside_nodes
    )

    # ========================================================
    # Structurally synergistic external nodes
    #
    # A node is synergistic when it has at least one
    # connection to every individual-specific region
    # and to the shared region.
    # ========================================================

    syn_count = 0

    for external_node in outside_nodes:

        internal_neighbors = {
            node
            for node in G.neighbors(external_node)
            if node in integ_nodes
        }

        connect_unique = [
            bool(
                internal_neighbors
                & unique_region
            )
            for unique_region
            in unique_sets
        ]

        connect_redundant = bool(
            internal_neighbors
            & redundant_nodes
        )

        if (
            all(connect_unique)
            and connect_redundant
        ):
            syn_count += 1

    # ========================================================
    # Boundary concentration
    # ========================================================

    if outside_count > 0:

        syn_ratio = (
            syn_count
            / outside_count
        )

    else:

        syn_ratio = 0.0

    return (
        outside_count,
        syn_count,
        syn_ratio
    )


# ============================================================
# Main calculation
# ============================================================

def WS_REWIRING_CAL():

    # ========================================================
    # Simulation parameters
    # ========================================================

    size = [
        100,
        100
    ]

    p_w = 0.5

    # Number of independent realizations
    N_iteration = 500

    # ========================================================
    # Overlap
    # ========================================================
    ov_size = 40

    ov_node = np.round(
        np.linspace(
            1,
            size[0],
            ov_size
        )
    ).astype(int)

    # ========================================================
    # Watts-Strogatz rewiring probabilities
    # ========================================================

    prob_list = np.concatenate(
        (
            [0],
            np.logspace(
                -4,
                0,
                10
            )
        )
    )

    # ========================================================
    # Output
    # ========================================================

    output_dir = (
        "./results/external_synergy/"
        "WS_rewiring/boundary"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # ========================================================
    # Rewiring probability
    # ========================================================

    for p_s in prob_list:

        rows = []

        # ====================================================
        # Overlap
        # ====================================================

        for ov in ov_node:

            print(
                f"\n"
                f"p_s = {p_s} | "
                f"D_ov = {ov / size[0]:.3f}"
            )

            # ------------------------------------------------
            # Each task corresponds to one independent
            # background-network realization
            # ------------------------------------------------

            task_args = [
                (
                    p_w,
                    p_s,
                    size,
                    ov,
                    num
                )
                for num in range(N_iteration)
            ]

            with Pool(
                processes=cpu_count()
            ) as pool:

                results = pool.map(
                    task,
                    task_args
                )

            # =================================================
            # Results
            # =================================================

            outside_result = np.array([
                r[0]
                for r in results
            ])

            syn_count_result = np.array([
                r[1]
                for r in results
            ])

            syn_ratio_result = np.array([
                r[2]
                for r in results
            ])

            # ------------------------------------------------
            # Mean
            # ------------------------------------------------

            outside_mean = np.mean(
                outside_result
            )

            syn_count_mean = np.mean(
                syn_count_result
            )

            syn_ratio_mean = np.mean(
                syn_ratio_result
            )

            # ------------------------------------------------
            # SEM
            # ------------------------------------------------

            outside_sem = (
                np.std(outside_result)
                / np.sqrt(N_iteration)
            )

            syn_count_sem = (
                np.std(syn_count_result)
                / np.sqrt(N_iteration)
            )

            syn_ratio_sem = (
                np.std(syn_ratio_result)
                / np.sqrt(N_iteration)
            )

            rows.append({

                "p_w":
                    p_w,

                "p_s":
                    p_s,

                "ov":
                    ov,

                "D_ov":
                    ov / size[0],

                "outside_mean":
                    outside_mean,

                "outside_sem":
                    outside_sem,

                "syn_count_mean":
                    syn_count_mean,

                "syn_count_sem":
                    syn_count_sem,

                "syn_ratio_mean":
                    syn_ratio_mean,

                "syn_ratio_sem":
                    syn_ratio_sem
            })

        # ====================================================
        # Save one CSV for each rewiring probability
        # ====================================================

        df = pd.DataFrame(
            rows,
            columns=[
                "p_w",
                "p_s",
                "ov",
                "D_ov",
                "outside_mean",
                "outside_sem",
                "syn_count_mean",
                "syn_count_sem",
                "syn_ratio_mean",
                "syn_ratio_sem"
            ]
        )

        path = os.path.join(
            output_dir,
            f"syn_result_ps{p_s}_boundary.csv"
        )

        df.to_csv(
            path,
            index=False,
            encoding="utf-8"
        )

        print(
            f"\nFinished p_s = {p_s}"
        )


if __name__ == "__main__":

    WS_REWIRING_CAL()