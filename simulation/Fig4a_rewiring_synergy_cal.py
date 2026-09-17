import os
import numpy as np
import pandas as pd
import networkx as nx

from multiprocessing import Pool, cpu_count
from collections import Counter

from src import subnetwork as sn
from src import integnetwork as ig


"""
First-exit random-walk simulation for Fig. 4A.

The Watts-Strogatz rewiring probability p is varied while p_w is fixed.

For each rewiring probability, a separate CSV file is generated:

    syn_result_{len(size)}agent_ps{p_s}.csv

Each CSV contains:
    ov, pattern, mean, sem
"""


# ============================================================
# One first-exit discovery
# ============================================================

def task(args):

    p_w, p_s, size, ov, num = args
    N_agent = len(size)

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
    # Shared / individual-specific regions
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
    # First-exit random walk
    # ========================================================

    current = np.random.choice(
        list(integ_nodes)
    )

    pattern_count = Counter()

    while True:

        whole_neighbor = set(
            G.neighbors(current)
        )

        known_neighbor = set(
            integ_network.neighbors(current)
        )

        external_neighbor = {
            node
            for node in whole_neighbor
            if node not in integ_nodes
        }

        candidates = list(
            known_neighbor
            | external_neighbor
        )

        next_node = np.random.choice(
            candidates
        )

        # ----------------------------------------------------
        # First external node reached
        # ----------------------------------------------------

        if next_node in external_neighbor:

            internal_neighbors = {
                node
                for node in G.neighbors(
                    next_node
                )
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

            # [U1, U2, ..., Un, R]
            connect_vector = tuple(
                connect_unique
                + [connect_redundant]
            )

            pattern_count[
                connect_vector
            ] += 1

            return pattern_count

        current = next_node


# ============================================================
# Fig. 4A calculation
# ============================================================

def WS_REWIRING_RW_CAL():

    # ========================================================
    # Knowledge-network parameters
    # ========================================================

    size = [
        100,
        100
    ]

    # Fig. 4 uses p_w = 0.5
    p_w = 0.5

    # ========================================================
    # Sampling
    # ========================================================

    # discoveries within one repeat
    N_iteration = 100

    # repeated ensembles
    N_repeat = 100

    # ========================================================
    # D_ov
    # ========================================================
    ov_size = 25

    ov_node = np.unique(
        np.round(
            np.linspace(
                1,
                size[0],
                ov_size
            )
        ).astype(int)
    )

    # ========================================================
    # Watts-Strogatz rewiring probabilities
    # ========================================================

    prob_list = np.concatenate(
        (
            [0.0],
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
        "WS_rewiring/random_walk"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # ========================================================
    # Rewiring probability
    # ========================================================

    for p_s in prob_list:

        summary_results = []

        # ====================================================
        # Overlap
        # ====================================================

        for ov in ov_node:

            repeat_results = []

            # ================================================
            # Repeated ensembles
            # ================================================

            for repeat in range(
                N_repeat
            ):

                print(
                    f"p={p_s} | "
                    f"ov={ov} | "
                    f"repeat={repeat}"
                )

                # --------------------------------------------
                # 100 independent discoveries
                # --------------------------------------------

                task_args = [
                    (
                        p_w,
                        p_s,
                        size,
                        ov,
                        num
                    )
                    for num in range(
                        N_iteration
                    )
                ]

                with Pool(
                    processes=cpu_count()
                ) as pool:

                    results = pool.map(
                        task,
                        task_args
                    )

                # --------------------------------------------
                # Count connection patterns
                # --------------------------------------------

                total_pattern = Counter()

                for counter in results:

                    total_pattern += counter

                repeat_results.append(
                    total_pattern
                )

            # ================================================
            # All patterns observed for this D_ov
            # ================================================

            all_patterns = set()

            for counter in repeat_results:

                all_patterns.update(
                    counter.keys()
                )

            # ================================================
            # Mean / SEM across repeats
            # ================================================

            for pattern in sorted(
                all_patterns
            ):

                values = np.array([
                    counter.get(
                        pattern,
                        0
                    )
                    for counter
                    in repeat_results
                ])

                summary_results.append({

                    "ov":
                        ov,

                    "pattern":
                        "".join(
                            "1" if x else "0"
                            for x in pattern
                        ),

                    "mean":
                        values.mean(),

                    "sem":
                        (
                            values.std(ddof=1)
                            / np.sqrt(N_repeat)
                        )
                })

            print(
                f"Finished | "
                f"p={p_s} | "
                f"ov={ov}"
            )

        # ====================================================
        # Save one CSV for each rewiring probability
        # ====================================================

        df = pd.DataFrame(
            summary_results,
            columns=[
                "ov",
                "pattern",
                "mean",
                "sem"
            ]
        )

        path = os.path.join(
            output_dir,
            (
                f"syn_result_{len(size)}agent_ps{p_s}.csv"
            )
        )

        df.to_csv(
            path,
            index=False,
            encoding="utf-8"
        )

        print(
            f"\nFinished rewiring p = {p_s}\n"
        )


if __name__ == "__main__":

    WS_REWIRING_RW_CAL()