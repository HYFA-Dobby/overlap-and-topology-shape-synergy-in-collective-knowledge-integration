import os
import numpy as np
import pandas as pd
import networkx as nx

from multiprocessing import Pool, cpu_count
from collections import Counter

from src import subnetwork as sn
from src import integnetwork as ig


"""
Simulation of realized external synergy through first-exit random walks.

Each discovered external node is classified according to whether it is
connected to the individual-specific regions U_1, ..., U_n and the
shared region R.

The connection pattern is represented as

    [U_1, U_2, ..., U_n, R]

where 1 indicates at least one background-network edge to the
corresponding region.

Strict external synergy corresponds to

    [1, 1, ..., 1, 1].

Two simulation modes are available:

    single_pw:
        Calculate discovery patterns as a function of D_ov
        for a fixed p_w.

    matrix:
        Calculate strict external synergy over the full
        D_ov x p_w parameter space.
"""


# ============================================================
# 1. One first-exit random-walk discovery
# ============================================================

def one_discovery(G, p_w, size, ov):

    N_agent = len(size)
    nodes = np.array(G.nodes())

    # --------------------------------------------------------
    # Generate individual knowledge networks
    # --------------------------------------------------------
    while True:

        base_node = np.random.choice(nodes)

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

    # --------------------------------------------------------
    # Integrated network
    # --------------------------------------------------------
    integ_network = ig.generate_integrated_network(
        G,
        *sub_networks
    )

    integ_nodes = set(
        integ_network.nodes()
    )

    # --------------------------------------------------------
    # Shared and individual-specific regions
    # --------------------------------------------------------
    sub_sets = [
        set(net.nodes())
        for net in sub_networks
    ]

    # Shared by all individuals
    redundant_nodes = set.intersection(
        *sub_sets
    )

    # Individual-specific regions
    unique_sets = [
        s - redundant_nodes
        for s in sub_sets
    ]

    # --------------------------------------------------------
    # Random-walk starting point
    # --------------------------------------------------------
    current = np.random.choice(
        list(integ_nodes)
    )

    # ========================================================
    # First-exit random walk
    # ========================================================
    while True:

        whole_neighbor = set(
            G.neighbors(current)
        )

        known_neighbor = set(
            integ_network.neighbors(current)
        )

        # Unknown nodes adjacent to the current node
        external_neighbor = (
            whole_neighbor - integ_nodes
        )

        # The walker can move along known integrated-network
        # edges or exit to an adjacent unknown node.
        candidates = list(
            known_neighbor
            | external_neighbor
        )

        # Safety for an isolated state
        if len(candidates) == 0:

            current = np.random.choice(
                list(integ_nodes)
            )

            continue

        next_node = np.random.choice(
            candidates
        )

        # ====================================================
        # First external node reached
        # ====================================================
        if next_node in external_neighbor:

            # Background-network connections from the
            # discovered external node to the integrated network
            internal_neighbors = {
                node
                for node in G.neighbors(next_node)
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

            # Connection pattern:
            #
            # [U1, U2, ..., Un, R]
            connect_vector = tuple(
                connect_unique
                + [connect_redundant]
            )

            return connect_vector

        current = next_node


# ============================================================
# 2. One parameter condition
#    fixed (p_w, D_ov, group size)
# ============================================================

def task_external(args):

    (
        network_type,
        p_w,
        size,
        ov,
        N_iteration,
        N_repeat
    ) = args

    N_agent = len(size)

    print(
        f"Calculation start | "
        f"Network : {network_type} | "
        f"Agents : {N_agent} | "
        f"D_ov : {ov / size[0]:.3f} | "
        f"Pw : {p_w:.3f}"
    )

    # Strict external synergy:
    #
    # [U1, U2, ..., Un, R]
    # [ 1,  1, ...,  1, 1]
    strict_pattern = tuple(
        [True] * (N_agent + 1)
    )

    strict_counts = []
    repeat_counters = []

    # ========================================================
    # Repeated background-network realizations
    # ========================================================
    for repeat in range(N_repeat):

        # ----------------------------------------------------
        # Generate one independent background network
        # for each repeat
        # ----------------------------------------------------
        if network_type == "random":

            G = nx.fast_gnp_random_graph(
                10**5,
                20 / (10**5 - 1)
            )

        elif network_type == "scale_free":

            G = nx.barabasi_albert_graph(
                10**5,
                10
            )

        elif network_type == "modular":

            block_sizes = [10**4] * 10

            p_in = 0.00182
            p_out = 0.000022

            probabilities = [
                [
                    p_in if i == j else p_out
                    for j in range(
                        len(block_sizes)
                    )
                ]
                for i in range(
                    len(block_sizes)
                )
            ]

            G = nx.stochastic_block_model(
                block_sizes,
                probabilities
            )

        elif network_type == "small_world":

            G = nx.watts_strogatz_graph(
                10**5,
                20,
                10**(-4)
            )

        else:

            raise ValueError(
                f"Unknown network type : "
                f"{network_type}"
            )

        # ----------------------------------------------------
        # One repeat consists of N_iteration discoveries
        # on the same background-network realization.
        # ----------------------------------------------------
        pattern_count = Counter()

        for _ in range(N_iteration):

            pattern = one_discovery(
                G,
                p_w,
                size,
                ov
            )

            pattern_count[
                pattern
            ] += 1

        repeat_counters.append(
            pattern_count
        )

        strict_counts.append(
            pattern_count.get(
                strict_pattern,
                0
            )
        )

    # ========================================================
    # Strict external synergy
    # ========================================================
    strict_counts = np.asarray(
        strict_counts,
        dtype=float
    )

    strict_mean = (
        strict_counts.mean()
    )

    strict_sem = (
        strict_counts.std(ddof=1)
        / np.sqrt(N_repeat)
    )


    # ========================================================
    # Complete connection-pattern classification
    # ========================================================
    all_patterns = set()

    for counter in repeat_counters:

        all_patterns.update(
            counter.keys()
        )

    pattern_summary = []

    for pattern in sorted(
        all_patterns
    ):

        values = np.array([
            counter.get(
                pattern,
                0
            )
            for counter
            in repeat_counters
        ])

        mean_count = (
            values.mean()
        )

        sem_count = (
            values.std(ddof=1)
            / np.sqrt(N_repeat)
        )

        pattern_summary.append({

            "pattern":
                "".join(
                    "1" if x else "0"
                    for x in pattern
                ),

            "mean_count":
                mean_count,

            "sem_count":
                sem_count
        })

    print(
        f"Calculation end | "
        f"Network : {network_type} | "
        f"Agents : {N_agent} | "
        f"D_ov : {ov / size[0]:.3f} | "
        f"Pw : {p_w:.3f} | "
        f"Syn : {strict_mean:.3f}"
    )

    return {

        "p_w":
            p_w,

        "ov":
            ov,

        "dov":
            ov / size[0],

        "strict_mean":
            strict_mean,

        "strict_sem":
            strict_sem,

        "pattern_summary":
            pattern_summary
    }


# ============================================================
# 3. Main simulation
# ============================================================

def generalized_external_synergy_simulation():

    # ========================================================
    # Background network
    # ========================================================
    network_type = "random"
    # network_type = "scale_free"
    # network_type = "modular"
    # network_type = "small_world"

    # ========================================================
    # Simulation mode
    # ========================================================
    simulation_mode = "single_pw"
    # simulation_mode = "matrix"

    # ========================================================
    # Group parameters
    # ========================================================
    N_agents = 2

    subnetwork_size = 100

    sub_sizes = (
        [subnetwork_size]
        * N_agents
    )

    # ========================================================
    # Simulation parameters
    # ========================================================

    # Number of external-node discoveries in one repeat
    N_iteration = 100

    # Number of independent repeats
    N_repeat = 100

    # Number of D_ov points
    ov_size = 25

    ov_node = np.unique(
        np.round(
            np.linspace(
                1,
                subnetwork_size,
                ov_size
            )
        ).astype(int)
    )

    # ========================================================
    # Network label
    # ========================================================
    network_label = {

        "random":
            "ER",

        "scale_free":
            "BA",

        "modular":
            "SBM",

        "small_world":
            "WS"

    }[network_type]

    # ========================================================
    # Output directory
    # ========================================================
    output_dir = (
        "./results/external_synergy/"
        f"{network_type}/"
        f"{N_agents}agent"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    network_label = {
        "random": "ER",
        "scale_free": "BA",
        "modular": "SBM",
        "small_world": "WS"
        }[network_type]

    # ========================================================
    # Mode 1
    #
    # Fixed p_w:
    # discovery patterns as a function of D_ov
    #
    # Used for Fig. 3.
    # ========================================================
    if simulation_mode == "single_pw":

        prob_wire = 0.5

        # ----------------------------------------------------
        # One task = one D_ov condition
        # ----------------------------------------------------
        task_args = [
            (
                network_type,
                prob_wire,
                sub_sizes,
                ov,
                N_iteration,
                N_repeat
            )
            for ov in ov_node
        ]

        with Pool(
            processes=cpu_count()
        ) as pool:

            results = pool.map(
                task_external,
                task_args
            )

        # ====================================================
        # Convert results to the original Fig. 3 CSV format
        # ====================================================
        summary_results = []

        for res in results:

            for patt in res["pattern_summary"]:

                summary_results.append({

                    "ov":
                        res["ov"],

                    "pattern":
                        patt["pattern"],

                    "mean":
                        patt["mean_count"],

                    "sem":
                        patt["sem_count"]
                })

        # ====================================================
        # Save as CSV
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
            f"{network_label}_syn_result_{N_agents}agent.csv"
        )

        df.to_csv(
            path,
            index=False,
            encoding="utf-8"
        )

    # ========================================================
    # Mode 2
    #
    # D_ov × p_w matrix
    #
    # Used for the external-synergy heatmap in SI Fig. S8.
    # ========================================================
    elif simulation_mode == "matrix":

        pw_size = 10
        ov_size_matrix = 10

        prob_wire = np.linspace(
            0,
            1,
            pw_size
        )

        ov_node_matrix = np.unique(
            np.round(
                np.linspace(
                    1,
                    subnetwork_size,
                    ov_size_matrix
                )
            ).astype(int)
        )

        # ----------------------------------------------------
        # One multiprocessing task =
        # one (p_w, D_ov) heatmap cell
        # ----------------------------------------------------
        task_args = [
            (
                network_type,
                p_w,
                sub_sizes,
                ov,
                N_iteration,
                N_repeat
            )
            for ov in ov_node_matrix
            for p_w in prob_wire
        ]

        with Pool(
            processes=cpu_count()
        ) as pool:

            results = pool.map(
                task_external,
                task_args
            )

        # ====================================================
        # Mean / SEM matrices
        #
        # rows    = D_ov
        # columns = p_w
        # ====================================================

        strict_mean_matrix = []
        strict_sem_matrix = []

        result_count = 0

        for ov in ov_node_matrix:

            mean_row = []
            sem_row = []

            for p_w in prob_wire:

                res = results[result_count]

                mean_row.append(
                    res["strict_mean"]
                )

                sem_row.append(
                    res["strict_sem"]
                )

                result_count += 1

            strict_mean_matrix.append(
                mean_row
            )

            strict_sem_matrix.append(
                sem_row
            )

        # ====================================================
        # DataFrame
        # ====================================================

        columns = [
            str(p_w)
            for p_w in prob_wire
        ]

        index = [
            ov / subnetwork_size
            for ov in ov_node_matrix
        ]

        mean_df = pd.DataFrame(
            strict_mean_matrix,
            index=index,
            columns=columns
        )

        sem_df = pd.DataFrame(
            strict_sem_matrix,
            index=index,
            columns=columns
        )

        mean_df.index.name = "dov"
        sem_df.index.name = "dov"

        # ====================================================
        # Save only the two heatmap CSV files
        # ====================================================

        mean_path = os.path.join(
            output_dir,
            f"{network_label}_mean_"
            f"{N_agents}p_heatmap.csv"
        )

        sem_path = os.path.join(
            output_dir,
            f"{network_label}_sem_"
            f"{N_agents}p_heatmap.csv"
        )

        mean_df.to_csv(
            mean_path,
            encoding="utf-8"
        )

        sem_df.to_csv(
            sem_path,
            encoding="utf-8"
        )

    else:

        raise ValueError(
            "simulation_mode must be "
            "'single_pw' or 'matrix'"
        )


if __name__ == "__main__":

    generalized_external_synergy_simulation()